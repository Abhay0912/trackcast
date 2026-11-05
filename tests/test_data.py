"""Tests for trackcast.data on small synthetic frames (never the raw CSVs)."""
import numpy as np
import pandas as pd
import pytest

from trackcast import data as D

# Train 101: 5 stops, stops 4-5 arrive on day 2.  Train 202: 3 stops, single day.
SCHEDULE_RAW = pd.DataFrame({
    "train_no":     [101, 101, 101, 101, 101, 202, 202, 202],
    "station_no":   [1, 2, 3, 4, 5, 1, 2, 3],
    "station_name": ["AAA", "BBB", "CCC", "DDD", "EEE", "AAA", "FFF", "GGG"],
    "distance_from_origin": [0, 100, 200, 400, 500, 0, 50, 90],
    "arrival_day":   [1, 1, 1, 2, 2, 1, 1, 1],
    "arrival_time":  [None, "20:00", "23:00", "01:30", "06:00", None, "10:00", "11:00"],
    "departure_day": [1, 1, 1, 2, None, 1, 1, None],
    "departure_time": ["18:00", "20:10", "23:10", "01:40", None, "09:00", "10:05", None],
})


def interim(rows):
    """rows: (train_no, 'YYYY-MM-DD', station_no, station_name, delay) -> pass-1 style frame."""
    df = pd.DataFrame(rows, columns=["train_no", "date", "station_no", "station_name", "delay"])
    return df.astype({"train_no": "int32", "date": "datetime64[s]", "station_no": "int16",
                      "station_name": "string", "delay": "float32"})


@pytest.fixture
def lookups():
    stats = D.QualityStats()
    schedule = D.prepare_schedule(SCHEDULE_RAW, stats)
    trains = D.prepare_lookup(pd.DataFrame({"train_no": [101, 101, 202], "train_name": ["A Exp", "dup", "B Pass"],
                                            "type_code": ["EXP", "EXP", "PASS"]}),
                              "train_no", ["train_name", "type_code"], stats, "trains")
    stations = D.prepare_lookup(pd.DataFrame({"station_name": ["AAA", "BBB", "BBB", "CCC"],
                                              "station_full_name": ["Alpha", "Beta", "Beta2", "Gamma"],
                                              "station_zone": ["SCR", "SCR", "SCR", "CR"]}),
                                "station_name", ["station_full_name", "station_zone"], stats, "stations")
    return stats, schedule, trains, stations


def test_schedule_offsets_and_flags(lookups):
    stats, s, _, _ = lookups
    t = s[s.train_no == 101].set_index("station_no")
    assert np.isnan(t.loc[1, "sched_arr_offset_min"])
    assert t.loc[1, "sched_dep_offset_min"] == 18 * 60
    assert t.loc[4, "sched_arr_offset_min"] == 1440 + 90          # day offset applied
    assert t.loc[1, "is_origin"] and t.loc[5, "is_terminus"] and not t.loc[3, "is_terminus"]
    assert stats.counts["schedule_offset_violation_trains"] == 0
    assert stats.counts["trains_duplicate_keys"] == 1 and stats.counts["stations_duplicate_keys"] == 1


def test_clean_month_rules(lookups):
    stats, schedule, trains, stations = lookups
    raw = interim([
        (101, "2024-01-01", 1, "AAA", 10), (101, "2024-01-01", 2, "BBB", -3),
        (101, "2024-01-01", 2, "BBB", -3),                       # duplicate
        (101, "2024-01-01", 3, "CCC", 20),
        (101, "2024-01-01", 7, "DDD", 2000),                      # renumbered (schedule: 4), > cap
        (101, "2024-01-01", 8, "EEE", np.nan),                    # renumbered (schedule: 5)
        (101, "2024-01-01", 4, "ZZZ", 5),                         # station not in this train's schedule
        (202, "2024-01-01", 1, "AAA", 5), (202, "2024-01-01", 2, "FFF", np.nan),  # only 1 observed -> dropped
        (202, "2024-01-01", 3, "GGG", np.nan),
    ])
    out = D.clean_month(raw, schedule, trains, stations, stats)
    c = stats.counts

    assert list(out.columns) == D.CLEAN_COLUMNS
    assert len(out) == 5 and set(out.train_no) == {101}          # no row explosion from dup lookups
    assert c["dropped_duplicates"] == 1 and c["dropped_no_schedule"] == 1
    assert c["dropped_short_runs"] == 1 and c["dropped_short_run_rows"] == 3
    assert c["station_no_renumbered"] == 2 and c["stop_order_inversions"] == 0
    assert c["delay_above_cap_set_nan"] == 1
    assert c["negative_clipped"] == 1

    o = out.set_index("station_no")
    assert o.loc[2, "delay_raw"] == -3 and o.loc[2, "delay"] == 0  # clipped, raw kept
    assert np.isnan(o.loc[4, "delay"]) and o.loc[4, "delay_missing"]  # cap -> NaN
    assert np.isnan(o.loc[5, "delay"]) and o.loc[5, "delay_missing"]  # NaN kept, not 0
    assert list(out.station_no) == [1, 2, 3, 4, 5]                    # schedule numbering + order
    assert o.loc[4, "station_name"] == "DDD"
    assert o.loc[4, "sched_arr_ts"] == pd.Timestamp("2024-01-02 01:30")
    assert out.run_id.nunique() == 1
    assert o.loc[2, "train_name"] == "A Exp" and o.loc[2, "station_full_name"] == "Beta"
    assert c["clean_rows"] == 5 and c["clean_missing"] == 2
    D.to_arrow(out)  # casts to the parquet schema without error


def test_sample_runs_whole_and_reproducible():
    sizes = {i: 5 + i % 7 for i in range(200)}
    cand = pd.DataFrame({"run_id": np.repeat(list(sizes), list(sizes.values()))})
    cand["train_no"], cand["date"] = cand.run_id, pd.Timestamp("2024-01-01")
    cand["station_no"] = cand.groupby("run_id").cumcount()

    a = D.sample_runs(cand, 300, np.random.default_rng(0))
    b = D.sample_runs(cand, 300, np.random.default_rng(0))
    assert a.equals(b)
    assert 250 < len(a) <= 300
    got = a.groupby("run_id").size()
    assert (got == pd.Series(sizes).loc[got.index]).all()        # every sampled run is complete


def _runs(dates, delay_by_run, calendar: bool):
    """Cleaned-like frame for train 101; if calendar, day-2 stops are dated date + 1."""
    stats = D.QualityStats()
    sched = D.prepare_schedule(SCHEDULE_RAW[SCHEDULE_RAW.train_no == 101], stats)
    rows = []
    for d, level in zip(dates, delay_by_run):
        for _, st in sched.iterrows():
            shift = pd.Timedelta(days=int(st.stop_day) - 1) if calendar else pd.Timedelta(0)
            rows.append({"train_no": 101, "date": d + shift, "station_no": st.station_no,
                         "arrival_day": st.arrival_day, "departure_day": st.departure_day,
                         "delay_raw": level + st.station_no})
    return pd.DataFrame(rows), sched


@pytest.mark.parametrize("calendar,expected", [(False, D.START_DATE), (True, D.CALENDAR_DATE)])
def test_date_check_weekday(calendar, expected):
    mondays = list(pd.date_range("2024-01-01", periods=10, freq="7D"))   # weekly train
    df, _ = _runs(mondays, [0] * 10, calendar)
    assert D.date_check_weekday(D.weekday_counts(df))["verdict"] == expected


@pytest.mark.parametrize("calendar,expected", [(False, D.START_DATE), (True, D.CALENDAR_DATE)])
def test_date_check_continuity(calendar, expected):
    days = list(pd.date_range("2024-01-01", periods=40, freq="D"))       # daily train
    levels = np.random.default_rng(1).uniform(0, 300, len(days))
    df, sched = _runs(days, levels, calendar)
    same, nxt = D.continuity_pairs(df, D.day_boundaries(sched))
    ps, pn = D.PairStats(), D.PairStats()
    ps.add(same.x, same.y)
    pn.add(nxt.x, nxt.y)
    assert D.date_check_continuity(ps, pn)["verdict"] == expected


def test_combine_verdicts():
    assert D.combine_verdicts(D.START_DATE, D.INCONCLUSIVE) == D.START_DATE
    assert D.combine_verdicts(D.START_DATE, D.CALENDAR_DATE) == D.INCONCLUSIVE
