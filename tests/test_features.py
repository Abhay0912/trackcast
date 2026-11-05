"""Tests for trackcast.features on small synthetic frames (never the raw CSVs)."""
import numpy as np
import pandas as pd
import pytest

from trackcast import data as D
from trackcast import features as F

# Two trains share the segment BBB -> CCC; train 202 is faster on it (40 vs 60 min), so 101 has 20 min slack.
SCHEDULE_RAW = pd.DataFrame({
    "train_no":       [101, 101, 101, 101, 101, 202, 202, 202],
    "station_no":     [1, 2, 3, 4, 5, 1, 2, 3],
    "station_name":   ["AAA", "BBB", "CCC", "DDD", "EEE", "XXX", "BBB", "CCC"],
    "distance_from_origin": [0, 100, 200, 300, 400, 0, 50, 150],
    "arrival_day":    [1, 1, 1, 1, 2, 1, 1, 1],
    "arrival_time":   [None, "10:00", "11:05", "12:10", "00:30", None, "10:30", "11:15"],
    "departure_day":  [1, 1, 1, 1, None, 1, 1, None],
    "departure_time": ["08:00", "10:05", "11:10", "12:15", None, "09:00", "10:35", None],
})
ZONES = {"AAA": "NR", "BBB": "NR", "CCC": "SCR", "DDD": "SCR", "EEE": "SCR", "XXX": "SCR"}


@pytest.fixture(scope="module")
def sched():
    s = D.prepare_schedule(SCHEDULE_RAW, D.QualityStats())
    return s, F.schedule_features(s)


def clean_frame(sched_df, runs):
    """runs: {(train_no, 'YYYY-MM-DD'): [delay per scheduled stop, NaN = missing]} -> Phase 1 style frame."""
    rows = []
    for (train, day), delays in runs.items():
        st = sched_df[sched_df.train_no == train].sort_values("station_no")
        for (_, r), d in zip(st.iterrows(), delays):
            rows.append({"train_no": train, "date": pd.Timestamp(day), **r[[
                "station_no", "station_name", "sched_arr_offset_min", "sched_dep_offset_min",
                "distance_from_origin", "is_origin", "is_terminus"]].to_dict(), "delay": d})
    df = pd.DataFrame(rows)
    df["run_id"] = D.make_run_id(df["train_no"], df["date"])
    df["delay_missing"] = df["delay"].isna()
    df["type_code"] = "EXP-TRAINS"
    df["station_zone"] = df["station_name"].map(ZONES)
    return df.astype({"station_no": "int16", "delay": "float32", "is_origin": bool, "is_terminus": bool,
                      "station_name": "category", "type_code": "category", "station_zone": "category"})


def test_schedule_slack(sched):
    _, sf = sched
    t = sf[sf.train_no == 101].set_index("station_no")
    assert t.loc[3, "seg_sched_min"] == 60 and t.loc[3, "run_slack_min"] == 20   # vs train 202's 40 min
    assert t.loc[3, "seg_n_trains"] == 2
    assert t.loc[4, "run_slack_min"] == 0                                         # only train on DDD
    assert t.loc[5, "seg_sched_min"] == 12 * 60 + 15                              # crosses midnight (day 2)
    assert t.loc[3, "seg_speed_kmph"] == pytest.approx(100.0)
    assert np.isnan(t.loc[1, "seg_sched_min"]) and t.loc[1, "cum_slack_min"] == 0
    # dwell at BBB: 101 = 5, 202 = 5 -> no dwell slack; cumulative slack = run slack only
    assert t.loc[3, "cum_slack_min"] == 20 and t.loc[5, "cum_slack_min"] == 20
    assert t.loc[3, "dwell_prev_min"] == 5
    assert t.loc[5, "route_frac"] == 1.0
    # congestion: BBB has 101 at 10:00 and 202 at 10:30 -> 2 trains in hour 10
    assert t.loc[2, "station_trains_hour"] == 2 and t.loc[4, "station_trains_hour"] == 1


def test_lags_skip_missing_and_respect_runs(sched):
    s, sf = sched
    nan = np.nan
    clean = clean_frame(s, {(101, "2025-01-06"): [5, 10, nan, 30, 25],
                            (202, "2025-01-06"): [50, 60, 70]})
    f = F.build_features(clean, sf, horizon=1, model_rows_only=False)
    r = f[f.train_no == 101].set_index("station_no")
    assert np.isnan(r.loc[1, "lag1"])                                   # origin: nothing upstream
    assert r.loc[2, "lag1"] == 5 and np.isnan(r.loc[2, "lag2"]) and r.loc[2, "lag_from_origin"] == 1
    assert r.loc[4, "lag1"] == 10 and r.loc[4, "lag2"] == 5             # stop 3 missing -> skip it, not 0
    assert r.loc[4, "delay_velocity"] == 5 and r.loc[4, "lag_stops_back"] == 2
    assert r.loc[4, "lag_sched_min_back"] == (12 * 60 + 10) - (10 * 60 + 5)
    assert r.loc[4, "slack_since_lag"] == 20                            # BBB -> CCC slack lies in between
    assert r.loc[5, "lag1"] == 30 and r.loc[5, "lag2"] == 10 and r.loc[5, "delay_velocity"] == 20
    assert r.loc[5, "upstream_max_delay"] == 30 and r.loc[5, "n_upstream_obs"] == 3
    q = f[f.train_no == 202].set_index("station_no")
    assert np.isnan(q.loc[1, "lag1"]) and q.loc[2, "lag1"] == 50       # no bleed from train 101's run

    m = F.build_features(clean, sf, horizon=1)                          # model rows only
    assert set(zip(m.train_no, m.station_no)) == {(101, 2), (101, 4), (101, 5), (202, 2), (202, 3)}
    assert list(m.columns) == F.OUTPUT_COLUMNS


def test_horizon(sched):
    s, sf = sched
    clean = clean_frame(s, {(101, "2025-01-06"): [5, 10, 20, 30, 25]})
    f = F.build_features(clean, sf, horizon=2, model_rows_only=False).set_index("station_no")
    assert np.isnan(f.loc[2, "lag1"]) and f.loc[3, "lag1"] == 5 and f.loc[5, "lag1"] == 20
    assert f.loc[5, "lag2"] == 10 and f.loc[5, "lag_stops_back"] == 2
    with pytest.raises(ValueError):
        F.build_features(clean, sf, horizon=0)


@pytest.mark.parametrize("horizon", [1, 2])
def test_no_leakage_from_stop_k_or_later(sched, horizon):
    """Changing delays at stops >= k must leave every feature of stop k unchanged."""
    s, sf = sched
    base = clean_frame(s, {(101, "2025-01-06"): [5, 10, np.nan, 30, 25], (202, "2025-01-06"): [1, 2, 3]})
    feats0 = F.build_features(base, sf, horizon, model_rows_only=False)
    cols = [c for c in F.OUTPUT_COLUMNS if c not in F.TARGETS]
    rng = np.random.default_rng(0)
    for k in range(1, 6):
        pert = base.copy()
        later = (pert.train_no == 101) & (pert.station_no >= k)
        pert.loc[later, "delay"] = rng.uniform(0, 500, later.sum()).astype("float32")
        pert.loc[later & (pert.station_no == 5), "delay"] = np.nan          # also flip missingness
        feats1 = F.build_features(pert, sf, horizon, model_rows_only=False)
        upto = (feats0.train_no == 101) & (feats0.station_no <= k)
        pd.testing.assert_frame_equal(feats0.loc[upto, cols], feats1.loc[upto, cols])


def test_time_and_season_flags(sched):
    s, sf = sched
    clean = clean_frame(s, {(101, "2025-01-04"): [0, 0, 0, 0, 0],      # Saturday, January
                            (101, "2025-07-07"): [0, 0, 0, 0, 0]})     # Monday, July
    f = F.build_features(clean, sf, model_rows_only=False)
    jan = f[f.date == "2025-01-04"].set_index("station_no")
    jul = f[f.date == "2025-07-07"].set_index("station_no")
    assert jan.loc[2, "fog_season"] and not jan.loc[3, "fog_season"]   # BBB is NR, CCC is SCR
    assert not jan.loc[2, "fog_night"]                                  # 10:00 is after the fog window
    assert jan.loc[2, "is_weekend"] and not jul.loc[2, "is_weekend"]
    assert jul.loc[2, "monsoon"] and not jul.loc[2, "fog_season"]
    # stop 5 arrives 00:30 on day 2 -> calendar Sunday, hour 0.5
    assert jan.loc[5, "sched_event_ts"] == pd.Timestamp("2025-01-05 00:30")
    assert jan.loc[5, "hour_sin"] == pytest.approx(np.sin(2 * np.pi * 0.5 / 24), abs=1e-6)
    assert jan.loc[5, "dow_sin"] == pytest.approx(np.sin(2 * np.pi * 6 / 7), abs=1e-6)


def test_targets(sched):
    s, sf = sched
    clean = clean_frame(s, {(101, "2025-01-06"): [0, 15, 16, 0, 100]})
    f = F.build_features(clean, sf).set_index("station_no")
    assert not f.loc[2, "y_late"] and f.loc[3, "y_late"] and not f.loc[4, "y_late"]
    assert f.loc[5, "y_log"] == pytest.approx(np.log1p(100))
