"""Tests for trackcast.profiles on small synthetic clean frames (never the real data)."""
import numpy as np
import pandas as pd
import pytest

from trackcast import data as D
from trackcast import profiles as P

nan = np.nan
ROUTES = {101: ["AAA", "BBB", "CCC", "DDD"], 202: ["XXX", "BBB", "CCC"]}


def clean_frame(runs: dict) -> pd.DataFrame:
    """runs: {(train_no, 'YYYY-MM-DD'): [delay per scheduled stop]} -> Phase 1-shaped clean frame."""
    rows = []
    for (train, day), delays in runs.items():
        route = ROUTES[train]
        for i, (st, d) in enumerate(zip(route, delays)):
            rows.append({"train_no": train, "date": pd.Timestamp(day), "station_no": i + 1, "station_name": st,
                         "delay": d, "delay_raw": d - 5 if d == d else nan,
                         "sched_arr_offset_min": nan if i == 0 else 480 + 60 * i,
                         "sched_dep_offset_min": nan if i == len(route) - 1 else 485 + 60 * i,
                         "distance_from_origin": 50.0 * i, "is_origin": i == 0, "is_terminus": i == len(route) - 1,
                         "train_name": f"T{train}", "type_code": "EXP-TRAINS",
                         "station_full_name": f"{st} JN", "station_zone": "NR"})
    df = pd.DataFrame(rows)
    df["run_id"] = D.make_run_id(df["train_no"], df["date"])
    return df.astype({"station_name": "category", "type_code": "category", "station_zone": "category",
                      "train_name": "category", "station_full_name": "category", "delay": "float32"})


RUNS = {(101, "2025-01-06"): [10, 30, nan, 20], (101, "2025-01-07"): [0, 5, 40, 50],
        (101, "2025-01-08"): [20, 25, 10, 5], (202, "2025-01-06"): [60, 70, 40], (202, "2025-01-07"): [0, 0, 3]}


@pytest.fixture(scope="module")
def tables():
    return P.merge_partials([P.partial_profiles(clean_frame(RUNS))])


def test_change_skips_missing_stops():
    df = P.prepare(clean_frame({(101, "2025-01-06"): [10, 30, nan, 20]}))
    r = df.set_index("station_no")
    assert np.isnan(r.loc[1, "change"]) and r.loc[2, "change"] == 20
    assert r.loc[4, "prev_delay"] == 30 and r.loc[4, "change"] == -10 and r.loc[4, "stops_back"] == 2
    assert r.loc[4, "prev_station"] == "BBB"


def test_station_profile(tables):
    s = tables["stations"].set_index("station_name")
    # BBB arrivals: 30, 5, 25 (train 101) and 70, 0 (train 202)
    assert s.loc["BBB", "n_obs"] == 5 and s.loc["BBB", "mean_delay"] == pytest.approx(26)
    assert s.loc["BBB", "p50_delay"] == 25 and s.loc["BBB", "p90_delay"] == 70
    assert s.loc["BBB", "late_pct"] == pytest.approx(60)
    # changes at BBB: +20, +5, +5, +10, 0 ; late upstream (prev > 15): 20 -> 25 (+5) and 60 -> 70 (+10)
    assert s.loc["BBB", "mean_change"] == pytest.approx(8)
    assert s.loc["BBB", "recovery_pct_when_late"] == 0
    assert s.loc["BBB", "station_full_name"] == "BBB JN"
    # CCC: 40 (from 5), 10 (from 25: late, recovered), 40 (from 70: late, recovered), 3 (from 0)
    assert s.loc["CCC", "recovery_pct_when_late"] == pytest.approx(100)


def test_segments_only_consecutive_stops(tables):
    seg = tables["segments"].set_index(["from_station", "to_station"])
    assert ("CCC", "DDD") in seg.index and ("BBB", "DDD") not in seg.index   # gap -> not a segment
    assert seg.loc[("BBB", "CCC"), "n_obs"] == 4                              # 101 on day 1 has CCC missing
    assert seg.loc[("BBB", "CCC"), "mean_sched_min"] == 55                    # arr 600 - dep 545


def test_train_profiles(tables):
    tr = tables["trains"].set_index("train_no")
    assert tr.loc[101, "n_runs"] == 3 and tr.loc[101, "mean_origin_delay"] == pytest.approx(10)
    assert tr.loc[101, "mean_terminus_delay"] == pytest.approx(25)            # 20, 50, 5
    assert tr.loc[101, "mean_gained_en_route"] == pytest.approx((10 + 50 - 15) / 3)
    assert tr.loc[202, "origin"] == "XXX" and tr.loc[202, "terminus"] == "CCC"
    ts = tables["train_stops"].set_index(["train_no", "station_no"])
    assert ts.loc[(101, 3), "n_obs"] == 2 and ts.loc[(101, 3), "missing_pct"] == pytest.approx(100 / 3)
    assert ts.loc[(101, 4), "median_delay"] == 20


def test_cascade(tables):
    c = tables["cascade"].set_index(["h", "bucket"])
    # h = 1, initial delay in (60, 180]: only 70 -> 40 (train 202 day 1)
    assert c.loc[(1, 2), "n_pairs"] == 1 and c.loc[(1, 2), "mean_change"] == -30
    assert c.loc[(1, 2), "late_pct_h"] == 100 and c.loc[(1, 2), "lower_pct_h"] == 100


def test_partitioning_is_exact():
    """Processing trains in separate groups must give the same tables as one pass."""
    one = P.merge_partials([P.partial_profiles(clean_frame(RUNS))])
    split = P.merge_partials([P.partial_profiles(clean_frame({k: v for k, v in RUNS.items() if k[0] == t}))
                              for t in (101, 202)])
    for name in P.TABLES:
        a, b = one[name], split[name]
        keys = [c for c in a.columns if a[c].dtype.kind not in "f"]
        a, b = a.sort_values(keys, ignore_index=True), b.sort_values(keys, ignore_index=True)
        pd.testing.assert_frame_equal(a, b[a.columns], check_dtype=False)


def test_sawtooth_station_is_flagged():
    # BBB's recorded delay jumps +50 on arrival and drops -50 on the next segment: an artefact at BBB.
    runs = {(101, f"2025-01-{d:02d}"): [10, 60, 10, 12] for d in range(1, 6)}
    t = P.merge_partials([P.partial_profiles(clean_frame(runs))])
    s = t["stations"].set_index("station_name")
    assert s.loc["BBB", "mean_change"] == 50 and s.loc["BBB", "outbound_mean_change"] == -50
    assert s.loc["BBB", "suspect_timetable"] and not s.loc["CCC", "suspect_timetable"]
    assert s.loc["CCC", "inbound_suspect_share"] == 1.0 and s.loc["DDD", "inbound_suspect_share"] == 0.0
    seg = t["segments"].set_index(["from_station", "to_station"])
    assert seg.loc[("AAA", "BBB"), "touches_suspect"] and not seg.loc[("CCC", "DDD"), "touches_suspect"]


def test_hist_quantiles_match_pandas():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"k": rng.choice(list("abc"), 500), "v": rng.integers(0, 100, 500)})
    q = P.hist_quantiles(df.groupby(["k", "v"]).size())
    for k, g in df.groupby("k"):
        assert q.loc[k, "p50_delay"] == g["v"].quantile(0.5, interpolation="lower")
        assert q.loc[k, "p90_delay"] == np.sort(g["v"])[int(np.ceil(0.9 * len(g))) - 1]
