"""Tests for trackcast.results on small synthetic runs and predictions (never the real data)."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from sklearn.metrics import f1_score

from trackcast import config as C
from trackcast import results as R


def runs_frame(runs: dict[int, list[tuple[str, float]]]) -> pd.DataFrame:
    """run_id -> [(station, delay)] in stop order (station_no 1, 2, ...); NaN = not observed."""
    rows = [{"run_id": rid, "train_no": rid // 100, "station_no": i + 1, "station_name": stn, "delay": d}
            for rid, stops in runs.items() for i, (stn, d) in enumerate(stops)]
    return pd.DataFrame(rows).astype({"station_name": "category", "delay": "float32"})


def test_through_partial_sums():
    df = runs_frame({100: [("A", 10), ("B", 30), ("C", 20), ("D", 50)]})
    s = R.through_partial(df)
    assert set(s.index) == {"B", "C"}                     # origin and terminus have no through traffic
    b, c = s.loc["B"], s.loc["C"]
    assert (b["sum_arr"], b["sum_dep"], b["sum_thr"]) == (20, -10, 10)
    assert (c["sum_arr"], c["sum_dep"], c["sum_thr"]) == (-10, 30, 20)
    assert b["n_ontime_in"] == 1 and b["n_made_late"] == 1          # 10 at A -> 20 at C
    assert c["n_late_in"] == 1 and c["n_recovered"] == 0            # 30 at B -> 50 at D


def test_through_needs_consecutive_observed_stops_in_one_run():
    df = runs_frame({100: [("A", 10), ("B", 20), ("C", np.nan), ("D", 40), ("E", 45)],
                     101: [("D", 0), ("E", 0)]})
    s = R.through_partial(df)
    # B's next stop is missing, D's previous stop is missing, E is the terminus; run 101 never joins run 100.
    assert s.empty


def test_through_drops_triples_touching_a_suspect():
    df = runs_frame({100: [("A", 10), ("B", 30), ("C", 20), ("D", 50), ("E", 50)]})
    s = R.through_partial(df, frozenset({"C"}))
    assert s.empty                         # every triple (A,B,C), (B,C,D), (C,D,E) touches C
    df2 = runs_frame({100: [("A", 10), ("B", 30), ("C", 20), ("D", 50), ("E", 50), ("F", 50)]})
    assert set(R.through_partial(df2, frozenset({"C"})).index) == {"E"}


def test_through_partials_merge_exactly():
    rng = np.random.default_rng(0)
    runs = {t * 100 + r: [(s, float(rng.integers(0, 90))) for s in "ABCDEF"] for t in (1, 2, 3) for r in range(5)}
    df = runs_frame(runs)
    one = R.through_partial(df)
    parts = [R.through_partial(df[df["train_no"].isin(g)]) for g in ([1], [2, 3])]
    pd.testing.assert_frame_equal(R.merge_through(parts).sort_index(), one.sort_index(), check_dtype=False,
                                  check_index_type=False)


def test_through_rates_and_ranking():
    s = pd.DataFrame({"n": [10, 10, 10], "sum_arr": [50.0, -100, 5], "sum_dep": [50.0, 20, 5],
                      "sum_thr": [100.0, -80, 10], "n_late_in": [4, 5, 0], "n_recovered": [1, 5, 0],
                      "n_ontime_in": [6, 5, 10], "n_made_late": [3, 0, 1]},
                     index=pd.Index(["X", "Y", "Z"], name="station_name"))
    r = R.through_rates(s, n_days=5)
    assert r.loc["X", "through_change"] == 10 and r.loc["X", "minutes_per_day"] == 20
    assert r.loc["X", "recovery_pct"] == 25 and r.loc["X", "made_late_pct"] == 50
    assert np.isnan(r.loc["Z", "recovery_pct"])
    st = r.reset_index().assign(station_class=["Junction", "Major station", "Minor halt"])
    adds, absorbs = R.rank_bottlenecks(st, min_through=1)
    assert adds["station_name"].tolist() == ["X"]           # Z adds delay but is a minor halt
    assert absorbs["station_name"].tolist() == ["Y"]


def test_worst_segments():
    seg = pd.DataFrame({"from_station": ["A", "B", "Q"], "to_station": ["C", "C", "C"], "n_obs": [100, 50, 999],
                        "mean_change": [2.0, 10.0, 50.0], "touches_suspect": [False, False, True]})
    w = R.worst_segments(seg, min_pairs=10)
    assert w.loc["C", "approach_from"] == "B" and w.loc["C", "approach_change"] == 10   # 500 > 200; Q suspect
    assert w.loc["A", "exit_to"] == "C"


# --------------------------------------------------------------------------------------------------
def predictions(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lag1 = rng.exponential(40, n)
    delay = np.clip(lag1 + rng.normal(0, 10, n), 0, None)
    pred = np.log1p(np.clip(delay + rng.normal(0, 3, n), 0, None))     # better than persistence (noise 10)
    df = pd.DataFrame({
        "run_id": rng.integers(0, 400, n), "train_no": rng.integers(1, 6, n).astype("int32"),
        "date": pd.Timestamp("2025-12-01") + pd.to_timedelta(rng.integers(0, 20, n), unit="D"),
        "station_name": pd.Categorical(rng.choice(["A", "B", "C", "D"], n)),
        "type_code": pd.Categorical(rng.choice(["EXP-TRAINS", "PASS-TRAINS"], n)),
        "lag1": lag1, "delay": delay, "y_log": np.log1p(delay), "y_late": delay > C.SIGNIFICANT_DELAY_MIN,
        "pred_log": pred, "late_pred": np.expm1(pred) > C.SIGNIFICANT_DELAY_MIN,
        "persistence_log": np.log1p(lag1), "direct_hgb_log": pred})
    df.loc[df["train_no"] == 5, ["delay", "y_log", "lag1", "persistence_log"]] = 0.0   # never late
    df.loc[df["train_no"] == 5, "y_late"] = False
    return df


def test_group_scores_match_direct_computation():
    df = predictions()
    s = R.group_scores(df, df["type_code"], min_rows=10, n_boot=200).set_index("group")
    part = df[df["type_code"] == "EXP-TRAINS"]
    row = s.loc["EXP-TRAINS"]
    assert row["rows"] == len(part) and row["runs"] == part["run_id"].nunique()
    assert row["rmsle_selected"] == pytest.approx(np.sqrt(np.mean((part["y_log"] - part["pred_log"]) ** 2)))
    assert row["mae_persistence"] == pytest.approx(np.mean(np.abs(part["lag1"] - part["delay"])))
    assert row["f1_selected"] == pytest.approx(f1_score(part["y_late"], part["late_pred"]))
    assert row["f1_persistence"] == pytest.approx(f1_score(part["y_late"], part["lag1"] > C.SIGNIFICANT_DELAY_MIN))
    assert row["skill_pct"] == pytest.approx(100 * (1 - row["rmsle_selected"] / row["rmsle_persistence"]))
    assert row["skill_lo"] <= row["skill_pct"] <= row["skill_hi"]


def test_group_scores_min_rows_and_undefined_skill():
    df = predictions()
    s = R.group_scores(df, df["train_no"], min_rows=10, n_boot=50).set_index("group")
    assert "5" in s.index and np.isnan(s.loc["5", "skill_pct"]) and np.isnan(s.loc["5", "skill_lo"])
    assert s.loc["5", "rmsle_gain"] < 0
    few = R.group_scores(df, df["train_no"], min_rows=len(df), n_boot=50)
    assert few.empty


def test_station_groups():
    st = pd.DataFrame({"station_name": ["A", "B", "C", "D"],
                       "station_class": ["Junction", "Major station", "Junction", "Minor halt"]})
    g = R.station_groups(pd.Series(["A", "B", "C", "D", "Z"]), st, st.iloc[[0]], st.iloc[[1]])
    assert g[:4].tolist() == ["Bottlenecks", "Absorbers", "Other junctions and major stations", "Minor halts"]
    assert pd.isna(g[4])


def test_report_and_figures_render():
    df = predictions()
    s = pd.DataFrame({"n": [40, 40, 40], "sum_arr": [50.0, -100, 5], "sum_dep": [50.0, 20, 5],
                      "sum_thr": [100.0, -80, 10], "n_late_in": [4, 5, 0], "n_recovered": [1, 5, 0],
                      "n_ontime_in": [36, 35, 40], "n_made_late": [3, 0, 1]},
                     index=pd.Index(["A", "B", "C"], name="station_name"))
    st = R.through_rates(s, 10).reset_index().assign(
        station_class=["Junction", "Major station", "Minor halt"], station_full_name=["AY JN", "BEE", "SEE"],
        station_zone="NR", trains_day=100, approach_from=None, approach_change=np.nan, exit_to=None,
        exit_change=np.nan)
    b, a = R.rank_bottlenecks(st, 1)
    sc = R.group_scores(df, df["type_code"], 10, n_boot=50)
    t = {"through": st, "scores_class": sc.assign(label=sc["group"]),
         "scores_corridor": R.group_scores(df, df["train_no"].map({1: "NR-SR"}), 10, n_boot=50),
         "scores_station_group": R.group_scores(df, R.station_groups(df["station_name"], st, b, a), 10, n_boot=50),
         "scores_train": R.group_scores(df, df["train_no"], 10, n_boot=50).rename(columns={"group": "train_no"})}
    r = R.summarize(t, s, df, full=False, n_days=10)
    assert r["bottlenecks"][0]["station_name"] == "A" and r["absorbers"][0]["station_name"] == "B"
    assert r["trains_below_persistence"] == 1
    r["meta"] = {"source": "x", "predictions": "p", "output_dir": "o", "stops": 1, "days": 10,
                 "test_rows": len(df), "suspect_stations": 0, "figures": [], "elapsed_s": 0}
    md = R.render_markdown(r)
    assert "## 4. Bottleneck junctions" in md and "| A | AY JN |" in md
    for name, fn in R.FIGURES.items():
        assert isinstance(fn(r), Figure), name
