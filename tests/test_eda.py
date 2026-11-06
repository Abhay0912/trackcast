"""Tests for trackcast.eda on small synthetic timetables and runs (never the real data)."""
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats as st

from trackcast import config as C
from trackcast import data as D
from trackcast import eda as E
from trackcast import profiles as P

# train -> (type, [(station, zone, km)]); 500 km between stops so trains 1 and 2 are long-distance
ROUTES = {
    1: ("SF-TRAINS", [("AAA", "NR", 0), ("BBB", "NR", 500), ("CJN", "ECR", 1000), ("DDD", "ECR", 1500)]),
    2: ("EXP-TRAINS", [("DDD", "ECR", 0), ("CJN", "ECR", 500), ("BBB", "NR", 1000), ("AAA", "NR", 1500)]),
    3: ("PASS-TRAINS", [("XXX", "SR", 0), ("YYY", "SR", 20), ("ZZZ", "SR", 40), ("WWW", "SR", 60)]),
}
FULL_NAMES = {"CJN": "CEE JN", "AAA": "AY", "BBB": "BEE", "DDD": "DEE", "XXX": "EX", "YYY": "WHY", "ZZZ": "ZED",
              "WWW": "DUB"}


def schedule_frame() -> pd.DataFrame:
    rows = []
    for train, (_, route) in ROUTES.items():
        for i, (stn, _, km) in enumerate(route):
            last = i == len(route) - 1
            rows.append({"train_no": train, "station_no": i + 1, "station_name": stn, "distance_from_origin": km,
                         "arrival_day": 1, "departure_day": 1,
                         "sched_arr_offset_min": np.nan if i == 0 else 360.0 + 120 * i + train,
                         "sched_dep_offset_min": np.nan if last else 365.0 + 120 * i + train,
                         "is_origin": i == 0, "is_terminus": last, "n_sched_stops": len(route)})
    s = pd.DataFrame(rows)
    s["stop_day"] = 1
    return s.astype({"train_no": "int32", "station_no": "int16", "distance_from_origin": "float32"})


def lookups():
    stations = pd.DataFrame([{"station_name": stn, "station_full_name": FULL_NAMES[stn], "station_zone": z}
                             for _, route in ROUTES.values() for stn, z, _ in route]).drop_duplicates("station_name")
    trains = pd.DataFrame([{"train_no": t, "train_name": f"T{t}", "type_code": ty} for t, (ty, _) in ROUTES.items()])
    return stations.astype("string"), trains.astype({"train_no": "int32", "train_name": "string", "type_code": "string"})


def clean_frame(days: int = 40, seed: int = 0) -> pd.DataFrame:
    """Runs on `days` dates spread over the year; delay grows along the route with noise, some stops missing."""
    rng = np.random.default_rng(seed)
    stations, trains = lookups()
    sched = schedule_frame()
    dates = pd.Timestamp("2025-01-03") + pd.to_timedelta(np.linspace(0, 360, days).round(), unit="D")
    rows = []
    for train, (ty, route) in ROUTES.items():
        for day in dates:
            base = rng.integers(0, 40)
            for i, (stn, zone, km) in enumerate(route):
                d = float(base + 10 * i + rng.integers(-15, 15))
                if rng.random() < 0.1 and 0 < i < len(route) - 1:
                    d = np.nan
                rows.append({"train_no": train, "date": day, "station_no": i + 1, "station_name": stn,
                             "delay_raw": d, "delay": max(d, 0) if d == d else np.nan})
    df = pd.DataFrame(rows).merge(sched.drop(columns="station_name"), on=["train_no", "station_no"])
    df = df.merge(stations, on="station_name").merge(trains, on="train_no")
    df["run_id"] = D.make_run_id(df["train_no"], df["date"])
    return df.astype({"station_name": "category", "type_code": "category", "station_zone": "category",
                      "train_name": "category", "station_full_name": "category", "delay": "float32"})


@pytest.fixture(scope="module")
def ctx():
    stations, trains = lookups()
    return E.timetable_context(schedule_frame(), stations, trains)


@pytest.fixture(scope="module")
def clean():
    return clean_frame()


@pytest.fixture(scope="module")
def sums(clean, ctx):
    return E.merge_partials([E.partial_eda(clean, ctx)])


def test_corridor_label_is_unordered():
    a = pd.Series(["NR", "ECR", "SR", None], dtype="string")
    b = pd.Series(["ECR", "NR", "SR", "NR"], dtype="string")
    out = E.corridor_label(a, b)
    assert out[:3].tolist() == ["ECR-NR", "ECR-NR", "within SR"] and pd.isna(out[3])


def test_station_classes():
    names = pd.Series(["ITARSI JN", "NEW DELHI", "SOME HALT", "JNANA"], dtype="string")
    traffic = pd.Series([50, C.MAJOR_STATION_MIN_TRAINS_DAY, 3, 500])
    assert E.station_classes(names, traffic).tolist() == ["Junction", "Major station", "Minor halt", "Major station"]


def test_timetable_context(ctx):
    tr = ctx["trains"].set_index("train_no")
    assert tr.loc[1, "corridor"] == "ECR-NR" and tr.loc[2, "corridor"] == "ECR-NR"
    assert tr.loc[3, "corridor"] == "within SR" and tr.loc[1, "route_km"] == 1500
    stn = ctx["stations"].set_index("station_name")
    assert stn.loc["CJN", "station_class"] == "Junction" and stn.loc["XXX", "station_class"] == "Minor halt"
    assert stn.loc["BBB", "trains_day"] == 2


def test_grouped_partials_equal_single_pass(clean, ctx, sums):
    parts = [E.partial_eda(clean[clean["train_no"].isin(g)], ctx) for g in ([1, 3], [2])]
    grouped = E.merge_partials(parts)
    for name, t in sums.items():
        pd.testing.assert_frame_equal(grouped[name].sort_index(), t.sort_index(), check_dtype=False, obj=name)


def test_distribution_matches_raw_values(clean, sums):
    d = clean["delay"].dropna().to_numpy("float64")
    s = E.distribution_summary(sums["hist"], sums["moments"])
    assert s["arrivals"] == len(d) and s["median"] == np.median(d) // 1
    assert s["zero_pct"] == pytest.approx(100 * (d == 0).mean())
    assert s["skew_raw"] == pytest.approx(st.skew(d))
    assert s["skew_log"] == pytest.approx(st.skew(np.log1p(d)))
    lh = E.log_histogram(sums["hist"])
    assert lh["n"].sum() == len(d) and lh.loc[0, "n"] == (d == 0).sum()


def test_congestion_slope_is_least_squares():
    x = np.array([1.0, 2, 4, 8, 16])
    y = 2 + 0.3 * x
    s = pd.DataFrame({"n": [5], "sum_delay": [0.0], "n_late": [0], "n_early": [0], "n_change": [5],
                      "sum_change": [y.sum()], "n_gain": [5], "n_late_up": [0], "n_recovered": [0],
                      "sum_change_late_up": [0.0], "n_seg": [5], "sx": [x.sum()], "sy": [y.sum()],
                      "sxx": [(x * x).sum()], "sxy": [(x * y).sum()], "n_runs": [1], "n_trains": [1]})
    assert E.profile_rates(s).loc[0, "congestion_slope_per10"] == pytest.approx(3.0)


def test_profile_type_counts(clean, sums):
    pt = E.profile_rates(sums["profile_type"])
    obs = clean[clean["delay"].notna()]
    assert pt.loc["SF-TRAINS", "n_obs"] == (obs["train_no"] == 1).sum()
    assert pt.loc["SF-TRAINS", "n_runs"] == clean.loc[clean["train_no"] == 1, "run_id"].nunique()
    assert pt.loc["PASS-TRAINS", "mean_delay"] == pytest.approx(obs.loc[obs["train_no"] == 3, "delay"].mean())


def test_season_did():
    # fog zone NR: 10 min outside fog months, 30 in Jan; other zone SR: 10 and 15 -> DiD = 20 - 5 = 15
    rows = [(1, 3, "NR", 30), (4, 3, "NR", 10), (1, 3, "SR", 15), (4, 3, "SR", 10), (7, 3, "SR", 10),
            (7, 3, "NR", 10), (4, 14, "NR", 10), (1, 14, "NR", 20)]
    mhz = pd.DataFrame([{"month": m, "hour": h, "station_zone": z, "n": 10, "sum_delay": 10.0 * d, "n_late": 0,
                         "n_change": 10, "sum_change": 0.0} for m, h, z, d in rows]).set_index(["month", "hour",
                                                                                              "station_zone"])
    s = E.season_tables(mhz, min_zone_obs=1)
    f = E.did_numbers(s["fog_did"], s["fog_night"])["mean_delay"]
    # NR: fog months (30 + 20) / 2 = 25 vs 10 -> +15; SR: 15 vs 10 -> +5
    assert f["fog_zones_uplift"] == pytest.approx(15) and f["other_zones_uplift"] == pytest.approx(5)
    assert f["did"] == pytest.approx(10)
    # NR at night (hour 3): 30 vs 10 -> +20; daytime (hour 14): 20 vs 10 -> +10
    assert f["night_uplift"] == pytest.approx(20) and f["night_minus_day"] == pytest.approx(10)
    mz = s["monsoon_zone"].set_index("station_zone")
    assert mz.loc["SR", "delay_diff"] == pytest.approx(0) and mz.loc["NR", "delay_diff"] == pytest.approx(0)


def test_station_class_rates_are_pooled():
    sc = pd.DataFrame({"station_name": list("abc"), "station_class": ["Junction", "Junction", "Minor halt"],
                       "trains_day": [50, 60, 2], "median_dwell_min": [5, 2, 1], "n_obs": [100, 300, 50],
                       "sum_delay": [1000.0, 3000, 100], "n_late": [50, 60, 5], "n_late_up": [10, 90, 20],
                       "n_recovered": [10, 0, 4], "sum_change_late_up": [-50.0, 0, -8]})
    r = E.station_class_rates(sc).set_index("station_class")
    assert r.loc["Junction", "recovery_pct"] == pytest.approx(10)       # (10 + 0) / (10 + 90), not mean(100, 0)
    assert r.loc["Junction", "mean_delay"] == pytest.approx(10)
    assert r.loc["Minor halt", "recovery_pct"] == pytest.approx(20)


def test_corridor_trajectory_drops_thin_tail():
    idx = pd.MultiIndex.from_tuples([("A-B", 0), ("A-B", 1), ("A-B", 2)], names=["corridor", "km_bin"])
    tc = pd.DataFrame({"n": [5000, 5000, 5000], "sum_delay": [1e5, 2e5, 9e5], "n_late": [0, 0, 0],
                       "n_trains": [20, 20, 2]}, index=idx)
    r = E.corridor_trajectories(tc, min_obs=1)
    assert r["km_bin"].tolist() == [0, 1]


def test_figures_render(clean, ctx, sums):
    prof = P.merge_partials([P.partial_profiles(clean[P.READ_COLUMNS])])
    t = dict(sums)
    t["station_classes"] = E.station_class_table(prof["stations"], ctx["stations"])
    ex = clean[clean["train_no"] == 1]
    t["example_runs"] = ex.assign(station_name=ex["station_name"].astype("string"),
                                  train_name=ex["train_name"].astype("string"))
    t.update({f"p4_{k}": prof[k] for k in ("stations", "trains", "cascade", "by_date")})
    t["full"] = False
    for name, fn in E.FIGURES.items():
        assert isinstance(fn(t), Figure), name
    r = E.summarize(t, full=False)
    r["meta"] = {"source": "x", "profiles": "p", "output_dir": "o", "stops": len(clean), "figures": [],
                 "elapsed_s": 0}
    md = E.render_markdown(r)
    assert "## 4. Schedule recovery" in md and r["example_train"]["train_no"] == 1
