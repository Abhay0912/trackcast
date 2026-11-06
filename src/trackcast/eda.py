"""Phase 6: exploratory analysis and report figures (the proposal's "Exploratory analysis" step).

Usage:  python -m trackcast.eda [--full]

  default   sample.parquet + profiles_sample/ -> data/processed/eda_sample/, reports/eda.md, reports/figures/
  --full    clean.parquet  + profiles/        -> data/processed/eda/,        reports/eda.md, reports/figures/

Run it after `python -m trackcast.profiles` with the same flag: the cascade curve, the daily series and the
per-station table come from Phase 4 and are not recomputed.

What it answers:
  1. distribution   raw vs log1p delay: why the regressor's target is log1p(delay) and Stage 1 exists
  2. trajectories   how delay builds along a route, by train class and on the busiest long corridors;
                    every run of one train; how far a delay cascades
  3. season         fog (Dec-Feb, northern zones, night hours) and monsoon (Jun-Sep) slowdowns
  4. recovery       how often late trains recover time at junctions vs major stations vs minor halts
  5. profiles       per train class and per corridor: mean delay, recovery rate and congestion
                    sensitivity (the proposal's profile metrics that Phase 4 does not compute)

The clean table is streamed one group of trains at a time, as in Phase 4. Every table is a set of additive
sums (counts, sums, cross-products), so group partials merge exactly and rates are computed at the end.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator

from trackcast import config as C
from trackcast import data as D
from trackcast import features as F
from trackcast import profiles as P

LATE = C.SIGNIFICANT_DELAY_MIN
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DRY_MONTHS = tuple(m for m in range(1, 13) if m not in C.FOG_MONTHS + C.MONSOON_MONTHS)
TYPE_LABELS = {"PASS-TRAINS": "Passenger", "EXP-TRAINS": "Express", "SF-TRAINS": "Superfast",
               "RAJ-TRAINS": "Rajdhani", "T18-TRAINS": "Vande Bharat (T18)", "SHT-TRAINS": "Shatabdi",
               "GRB-TRAINS": "Garib Rath", "PRM-TRAINS": "Premium / special"}
FIGURE_TYPES = ["PASS-TRAINS", "EXP-TRAINS", "SF-TRAINS", "RAJ-TRAINS"]   # the classes the proposal names
CLASS_ORDER = ["Junction", "Major station", "Minor halt"]
CAL_SUMS = ["n", "sum_delay", "n_late", "n_change", "sum_change"]
TABLES = ["hist", "moments", "month_hour_zone", "profile_type", "profile_corridor", "traj_type",
          "traj_corridor", "congestion_type", "station_classes", "example_runs"]


def congestion_labels() -> list[str]:
    e = C.CONGESTION_BIN_EDGES
    return [str(e[0])] + [f"{a + 1}-{b}" if b > a + 1 else str(b) for a, b in zip(e, e[1:])] + [f"{e[-1] + 1}+"]


# --------------------------------------------------------------------------------------------------
# Timetable context (no delays involved)
# --------------------------------------------------------------------------------------------------
def corridor_label(a: pd.Series, b: pd.Series) -> pd.Series:
    """Zone of origin and terminus as an unordered pair ("ER-NR"): both directions are one corridor."""
    def one(x, y):
        if pd.isna(x) or pd.isna(y):
            return None
        return f"within {x}" if x == y else "-".join(sorted((str(x), str(y))))
    return pd.Series([one(x, y) for x, y in zip(a, b)], index=a.index, dtype="string")


def station_classes(full_name: pd.Series, trains_day: pd.Series) -> pd.Series:
    """Junction (named "... JN"), major station (other stations with heavy scheduled traffic), minor halt."""
    jn = full_name.astype("string").fillna("").str.contains(C.JUNCTION_NAME_PATTERN, regex=True).to_numpy(bool)
    major = ~jn & (trains_day.to_numpy("float64") >= C.MAJOR_STATION_MIN_TRAINS_DAY)
    return pd.Series(np.select([jn, major], CLASS_ORDER[:2], CLASS_ORDER[2]), index=full_name.index)


def timetable_context(schedule: pd.DataFrame, stations: pd.DataFrame, trains: pd.DataFrame) -> dict:
    """Per stop: congestion and route fraction; per train: class, length, corridor; per station: class."""
    sf = F.schedule_features(schedule)
    stop = schedule.merge(sf[["train_no", "station_no", "station_trains_hour", "route_frac"]],
                          on=["train_no", "station_no"], how="left")
    stop["station_name"] = stop["station_name"].astype("string")
    stop = stop.merge(stations[["station_name", "station_full_name", "station_zone"]], on="station_name", how="left")

    tm = pd.DataFrame({"route_km": stop.groupby("train_no")["distance_from_origin"].max(),
                       "origin_zone": stop[stop["is_origin"]].set_index("train_no")["station_zone"],
                       "terminus_zone": stop[stop["is_terminus"]].set_index("train_no")["station_zone"]})
    tm["corridor"] = corridor_label(tm["origin_zone"], tm["terminus_zone"])
    tm = tm.join(trains.set_index("train_no")["type_code"]).rename_axis("train_no").reset_index()

    dwell = (stop["sched_dep_offset_min"] - stop["sched_arr_offset_min"]).clip(lower=0)
    sm = stop.assign(dwell=dwell).groupby("station_name").agg(
        station_full_name=("station_full_name", "first"), trains_day=("train_no", "nunique"),
        median_dwell_min=("dwell", "median"))
    sm["station_class"] = station_classes(sm["station_full_name"], sm["trains_day"])
    return {"stops": stop[["train_no", "station_no", "station_trains_hour", "route_frac"]],
            "trains": tm, "stations": sm.reset_index()}


def load_context() -> dict:
    stats = D.QualityStats()
    return timetable_context(D.load_schedule(stats), D.load_stations(stats), D.load_trains(stats))


# --------------------------------------------------------------------------------------------------
# Partials for one group of trains (complete runs)
# --------------------------------------------------------------------------------------------------
def _profile_sums(obs: pd.DataFrame, fl: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    s = fl.groupby(key).sum()
    g = obs.groupby(key)
    s["n_runs"] = g["run_id"].nunique()     # exact after merging: a train never spans two groups
    s["n_trains"] = g["train_no"].nunique()
    return s


def partial_eda(clean: pd.DataFrame, ctx: dict) -> dict[str, pd.DataFrame]:
    df = P.prepare(clean)
    df = df.merge(ctx["stops"], on=["train_no", "station_no"], how="left")
    df = df.merge(ctx["trains"][["train_no", "corridor", "route_km"]], on="train_no", how="left")
    obs = df[df["delay"].notna()].reset_index(drop=True)
    fl = P._flags(obs)

    d = obs["delay"].to_numpy("float64")
    y = np.log1p(d)
    hist = np.bincount(np.clip(np.rint(d), 0, C.HIST_MAX_DELAY_MIN).astype(np.int64),
                       minlength=C.HIST_MAX_DELAY_MIN + 1)
    moments = {"n": float(len(d))} | {f"{k}_s{p}": float(np.sum(v ** p)) for k, v in (("raw", d), ("log", y))
                                      for p in (1, 2, 3)}

    # Congestion sensitivity: delay change over a single scheduled segment vs scheduled trains per hour at
    # the arrival station. Changes spanning missing stops are left out, since they mix several stations.
    x = obs["station_trains_hour"].to_numpy("float64", na_value=np.nan)
    ch = obs["change"].to_numpy("float64")
    seg = (obs["stops_back"].to_numpy("float64") == 1) & ~np.isnan(ch) & ~np.isnan(x)
    xs, ys = np.where(seg, x, 0.0), np.where(seg, ch, 0.0)
    fl = fl.assign(n_seg=seg.astype(int), sx=xs, sy=ys, sxx=xs * xs, sxy=xs * ys)

    type_ = obs["type_code"].astype("string").rename("type_code")
    corr = obs["corridor"].astype("string").rename("corridor")
    zone = obs["station_zone"].astype("string").rename("station_zone")
    frac_bin = pd.Series(np.clip(np.floor(obs["route_frac"].to_numpy("float64") * 10), 0, 9),
                         name="frac_bin").astype("Int8")
    long = (obs["route_km"] >= C.CORRIDOR_MIN_KM).to_numpy()
    km_bin = pd.Series(np.floor(obs["distance_from_origin"].to_numpy("float64") / C.TRAJECTORY_KM_BIN),
                       name="km_bin").astype("Int16")
    cong_bin = pd.Series(np.searchsorted(np.asarray(C.CONGESTION_BIN_EDGES, "float64"), x, side="left"),
                         name="cong_bin")
    traj = fl[["n", "sum_delay", "n_late"]]
    corr_keys = [corr[long], km_bin[long]]
    traj_corr = traj[long].groupby(corr_keys).sum()
    traj_corr["n_trains"] = obs.loc[long, "train_no"].groupby(corr_keys).nunique()   # exact: groups split trains
    return {
        "hist": pd.Series(hist, name="n").rename_axis("delay_min").to_frame(),
        "moments": pd.DataFrame([moments]),
        "month_hour_zone": fl[CAL_SUMS].groupby([obs["month"].rename("month"), obs["hour"].rename("hour"), zone]).sum(),
        "profile_type": _profile_sums(obs, fl, type_),
        "profile_corridor": _profile_sums(obs, fl, corr),
        "traj_type": traj.groupby([type_, frac_bin]).sum(),
        "traj_corridor": traj_corr,
        "congestion_type": fl.loc[seg, ["n_seg", "sy"]].groupby([type_[seg], cong_bin[seg]]).sum(),
    }


def merge_partials(parts: list[dict]) -> dict[str, pd.DataFrame]:
    out = {}
    for name in parts[0]:
        frames = [p[name] for p in parts if len(p[name])]
        x = pd.concat(frames)
        out[name] = x.groupby(level=list(range(x.index.nlevels))).sum()
    out["moments"] = out["moments"].sum().to_frame().T
    return out


# --------------------------------------------------------------------------------------------------
# Tables built from Phase 4 profiles
# --------------------------------------------------------------------------------------------------
def station_class_table(st_prof: pd.DataFrame, st_meta: pd.DataFrame) -> pd.DataFrame:
    """Per station: Phase 4 counts rebuilt from its rates, plus class, traffic and dwell from the timetable.

    Stations Phase 4 flags as timetable errors (sawtooth), and stations whose arrivals mostly come from one,
    are dropped: their "recovery" is an artefact of a wrong scheduled time."""
    s = st_prof.merge(st_meta[["station_name", "station_class", "trains_day", "median_dwell_min"]],
                      on="station_name", how="left")
    s = s[~s["suspect_timetable"].astype(bool) & (s["inbound_suspect_share"] < 0.5) & s["station_class"].notna()]
    late_up = s["n_late_upstream"].fillna(0)
    return pd.DataFrame({
        "station_name": s["station_name"], "station_class": s["station_class"], "trains_day": s["trains_day"],
        "median_dwell_min": s["median_dwell_min"], "n_obs": s["n_obs"], "sum_delay": s["mean_delay"] * s["n_obs"],
        "n_late": s["late_pct"] / 100 * s["n_obs"], "n_late_up": late_up,
        "n_recovered": (s["recovery_pct_when_late"] / 100 * late_up).fillna(0),
        "sum_change_late_up": (s["mean_change_when_late"] * late_up).fillna(0)}).reset_index(drop=True)


def pick_example_train(trains_prof: pd.DataFrame) -> int | None:
    """A long, frequently run express / superfast train: the one with the most runs (then the longest)."""
    t = trains_prof[trains_prof["type_code"].isin(["SF-TRAINS", "EXP-TRAINS", "RAJ-TRAINS"])
                    & (trains_prof["route_km"] >= C.CORRIDOR_MIN_KM) & (trains_prof["n_sched_stops"] >= 10)]
    if t.empty:
        return None
    return int(t.sort_values(["n_runs", "route_km", "train_no"], ascending=[False, False, True]).iloc[0]["train_no"])


def example_runs(src, train_no: int) -> pd.DataFrame:
    cols = ["run_id", "date", "station_no", "station_name", "distance_from_origin", "delay", "train_name"]
    df = pq.read_table(src, columns=cols, filters=[("train_no", "=", train_no)]).to_pandas()
    df["train_no"] = train_no
    for c in ("station_name", "train_name"):
        df[c] = df[c].astype("string")
    return df.sort_values(["run_id", "station_no"], ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Derived rates (from the saved sum tables)
# --------------------------------------------------------------------------------------------------
def _cal_rates(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    out["n_obs"] = s["n"]
    out["mean_delay"] = s["sum_delay"] / s["n"]
    out["late_pct"] = 100 * s["n_late"] / s["n"]
    out["mean_change"] = s["sum_change"] / s["n_change"].where(s["n_change"] > 0)
    return out


def distribution_summary(hist: pd.DataFrame, moments: pd.DataFrame) -> dict:
    h = hist["n"]
    m = moments.iloc[0]
    n = m["n"]

    def skew(prefix):
        mu, s2, s3 = m[f"{prefix}_s1"] / n, m[f"{prefix}_s2"] / n, m[f"{prefix}_s3"] / n
        var = s2 - mu ** 2
        return (s3 - 3 * mu * s2 + 2 * mu ** 3) / var ** 1.5, mu, var ** 0.5

    cum = h.cumsum() / h.sum()
    q = lambda p: int(cum.index[np.searchsorted(cum.to_numpy(), p)])
    sk_raw, mu_raw, sd_raw = skew("raw")
    sk_log, mu_log, sd_log = skew("log")
    return {"arrivals": int(n), "zero_pct": 100 * h.iloc[0] / n, "late_pct": 100 * h[h.index > LATE].sum() / n,
            "over_3h_pct": 100 * h[h.index > 180].sum() / n, "mean": mu_raw, "sd": sd_raw, "median": q(0.5),
            "p90": q(0.9), "p99": q(0.99), "max": int(h[h > 0].index.max()), "skew_raw": sk_raw,
            "skew_log": sk_log, "mean_log": mu_log, "sd_log": sd_log}


def log_histogram(hist: pd.DataFrame, width: float = 0.4) -> pd.DataFrame:
    """1-minute counts re-binned on log1p(delay): exact, since delays are whole minutes. Bins narrower than
    ~0.4 leave empty bins between the first few whole minutes (log1p 0, 0.69, 1.10, 1.39, ...)."""
    b = np.floor(np.log1p(hist.index.to_numpy("float64")) / width).astype(int)
    g = hist["n"].groupby(b).sum()
    return pd.DataFrame({"log_lo": g.index * width, "n": g.to_numpy()})


def profile_rates(s: pd.DataFrame) -> pd.DataFrame:
    r = P._rates(s)
    n = s["n_seg"].where(s["n_seg"] > 0)
    mx, my = s["sx"] / n, s["sy"] / n
    var = s["sxx"] / n - mx ** 2
    r["congestion_slope_per10"] = 10 * (s["sxy"] / n - mx * my) / var.where(var > 0)
    r["n_runs"], r["n_trains"], r["n_segments"] = s["n_runs"], s["n_trains"], s["n_seg"]
    return r


def trajectory_rates(t: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    r = _cal_rates(t.assign(n_change=0, sum_change=0)).drop(columns="mean_change")
    return r[r["n_obs"] >= min_obs].reset_index()


def corridor_trajectories(tc: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    """Mean delay per km bin, kept while >= EDA_CORRIDOR_MIN_TRAIN_SHARE of the corridor's trains still run
    there. Past that, a bin holds only the few longest trains and their delays, not the corridor's."""
    total = tc["n_trains"].groupby(level="corridor").transform("max")
    keep = tc["n_trains"] >= C.EDA_CORRIDOR_MIN_TRAIN_SHARE * total
    return trajectory_rates(tc[keep].drop(columns="n_trains"), min_obs)


def top_corridors(profile_corridor: pd.DataFrame, traj_corridor: pd.DataFrame, n: int) -> list[str]:
    """Long-distance corridors (between two zones) with the most observed arrivals."""
    tot = traj_corridor.groupby(level="corridor")["n"].sum()
    tot = tot[~tot.index.str.startswith("within ")]
    return list(tot.sort_values(ascending=False).index[:n])


def season_frame(mhz: pd.DataFrame) -> pd.DataFrame:
    t = mhz.reset_index()
    lo, hi = C.FOG_NIGHT_HOURS
    t["fog_zone"] = t["station_zone"].isin(C.FOG_ZONES)
    t["season"] = np.select([t["month"].isin(C.FOG_MONTHS), t["month"].isin(C.MONSOON_MONTHS)],
                            ["fog", "monsoon"], "dry")
    t["night"] = (t["hour"] >= lo) | (t["hour"] < hi)
    return t


def season_tables(mhz: pd.DataFrame, min_zone_obs: int) -> dict[str, pd.DataFrame]:
    t = season_frame(mhz)
    rates = lambda keys, d=t: _cal_rates(d.groupby(keys)[CAL_SUMS].sum()).reset_index()
    by_month = rates(["fog_zone", "month"])
    fogm = t.assign(fog_months=t["month"].isin(C.FOG_MONTHS))
    by_hour = rates(["fog_zone", "fog_months", "hour"], fogm)

    # Difference in differences: the fog-month uplift in fog zones minus the same uplift elsewhere.
    did = rates(["fog_zone", "fog_months"], fogm).set_index(["fog_zone", "fog_months"])
    night = rates(["night", "fog_months"], fogm[fogm["fog_zone"]]).set_index(["night", "fog_months"])

    rows = []
    for zone, g in t.groupby("station_zone"):
        r = _cal_rates(g.groupby("season")[CAL_SUMS].sum())
        if not {"monsoon", "dry"} <= set(r.index) or min(r.loc["monsoon", "n_obs"], r.loc["dry", "n_obs"]) < min_zone_obs:
            continue
        rows.append({"station_zone": zone, "n_monsoon": r.loc["monsoon", "n_obs"], "n_dry": r.loc["dry", "n_obs"],
                     "mean_delay_dry": r.loc["dry", "mean_delay"], "mean_delay_monsoon": r.loc["monsoon", "mean_delay"],
                     "mean_change_dry": r.loc["dry", "mean_change"],
                     "mean_change_monsoon": r.loc["monsoon", "mean_change"]})
    mz = pd.DataFrame(rows, columns=["station_zone", "n_monsoon", "n_dry", "mean_delay_dry", "mean_delay_monsoon",
                                     "mean_change_dry", "mean_change_monsoon"])
    if len(mz):
        mz["delay_diff"] = mz["mean_delay_monsoon"] - mz["mean_delay_dry"]
        mz["change_diff"] = mz["mean_change_monsoon"] - mz["mean_change_dry"]
        mz = mz.sort_values("delay_diff", ascending=False, ignore_index=True)
    else:
        mz = mz.assign(delay_diff=pd.Series(dtype=float), change_diff=pd.Series(dtype=float))
    return {"by_month": by_month, "by_hour": by_hour, "fog_did": did, "fog_night": night, "monsoon_zone": mz}


def did_numbers(did: pd.DataFrame, night: pd.DataFrame) -> dict:
    def g(tab, a, b, col="mean_delay"):
        try:
            return float(tab.loc[(a, b), col])
        except KeyError:
            return float("nan")
    out = {}
    for col in ("mean_delay", "mean_change"):
        up_fog = g(did, True, True, col) - g(did, True, False, col)
        up_other = g(did, False, True, col) - g(did, False, False, col)
        up_night = g(night, True, True, col) - g(night, True, False, col)
        up_day = g(night, False, True, col) - g(night, False, False, col)
        out[col] = {"fog_zones_uplift": up_fog, "other_zones_uplift": up_other, "did": up_fog - up_other,
                    "night_uplift": up_night, "day_uplift": up_day, "night_minus_day": up_night - up_day}
    return out


def station_class_rates(sc: pd.DataFrame) -> pd.DataFrame:
    g = sc.groupby("station_class")
    s = g[["n_obs", "sum_delay", "n_late", "n_late_up", "n_recovered", "sum_change_late_up"]].sum()
    out = pd.DataFrame({"stations": g.size(), "arrivals": s["n_obs"], "mean_delay": s["sum_delay"] / s["n_obs"],
                        "late_pct": 100 * s["n_late"] / s["n_obs"], "late_arrivals_upstream": s["n_late_up"],
                        "recovery_pct": 100 * s["n_recovered"] / s["n_late_up"].where(s["n_late_up"] > 0),
                        "mean_change_when_late": s["sum_change_late_up"] / s["n_late_up"].where(s["n_late_up"] > 0),
                        "median_trains_day": g["trains_day"].median(), "median_dwell_min": g["median_dwell_min"].median()})
    return out.reindex([c for c in CLASS_ORDER if c in out.index]).reset_index()


def recovery_by_traffic(sc: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Recovery rate by scheduled traffic (station quintiles), junctions vs other stations: separates the
    junction effect from simply being a busy station."""
    s = sc[sc["n_late_up"] > 0].copy()
    if len(s) < n_bins:
        return pd.DataFrame(columns=["traffic_bin", "traffic", "junction", "stations", "recovery_pct"])
    edges = np.unique(np.quantile(s["trains_day"], np.linspace(0, 1, n_bins + 1)))
    s["traffic_bin"] = np.clip(np.searchsorted(edges, s["trains_day"], side="right") - 1, 0, len(edges) - 2)
    s["junction"] = np.where(s["station_class"] == "Junction", "Junction", "Other stations")
    g = s.groupby(["traffic_bin", "junction"])
    out = pd.DataFrame({"stations": g.size(), "recovery_pct": 100 * g["n_recovered"].sum() / g["n_late_up"].sum()})
    out = out.reset_index()
    labels = [f"{int(a) + (i > 0)}-{int(b)}" for i, (a, b) in enumerate(zip(edges, edges[1:]))]
    out["traffic"] = [labels[b] for b in out["traffic_bin"]]
    return out


def congestion_rates(ct: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    r = ct.assign(mean_change=ct["sy"] / ct["n_seg"]).reset_index()
    r = r[r["n_seg"] >= min_obs].copy()
    labels = congestion_labels()
    r["congestion"] = [labels[int(b)] for b in r["cong_bin"]]
    return r


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def _dirs(full: bool):
    return (C.CLEAN_PARQUET if full else C.SAMPLE_PARQUET, C.PROFILES_DIR if full else C.PROFILES_SAMPLE_DIR,
            C.EDA_DIR if full else C.EDA_SAMPLE_DIR)


def build(full: bool = False) -> dict:
    t0 = time.time()
    src, prof_dir, out_dir = _dirs(full)
    if not (prof_dir / "stations.parquet").exists():
        raise SystemExit(f"{prof_dir} is missing: run `python -m trackcast.profiles{' --full' if full else ''}` first")
    ctx = load_context()
    groups = P.train_partitions(src, C.PROFILE_TRAIN_PARTITIONS if full else 1)
    parts, rows = [], 0
    for i, trains in enumerate(groups, 1):
        clean = pq.read_table(src, columns=P.READ_COLUMNS, filters=[("train_no", "in", trains)]).to_pandas()
        rows += len(clean)
        parts.append(partial_eda(clean, ctx))
        del clean
        print(f"  group {i}/{len(groups)}: {rows:,} stops so far ({time.time() - t0:.0f} s)")
    tables = merge_partials(parts)
    prof = load_profiles(full)
    tables["station_classes"] = station_class_table(prof["stations"], ctx["stations"])
    ex = pick_example_train(prof["trains"])
    tables["example_runs"] = example_runs(src, ex) if ex is not None else pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tab in tables.items():
        (tab.reset_index() if name in INDEX else tab).to_parquet(out_dir / f"{name}.parquet", index=False)
    t = load(full)
    report = summarize(t, full)
    report["meta"] = {"source": src.name, "profiles": str(prof_dir.relative_to(C.ROOT)),
                      "output_dir": str(out_dir.relative_to(C.ROOT)), "stops": rows,
                      "figures": save_figures(t), "elapsed_s": round(time.time() - t0, 1)}
    write_report(report)
    return report


def load_profiles(full: bool) -> dict[str, pd.DataFrame]:
    d = _dirs(full)[1]
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in ("stations", "trains", "cascade", "by_date")}


INDEX = {"hist": ["delay_min"], "month_hour_zone": ["month", "hour", "station_zone"], "profile_type": ["type_code"],
         "profile_corridor": ["corridor"], "traj_type": ["type_code", "frac_bin"],
         "traj_corridor": ["corridor", "km_bin"], "congestion_type": ["type_code", "cong_bin"]}


def load(full: bool = False) -> dict:
    """Everything the figures and the report need: the EDA sum tables (indexed as built), the Phase 4
    tables they draw on, and `full` (which sets the minimum-support thresholds)."""
    d = _dirs(full)[2]
    t = {}
    for name in TABLES:
        x = pd.read_parquet(d / f"{name}.parquet")
        t[name] = x.set_index(INDEX[name]) if name in INDEX else x
    t.update({f"p4_{k}": v for k, v in load_profiles(full).items()})
    t["full"] = full
    return t


def _min(t: dict, what: str) -> int:
    return C.EDA_MIN_OBS[what][1 if t["full"] else 0]


# --------------------------------------------------------------------------------------------------
# Figures (matplotlib, light surface; the reference data-viz palette, as in the app)
# --------------------------------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED, GRID, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df", "#efeeea"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]    # slots 1-4: blue, orange, aqua, yellow
ORDINAL = ["#86b6ef", "#3987e5", "#256abf", "#184f95"]        # ordered buckets, light = small
BLUE, DARK_BLUE = CATEGORICAL[0], ORDINAL[3]


def _fig(ncols: int = 1, w: float = 7.2, h: float = 3.6, sharey: bool = False):
    fig = Figure(figsize=(w, h), dpi=100, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, ncols, sharey=sharey, squeeze=False)[0]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8, length=0, pad=4)
    return fig, axes


def _titles(ax, title: str, sub: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=18 if sub else 6)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=INK2)


def _line(ax, x, y, color, label=None, lw: float = 2.0, dot: bool = True, **kw):
    ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", label=label, **kw)
    if dot and len(x):
        ax.scatter([list(x)[-1]], [list(y)[-1]], s=30, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)


def _end_labels(ax, items: list[tuple[float, float, str]], pad_frac: float = 0.22):
    """Direct labels at line ends, in text ink, nudged apart so they never overlap; widens the x range."""
    if not items:
        return
    x0, x1 = ax.get_xlim()
    if not isinstance(ax.xaxis.get_major_locator(), FixedLocator):   # keep ticks inside the data range
        ax.set_xticks([v for v in ax.get_xticks() if x0 <= v <= x1])
    ax.set_xlim(x0, x1 + pad_frac * (x1 - x0))
    lo, hi = ax.get_ylim()
    gap = 0.065 * (hi - lo)
    near = 0.15 * (x1 - x0)          # only labels whose line ends are this close in x can collide
    items = sorted(items, key=lambda r: r[1])
    ys = []
    for x, y, _ in items:
        below = [yj for (xj, _, _), yj in zip(items, ys) if abs(xj - x) < near]
        ys.append(max([y] + [yj + gap for yj in below]))
    for (x, _, text), y in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                    annotation_clip=False)


def _legend(ax_or_fig, ncol: int = 4, **kw):
    leg = ax_or_fig.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=ncol, handlelength=1.6, **kw)
    return leg


def _bar_height(n: int, axes_h_in: float) -> float:
    """Bar thickness capped at ~24 px (0.24 in at 100 dpi): never fill the band."""
    return min(0.6, 0.24 / (axes_h_in / max(n, 1)))


def _shade_months(ax, months: list[int], label: str):
    """Shade a run of consecutive months (x = month number) and name it at the top of the plot."""
    ax.axvspan(months[0] - 0.5, months[-1] + 0.5, color=SHADE, zorder=0, lw=0)
    ax.text(np.mean(months), 0.03, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7.5, color=MUTED)


def fig_delay_distribution(t: dict) -> Figure:
    hist, s = t["hist"], distribution_summary(t["hist"], t["moments"])
    fig, (a, b) = _fig(2, w=8.0, h=3.4)
    n = hist["n"].sum()
    width = 5
    binned = hist["n"].groupby(hist.index // width * width).sum()
    binned = binned[binned.index < 360]
    a.bar(binned.index, 100 * binned / n, width=width, align="edge", color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    a.axvline(LATE, color=INK2, lw=1)
    a.annotate(f"{LATE}-min hurdle", (LATE, 0.92), xycoords=("data", "axes fraction"), xytext=(5, 0),
               textcoords="offset points", fontsize=7.5, color=INK2)
    _titles(a, "Delay in minutes", f"Skewness {s['skew_raw']:.1f}; {s['over_3h_pct']:.1f}% of arrivals are > 3 h late "
            f"(max {s['max']:,})", "Arrival delay (min, 5-min bins, first 6 h)", "% of arrivals")
    lh = log_histogram(hist)
    b.bar(lh["log_lo"], 100 * lh["n"] / n, width=0.4, align="edge", color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    b.axvline(np.log1p(LATE), color=INK2, lw=1)
    b.annotate(f"{s['zero_pct']:.0f}% on time", (0.4, 100 * lh["n"].iloc[0] / n), xytext=(4, 0),
               textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ticks = [0, 5, 15, 60, 180, 600]
    b.set_xticks(np.log1p(ticks), [str(v) for v in ticks])
    _titles(b, "Delay on the log1p scale", f"Skewness {s['skew_log']:.2f}: the regression target log1p(delay)",
            "Arrival delay (min, log1p axis)", "% of arrivals")
    return fig


def fig_cascade(t: dict) -> Figure:
    cas = t["p4_cascade"].sort_values(["bucket", "h"])
    fig, (a, b) = _fig(2, w=8.0, h=3.5)
    labels = {0: f"<= {C.CASCADE_BUCKETS[0]} min", 1: f"{C.CASCADE_BUCKETS[0]}-{C.CASCADE_BUCKETS[1]} min",
              2: f"{C.CASCADE_BUCKETS[1] // 60}-{C.CASCADE_BUCKETS[2] // 60} h", 3: f"> {C.CASCADE_BUCKETS[2] // 60} h"}
    for ax, col in ((a, "mean_change"), (b, "late_pct_h")):
        ends = []
        for bk, g in cas.groupby("bucket"):
            _line(ax, g["h"], g[col], ORDINAL[int(bk)], label=labels[int(bk)] if ax is a else None)
            ends.append((g["h"].iloc[-1], g[col].iloc[-1], labels[int(bk)]))
        if col == "mean_change":
            ax.axhline(0, color=MUTED, lw=1)
        else:
            ax.set_ylim(0, 105)
        _end_labels(ax, ends, 0.3)
    _titles(a, "Delay gained after h stops", "Mean delay h observed stops later minus the delay at the start",
            "Stops later (h)", "Minutes")
    _titles(b, "Still late after h stops", "% of trains > 15 min late h stops later", "Stops later (h)", "%")
    _legend(fig, loc="outside lower center", title=None)
    return fig


def fig_trajectories(t: dict) -> Figure:
    fig, (a, b) = _fig(2, w=8.4, h=3.7)
    tt = trajectory_rates(t["traj_type"], _min(t, "bin"))
    ends = []
    for i, code in enumerate(FIGURE_TYPES):
        g = tt[tt["type_code"] == code].sort_values("frac_bin")
        if g.empty:
            continue
        x = (g["frac_bin"].astype(float) + 0.5) * 10
        _line(a, x, g["mean_delay"], CATEGORICAL[i], label=TYPE_LABELS[code])
        ends.append((x.iloc[-1], g["mean_delay"].iloc[-1], TYPE_LABELS[code]))
    a.set_ylim(bottom=0)
    _end_labels(a, ends)
    _titles(a, "Delay along the route, by train class", "Mean arrival delay by share of the route covered",
            "Route covered (%)", "Mean delay (min)")
    _legend(a, ncol=2, loc="lower right")

    corridors = top_corridors(t["profile_corridor"], t["traj_corridor"], C.EDA_FIGURE_SERIES)
    tc = corridor_trajectories(t["traj_corridor"], _min(t, "bin"))
    ends = []
    for i, c in enumerate(corridors):
        g = tc[tc["corridor"] == c].sort_values("km_bin")
        x = (g["km_bin"].astype(float) + 0.5) * C.TRAJECTORY_KM_BIN
        _line(b, x, g["mean_delay"], CATEGORICAL[i], label=c)
        ends.append((x.iloc[-1], g["mean_delay"].iloc[-1], c))
    b.set_ylim(bottom=0)
    _end_labels(b, ends)
    _titles(b, "Delay along the busiest long corridors", f"Trains >= {C.CORRIDOR_MIN_KM:,} km, by zones of origin "
            "and terminus (both directions)", "Distance from origin (km)", "Mean delay (min)")
    _legend(b, ncol=2, loc="lower right")
    return fig


def fig_example_train(t: dict) -> Figure:
    r = t["example_runs"]
    fig, (ax,) = _fig(1, w=7.6, h=3.6)
    if r.empty:
        _titles(ax, "No example train in this data")
        return fig
    obs = r[r["delay"].notna()]
    run_ids = np.sort(obs["run_id"].unique())
    shown = run_ids[:: max(1, len(run_ids) // 60)]          # ~60 runs spread over the year stay readable
    for i, (_, g) in enumerate(obs[obs["run_id"].isin(shown)].groupby("run_id")):
        ax.plot(g["distance_from_origin"], g["delay"], color=ORDINAL[0], lw=0.8, alpha=0.45,
                label=f"One run ({len(shown)} of {len(run_ids)} shown)" if i == 0 else None)
    med = obs.groupby("station_no").agg(km=("distance_from_origin", "first"), median=("delay", "median"),
                                        n=("delay", "size"))
    med = med[med["n"] >= max(3, 0.2 * obs["run_id"].nunique())]
    _line(ax, med["km"], med["median"], DARK_BLUE, label="Median over runs")
    first, last = r.iloc[0], r.sort_values("station_no").iloc[-1]
    _titles(ax, f"Every run of {int(first['train_no'])} {first['train_name']}",
            f"{obs['run_id'].nunique()} runs, {first['station_name']} to {last['station_name']} "
            f"({last['distance_from_origin']:,.0f} km)", "Distance from origin (km)", "Arrival delay (min)")
    hi = np.nanquantile(obs["delay"], 0.99) if len(obs) else 1
    ax.set_ylim(0, max(hi * 1.05, 10))
    _legend(ax, ncol=2, loc="upper left")
    return fig


def fig_season_month(t: dict) -> Figure:
    s = season_tables(t["month_hour_zone"], _min(t, "zone"))["by_month"]
    fig, (ax,) = _fig(1, w=7.2, h=3.5)
    _shade_months(ax, [1, 2], "Fog")
    _shade_months(ax, [12], "Fog")
    _shade_months(ax, list(C.MONSOON_MONTHS), "Monsoon")
    ends = []
    for i, (fz, label) in enumerate(((True, "Fog-prone zones"), (False, "Other zones"))):
        g = s[s["fog_zone"] == fz].sort_values("month")
        _line(ax, g["month"], g["mean_delay"], CATEGORICAL[i], label=label)
        ends.append((g["month"].iloc[-1], g["mean_delay"].iloc[-1], label))
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(bottom=0)
    _end_labels(ax, ends, 0.2)
    _titles(ax, "Delay by month: fog-prone northern zones vs the rest",
            f"Mean arrival delay; fog-prone = {', '.join(C.FOG_ZONES)}", None, "Mean delay (min)")
    _legend(ax, ncol=2, loc="upper center")
    return fig


def fig_fog_hour(t: dict) -> Figure:
    s = season_tables(t["month_hour_zone"], _min(t, "zone"))["by_hour"]
    fig, axes = _fig(2, w=8.4, h=3.4, sharey=True)
    lo, hi = C.FOG_NIGHT_HOURS
    for ax, fz, title in ((axes[0], True, "Fog-prone zones"), (axes[1], False, "Other zones")):
        ax.axvspan(lo - 0.5, 23.5, color=SHADE, lw=0, zorder=0)
        ax.axvspan(-0.5, hi - 0.5, color=SHADE, lw=0, zorder=0)
        ax.text((hi - 1) / 2, 0.97, "Fog hours", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color=MUTED)
        for i, (fm, label) in enumerate(((True, "Dec-Feb"), (False, "Mar-Nov"))):
            g = s[(s["fog_zone"] == fz) & (s["fog_months"] == fm)].sort_values("hour")
            _line(ax, g["hour"], g["mean_delay"], CATEGORICAL[i], label=label if fz else None, dot=False)
        ax.set_xticks(range(0, 24, 3), [f"{h:02d}" for h in range(0, 24, 3)])
        ax.set_xlim(-0.5, 23.5)
        _titles(ax, title, None, "Scheduled arrival hour", "Mean delay (min)" if fz else None)
    axes[0].set_ylim(bottom=0)
    _legend(fig, ncol=2, loc="outside lower center")
    return fig


def fig_daily(t: dict) -> Figure:
    d = t["p4_by_date"].sort_values("date")
    d = d[d["n_obs"] >= _min(t, "bin")]
    fig, (ax,) = _fig(1, w=8.0, h=3.3)
    dates = pd.to_datetime(d["date"])
    for months, label in ((C.FOG_MONTHS, "Fog"), (C.MONSOON_MONTHS, "Monsoon")):
        m = dates.dt.month.isin(months).to_numpy()
        for run in np.split(np.arange(len(m)), np.where(np.diff(m.astype(int)) != 0)[0] + 1):
            if len(run) and m[run[0]]:
                ax.axvspan(dates.iloc[run[0]], dates.iloc[run[-1]], color=SHADE, lw=0, zorder=0)
                ax.text(dates.iloc[run[len(run) // 2]], 0.97, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.plot(dates, d["mean_delay"], color=ORDINAL[0], lw=0.8, label="Daily")
    roll = d["mean_delay"].rolling(7, center=True, min_periods=4).mean()
    ax.plot(dates, roll, color=DARK_BLUE, lw=2, label="7-day mean")
    ax.set_ylim(bottom=0)
    _titles(ax, "Network delay through the year", "Mean arrival delay of runs starting each day", None,
            "Mean delay (min)")
    _legend(ax, ncol=2, loc="lower left")
    return fig


def fig_monsoon_zone(t: dict) -> Figure:
    mz = season_tables(t["month_hour_zone"], _min(t, "zone"))["monsoon_zone"]
    n = max(len(mz), 1)
    h = max(2.6, 0.28 * n + 1.0)
    fig, (ax,) = _fig(1, w=6.4, h=h)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    if len(mz):
        m = mz.iloc[::-1]
        y = np.arange(len(m))
        ax.barh(y, m["delay_diff"], height=_bar_height(len(m), h - 1.0), color=BLUE)
        ax.set_yticks(y, m["station_zone"])
        ax.axvline(0, color=INK2, lw=1)
        span = np.ptp(np.r_[m["delay_diff"].to_numpy(), 0]) or 1
        for yi, v in zip(y, m["delay_diff"]):
            ax.annotate(f"{v:+.1f}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=7.5, color=INK2)
        lo, hi = min(m["delay_diff"].min(), 0), max(m["delay_diff"].max(), 0)
        ax.set_xlim(lo - 0.12 * span, hi + 0.12 * span)
    months = "-".join(MONTHS[m - 1] for m in (C.MONSOON_MONTHS[0], C.MONSOON_MONTHS[-1]))
    _titles(ax, "Monsoon slowdown by zone", f"Mean delay in {months} minus {', '.join(MONTHS[m - 1] for m in DRY_MONTHS)}",
            "Minutes (positive = later in the monsoon)")
    return fig


def fig_recovery(t: dict) -> Figure:
    sc = t["station_classes"]
    cls = station_class_rates(sc)
    fig, (a, b) = _fig(2, w=8.4, h=3.4)
    a.grid(axis="y", visible=False)
    a.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(cls))[::-1]
    a.barh(y, cls["recovery_pct"], height=_bar_height(len(cls), 2.4), color=BLUE)
    a.set_yticks(y, cls["station_class"])
    for yi, v in zip(y, cls["recovery_pct"]):
        a.annotate(f"{v:.1f}%", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                   color=INK2)
    a.set_xlim(0, cls["recovery_pct"].max() * 1.22 if len(cls) else 1)
    _titles(a, "Late trains that recover time", "% of arrivals late upstream whose delay falls on arrival",
            "% of late arrivals")
    rt = recovery_by_traffic(sc)
    ends = []
    for i, j in enumerate(["Junction", "Other stations"]):
        g = rt[rt["junction"] == j].sort_values("traffic_bin")
        if g.empty:
            continue
        _line(b, g["traffic_bin"], g["recovery_pct"], CATEGORICAL[i], label=j)
        ends.append((g["traffic_bin"].iloc[-1], g["recovery_pct"].iloc[-1], j))
    if len(rt):
        bins = rt.drop_duplicates("traffic_bin").sort_values("traffic_bin")
        b.set_xticks(bins["traffic_bin"], bins["traffic"])
    b.set_ylim(bottom=0)
    _end_labels(b, ends, 0.3)
    _titles(b, "Same traffic, junction or not", "Recovery rate by scheduled trains per day (station quintiles)",
            "Scheduled trains per day at the station", "% of late arrivals")
    _legend(b, ncol=2, loc="lower left")
    return fig


def fig_congestion(t: dict) -> Figure:
    cr = congestion_rates(t["congestion_type"], _min(t, "bin"))
    fig, (ax,) = _fig(1, w=7.2, h=3.5)
    ax.axhline(0, color=MUTED, lw=1)
    ends = []
    for i, code in enumerate(FIGURE_TYPES):
        g = cr[cr["type_code"] == code].sort_values("cong_bin")
        if g.empty:
            continue
        _line(ax, g["cong_bin"], g["mean_change"], CATEGORICAL[i], label=TYPE_LABELS[code])
        ax.scatter(g["cong_bin"], g["mean_change"], s=22, color=CATEGORICAL[i], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3)
        ends.append((g["cong_bin"].iloc[-1], g["mean_change"].iloc[-1], TYPE_LABELS[code]))
    labels = congestion_labels()
    ax.set_xticks(range(len(labels)), labels)
    _end_labels(ax, ends, 0.25)
    _titles(ax, "Delay gained on arrival vs station congestion",
            "Mean change in delay over one scheduled segment, by trains scheduled at the arrival station that hour",
            "Scheduled trains at the station in that hour", "Minutes gained (negative = recovered)")
    _legend(ax, ncol=4, loc="upper left")
    return fig


FIGURES = {
    "eda_delay_distribution": fig_delay_distribution,
    "eda_cascade": fig_cascade,
    "eda_trajectories": fig_trajectories,
    "eda_example_train": fig_example_train,
    "eda_season_month": fig_season_month,
    "eda_fog_hour": fig_fog_hour,
    "eda_daily": fig_daily,
    "eda_monsoon_zone": fig_monsoon_zone,
    "eda_recovery": fig_recovery,
    "eda_congestion": fig_congestion,
}


def save_figures(t: dict) -> list[str]:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fn in FIGURES.items():
        path = C.FIGURES_DIR / f"{name}.png"
        fn(t).savefig(path, dpi=200, facecolor=SURFACE)
        out.append(str(path.relative_to(C.ROOT)).replace("\\", "/"))
    return out


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _rec(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return P._records(df, cols)


def example_stats(ex: pd.DataFrame) -> dict | None:
    """Median and p10-p90 spread across runs at the first and last stop most runs report."""
    if ex.empty:
        return None
    obs = ex[ex["delay"].notna()]
    runs = obs["run_id"].nunique()
    size = obs.groupby("station_no").size()
    ok = size[size >= 0.5 * runs].index
    out = {"train_no": int(ex["train_no"].iloc[0]), "train_name": str(ex["train_name"].iloc[0]), "runs": int(runs)}
    for where, st in (("first", ok.min()), ("last", ok.max())):
        d = obs.loc[obs["station_no"] == st]
        out[f"{where}_station"] = str(d["station_name"].iloc[0])
        out.update({f"{where}_{k}": round(float(d["delay"].quantile(p)), 1)
                    for k, p in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))})
    return out


def summarize(t: dict, full: bool) -> dict:
    min_runs = _min(t, "profile_runs")
    pt = profile_rates(t["profile_type"]).reset_index()
    pt["label"] = pt["type_code"].map(TYPE_LABELS).fillna(pt["type_code"])
    pc_ = profile_rates(t["profile_corridor"]).reset_index()
    pc_ = pc_[pc_["n_runs"] >= min_runs]
    seasons = season_tables(t["month_hour_zone"], _min(t, "zone"))
    sf = season_frame(t["month_hour_zone"])
    season_level = _cal_rates(sf.groupby("season")[CAL_SUMS].sum()).reset_index()
    prof_cols = ["n_trains", "n_runs", "n_obs", "mean_delay", "late_pct", "mean_change", "recovery_pct_when_late",
                 "mean_change_when_late", "congestion_slope_per10", "n_segments"]
    ex = t["example_runs"]
    cong = congestion_rates(t["congestion_type"], _min(t, "bin"))
    cas = t["p4_cascade"]
    corridors = top_corridors(t["profile_corridor"], t["traj_corridor"], C.EDA_FIGURE_SERIES)
    return {
        "distribution": {k: (round(float(v), 3) if isinstance(v, float) else v)
                         for k, v in distribution_summary(t["hist"], t["moments"]).items()},
        "profile_type": _rec(pt.sort_values("n_runs", ascending=False), ["type_code", "label", *prof_cols]),
        "profile_corridor": _rec(pc_.sort_values("n_runs", ascending=False).head(15), ["corridor", *prof_cols]),
        "figure_corridors": corridors,
        "season_level": _rec(season_level, ["season", "n_obs", "mean_delay", "late_pct", "mean_change"]),
        "fog": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                for k, v in did_numbers(seasons["fog_did"], seasons["fog_night"]).items()},
        "monsoon_zone": _rec(seasons["monsoon_zone"], ["station_zone", "n_monsoon", "n_dry", "mean_delay_dry",
                                                        "mean_delay_monsoon", "delay_diff", "change_diff"]),
        "station_class": _rec(station_class_rates(t["station_classes"]),
                              ["station_class", "stations", "arrivals", "mean_delay", "late_pct",
                               "late_arrivals_upstream", "recovery_pct", "mean_change_when_late",
                               "median_trains_day", "median_dwell_min"]),
        "recovery_by_traffic": _rec(recovery_by_traffic(t["station_classes"]),
                                    ["traffic", "junction", "stations", "recovery_pct"]),
        "congestion_bins": _rec(cong[cong["type_code"].isin(FIGURE_TYPES)],
                                ["type_code", "congestion", "n_seg", "mean_change"]),
        "cascade_last": _rec(cas[cas["h"] == cas["h"].max()], ["h", "initial_delay", "n_pairs", "mean_change",
                                                               "late_pct_h"]),
        "example_train": example_stats(ex),
        "thresholds": {"junction_pattern": C.JUNCTION_NAME_PATTERN,
                       "major_station_min_trains_day": C.MAJOR_STATION_MIN_TRAINS_DAY,
                       "corridor_min_km": C.CORRIDOR_MIN_KM, "min_obs": {k: v[1 if full else 0]
                                                                         for k, v in C.EDA_MIN_OBS.items()}},
    }


def write_report(r: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.EDA_JSON.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    C.EDA_MD.write_text(render_markdown(r), encoding="utf-8")
    d, m = r["distribution"], r["meta"]
    print(f"\n=== EDA ({m['source']} -> {m['output_dir']}, {len(m['figures'])} figures) ===")
    print(f"  {d['arrivals']:,} arrivals: skewness {d['skew_raw']:.2f} raw -> {d['skew_log']:.2f} log1p")
    f = r["fog"]["mean_delay"]
    print(f"  fog: +{f['fog_zones_uplift']:.1f} min in fog zones vs +{f['other_zones_uplift']:.1f} elsewhere "
          f"(DiD {f['did']:+.1f}); night - day {f['night_minus_day']:+.1f}  ({m['elapsed_s']} s)")


def _fig_md(name: str) -> str:
    return f"![{name}](figures/{name}.png)"


def render_markdown(r: dict) -> str:
    d, m, th = r["distribution"], r["meta"], r["thresholds"]
    md = P._md_table
    L = ["# Exploratory analysis (Phase 6)", "",
         f"Source `{m['source']}` ({m['stops']:,} scheduled stops) and the Phase 4 profiles in `{m['profiles']}/`. "
         f"Sum tables in `{m['output_dir']}/`; figures in `reports/figures/`; `notebooks/01_eda.ipynb` shows them "
         "with the tables. All delays in minutes; *change* = delay at a stop minus delay at the previous observed "
         "stop (negative = time recovered); *late* = more than 15 min.", "",
         "## 1. Delay distribution: why log1p and why a hurdle", "", _fig_md("eda_delay_distribution"), "",
         f"- {d['arrivals']:,} observed arrivals. Mean {d['mean']:.1f}, median {d['median']}, p90 {d['p90']}, "
         f"p99 {d['p99']}, max {d['max']:,}.",
         f"- {d['zero_pct']:.1f}% exactly on time (0 after clipping early arrivals), {d['late_pct']:.1f}% late, "
         f"{d['over_3h_pct']:.1f}% more than 3 h late.",
         f"- Skewness {d['skew_raw']:.2f} in minutes vs {d['skew_log']:.2f} after log1p: squared error on raw "
         "minutes would be dominated by the few multi-hour delays, so Stage 2 regresses log1p(delay) and is "
         "scored with RMSLE. The spike at 0 is a separate population, which is what Stage 1 models.", "",
         "## 2. Cascading delay trajectories", "", _fig_md("eda_cascade"), "",
         "From Phase 4's cascade table, by the delay at the starting stop, "
         f"{r['cascade_last'][0]['h'] if r['cascade_last'] else '-'} observed stops later:", "",
         *md(r["cascade_last"], {"initial_delay": "Initial delay", "n_pairs": "Pairs",
                                 "mean_change": "Mean delay gained", "late_pct_h": "% still late"}), "",
         _fig_md("eda_trajectories"), "", "**By train class** (all classes; the chart shows the four the proposal "
         "names). *Recovery* = % of arrivals late at the previous stop whose delay falls on arrival. *Congestion "
         "sensitivity* = least-squares slope of the delay change over one scheduled segment on the trains "
         "scheduled at the arrival station in that hour, in minutes per 10 extra trains.", ""]
    H = {"n_trains": "Trains", "n_runs": "Runs", "n_obs": "Arrivals", "mean_delay": "Mean delay",
         "late_pct": "% late", "mean_change": "Mean change", "recovery_pct_when_late": "% late that recover",
         "mean_change_when_late": "Change if late", "congestion_slope_per10": "Congestion: min per +10 trains/h"}
    L += md(r["profile_type"], {"label": "Class", **H})
    L += ["", f"**By corridor** (zones of origin and terminus, both directions; the 15 with the most runs, "
          f">= {th['min_obs']['profile_runs']:,} runs). Charted: {', '.join(r['figure_corridors'])} "
          f"(trains >= {th['corridor_min_km']:,} km).", ""]
    L += md(r["profile_corridor"], {"corridor": "Corridor", **H})
    ex = r["example_train"]
    if ex:
        L += ["", _fig_md("eda_example_train"), "",
              f"Train {ex['train_no']} {ex['train_name']} ({ex['runs']} runs), across runs: at "
              f"{ex['first_station']} median {ex['first_median']:g} min (p10-p90 {ex['first_p10']:g}-"
              f"{ex['first_p90']:g}); at {ex['last_station']} median {ex['last_median']:g} min (p10-p90 "
              f"{ex['last_p10']:g}-{ex['last_p90']:g}). The same train on the same route ends hours apart, so a "
              "train's or a station's average delay says little about one run; the run's own upstream delay does."]
    f, fc = r["fog"]["mean_delay"], r["fog"]["mean_change"]
    L += ["", "## 3. Seasonal slowdowns: fog and monsoon", "", _fig_md("eda_season_month"), "",
          _fig_md("eda_fog_hour"), "", _fig_md("eda_daily"), "",
          "| | Fog-prone zones | Other zones | Difference |", "|---|---:|---:|---:|",
          f"| Mean delay, Dec-Feb minus Mar-Nov | {f['fog_zones_uplift']:+.1f} | {f['other_zones_uplift']:+.1f} "
          f"| **{f['did']:+.1f}** |",
          f"| Mean change on arrival, Dec-Feb minus Mar-Nov | {fc['fog_zones_uplift']:+.2f} | "
          f"{fc['other_zones_uplift']:+.2f} | **{fc['did']:+.2f}** |", "",
          f"Within fog-prone zones, the Dec-Feb uplift in mean delay is {f['night_uplift']:+.1f} min for "
          f"arrivals scheduled in fog hours ({C.FOG_NIGHT_HOURS[0]}:00-{C.FOG_NIGHT_HOURS[1]}:00) vs "
          f"{f['day_uplift']:+.1f} min in daytime ({f['night_minus_day']:+.1f}). This is what the `fog_season` "
          "and `fog_night` features encode.", "",
          "**Season level** (all zones)", ""]
    L += md(r["season_level"], {"season": "Season", "n_obs": "Arrivals", "mean_delay": "Mean delay",
                                "late_pct": "% late", "mean_change": "Mean change"})
    L += ["", _fig_md("eda_monsoon_zone"), "", f"Zones with >= {th['min_obs']['zone']:,} arrivals in each season. "
          "*Diff* columns are monsoon minus dry months (Mar-May, Oct-Nov).", ""]
    L += md(r["monsoon_zone"], {"station_zone": "Zone", "n_monsoon": "Monsoon arrivals", "n_dry": "Dry arrivals",
                                "mean_delay_dry": "Mean delay, dry", "mean_delay_monsoon": "Mean delay, monsoon",
                                "delay_diff": "Delay diff", "change_diff": "Change diff"})
    L += ["", "Caveat: there are no weather observations in the data, so these are *season* effects. "
          "Fog-month delay also carries winter timetable changes and holiday traffic; the comparison with other "
          "zones in the same months (difference in differences) removes the network-wide part of that.", "",
          "## 4. Schedule recovery: junctions vs minor halts", "", _fig_md("eda_recovery"), "",
          f"Junction = station name contains `JN`; major station = any other station with >= "
          f"{th['major_station_min_trains_day']} distinct scheduled trains a day; the rest are minor halts. "
          "Stations Phase 4 flags as timetable errors (sawtooth) are left out.", ""]
    L += md(r["station_class"], {"station_class": "Class", "stations": "Stations", "arrivals": "Arrivals",
                                 "mean_delay": "Mean delay", "late_pct": "% late",
                                 "late_arrivals_upstream": "Late arrivals", "recovery_pct": "% that recover",
                                 "mean_change_when_late": "Change if late", "median_trains_day": "Median trains/day",
                                 "median_dwell_min": "Median sched. dwell"})
    L += ["", "**Recovery by traffic** (station quintiles of scheduled trains per day)", ""]
    L += md(r["recovery_by_traffic"], {"traffic": "Trains/day", "junction": "Stations", "stations": "Count",
                                       "recovery_pct": "% that recover"})
    L += ["", "Recovery is measured on arrival, so it covers the inbound segment and the dwell at the previous "
          "stop, not the dwell at the station itself.", "",
          "## 5. Congestion", "", _fig_md("eda_congestion"), ""]
    L += md([{**x, "type_code": TYPE_LABELS.get(x["type_code"], x["type_code"])} for x in r["congestion_bins"]],
            {"type_code": "Class", "congestion": "Trains/h at station", "n_seg": "Segments",
             "mean_change": "Mean change"})
    L += ["", "Congestion counts trains *scheduled* at the station in the same clock hour (the timetable has no "
          "running days, so every train counts as daily). Busy stations are also the big stations with recovery "
          "allowances, so this is an association, not the effect of one extra train.", "",
          f"Elapsed: {m['elapsed_s']} s."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="use clean.parquet and the --full profiles")
    build(full=ap.parse_args().full)


if __name__ == "__main__":
    main()
