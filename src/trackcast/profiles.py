"""Phase 4: delay and route profiles (descriptive statistics for the report and the Streamlit app).

Usage:  python -m trackcast.profiles [--full]

  default   data/processed/sample.parquet -> data/processed/profiles_sample/*.parquet  (development)
  --full    data/processed/clean.parquet  -> data/processed/profiles/*.parquet

Tables (one parquet file each):
  stations     per station: delay level (mean / median / p90, % late, % early) and what happens *at* the
               station: mean change from the previous observed stop, recovery rate of late arrivals
  segments     per consecutive scheduled stop pair (A -> B): delay gained or recovered, scheduled minutes
  train_stops  per (train, stop): the route profile of a train, i.e. how delay builds up along its route
  trains       per train: origin / terminus delay, delay gained en route, punctuality
  by_month_zone, by_dow, by_hour, by_date   network delay over the calendar
  cascade      delay h observed stops after a stop, by the initial delay: how far a delay cascades

The clean table is processed one group of trains at a time. A group holds every run of its trains, so the
per-train and per-(train, stop) tables, including medians and p90, are exact. Station, segment, calendar and
cascade tables are summed from per-group partials (station quantiles from exact 1-minute histograms).
"Change" at stop k = delay_k - delay at the previous *observed* stop of the same run (as lag1 in Phase 2).

These profiles describe all dates and are never model inputs; the Phase 3 models use only their own
fold-fitted encoders, so no profile can leak test-period delays into a forecast.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from trackcast import config as C
from trackcast import data as D
from trackcast import features as F

LATE = C.SIGNIFICANT_DELAY_MIN
READ_COLUMNS = ["run_id", "train_no", "date", "station_no", "station_name", "delay", "delay_raw",
                "sched_arr_offset_min", "sched_dep_offset_min", "distance_from_origin", "is_origin",
                "is_terminus", "train_name", "type_code", "station_full_name", "station_zone"]
TABLES = ["stations", "segments", "train_stops", "trains", "by_month_zone", "by_dow", "by_hour", "by_date",
          "cascade"]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------------------------------
# Per-stop derived columns
# --------------------------------------------------------------------------------------------------
def prepare(clean: pd.DataFrame) -> pd.DataFrame:
    """Sort by run and stop, attach the previous observed stop's delay / station and calendar fields."""
    df = clean.sort_values(["run_id", "station_no"], ignore_index=True)
    n = len(df)
    run_start = np.arange(n) - df.groupby("run_id", sort=False).cumcount().to_numpy()
    delay = df["delay"].to_numpy("float64", na_value=np.nan)
    j1, _ = F.lag_positions(run_start, ~np.isnan(delay), 1)

    arr = df["sched_arr_offset_min"].to_numpy("float64", na_value=np.nan)
    dep = df["sched_dep_offset_min"].to_numpy("float64", na_value=np.nan)
    event = np.where(np.isnan(arr), dep, arr)
    leave = np.where(np.isnan(dep), arr, dep)
    stations = df["station_name"].astype("string").to_numpy(dtype=object, na_value=None)

    df["prev_delay"] = F._gather(delay, j1)
    df["prev_station"] = np.where(j1 >= 0, stations[np.maximum(j1, 0)], None)
    df["stops_back"] = df["station_no"].to_numpy("float64") - F._gather(df["station_no"].to_numpy("float64"), j1)
    df["seg_sched_min"] = event - F._gather(leave, j1)
    df["change"] = delay - df["prev_delay"].to_numpy()
    ts = df["date"] + pd.to_timedelta(event, unit="min")
    df["month"] = ts.dt.month.astype("Int8")
    df["dow"] = ts.dt.day_of_week.astype("Int8")
    df["hour"] = pd.array(np.floor(event % 1440 / 60), dtype="Int8")
    return df


# --------------------------------------------------------------------------------------------------
# Partials for one group of trains
# --------------------------------------------------------------------------------------------------
def _flags(obs: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns whose group sums give every rate in the station / segment / calendar tables."""
    late_up = (obs["prev_delay"] > LATE).to_numpy()
    change = obs["change"].to_numpy("float64")
    has_change = ~np.isnan(change)
    return pd.DataFrame({
        "n": 1, "sum_delay": obs["delay"].to_numpy("float64"),
        "n_late": (obs["delay"] > LATE).to_numpy(int), "n_early": (obs["delay_raw"] < 0).to_numpy(int),
        "n_change": has_change.astype(int), "sum_change": np.where(has_change, change, 0.0),
        "n_gain": (change > 0).astype(int),
        "n_late_up": (late_up & has_change).astype(int),
        "n_recovered": (late_up & (change < 0)).astype(int),
        "sum_change_late_up": np.where(late_up & has_change, change, 0.0),
    }, index=obs.index)


def train_stop_profile(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    keys = ["train_no", "station_no"]
    sched = df.groupby(keys, sort=False, observed=True).agg(
        station_name=("station_name", "first"), sched_arr_offset_min=("sched_arr_offset_min", "first"),
        sched_dep_offset_min=("sched_dep_offset_min", "first"),
        distance_from_origin=("distance_from_origin", "first"), is_origin=("is_origin", "first"),
        is_terminus=("is_terminus", "first"), n_runs=("run_id", "size"))
    g = obs.groupby(keys, sort=False, observed=True)
    stats = g.agg(n_obs=("delay", "size"), mean_delay=("delay", "mean"), median_delay=("delay", "median"),
                  mean_change=("change", "mean"))
    stats["p90_delay"] = g["delay"].quantile(0.9)
    fl = _flags(obs).groupby([obs["train_no"], obs["station_no"]], sort=False).sum()
    stats["late_pct"] = 100 * fl["n_late"] / fl["n"]
    stats["early_pct"] = 100 * fl["n_early"] / fl["n"]
    stats["recovery_pct_when_late"] = 100 * fl["n_recovered"] / fl["n_late_up"].where(fl["n_late_up"] > 0)
    out = sched.join(stats, how="left")
    out["missing_pct"] = 100 * (1 - out["n_obs"].fillna(0) / out["n_runs"])
    return out.reset_index().sort_values(keys, ignore_index=True)


def train_profile(df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    meta = df.groupby("train_no", sort=False, observed=True).agg(
        train_name=("train_name", "first"), type_code=("type_code", "first"), n_runs=("run_id", "nunique"),
        n_sched_stops=("station_no", "nunique"), route_km=("distance_from_origin", "max"))
    first = df[df["is_origin"]].groupby("train_no", observed=True)["station_name"].first()
    last = df[df["is_terminus"]].groupby("train_no", observed=True)["station_name"].first()
    meta["origin"], meta["terminus"] = first.astype("string"), last.astype("string")

    origin = obs[obs["is_origin"]].set_index("run_id")["delay"]
    term = obs[obs["is_terminus"]].set_index("run_id")["delay"]
    runs = pd.DataFrame({"origin_delay": origin, "terminus_delay": term})
    runs["gained_en_route"] = runs["terminus_delay"] - runs["origin_delay"]
    runs["train_no"] = runs.index.to_numpy() // D.RUN_ID_MULT   # run_id = train_no * RUN_ID_MULT + days
    g = runs.groupby("train_no")
    per_run = g.agg(mean_origin_delay=("origin_delay", "mean"), mean_terminus_delay=("terminus_delay", "mean"),
                    median_terminus_delay=("terminus_delay", "median"),
                    mean_gained_en_route=("gained_en_route", "mean"))
    per_run["terminus_late_pct"] = 100 * g["terminus_delay"].apply(lambda s: (s.dropna() > LATE).mean())
    stops = obs.groupby("train_no", observed=True).agg(mean_delay=("delay", "mean"), n_obs=("delay", "size"))
    stops["late_pct"] = 100 * obs.assign(l=obs["delay"] > LATE).groupby("train_no", observed=True)["l"].mean()
    return meta.join(per_run).join(stops).reset_index().sort_values("train_no", ignore_index=True)


def cascade_partial(obs: pd.DataFrame, max_h: int = C.CASCADE_MAX_STOPS) -> pd.DataFrame:
    """Sums over all pairs (stop i, the h-th observed stop after i in the same run), by initial delay bucket."""
    rid = obs["run_id"].to_numpy()
    d = obs["delay"].to_numpy("float64")
    nb = len(C.CASCADE_BUCKETS) + 1
    rows = []
    for h in range(1, max_h + 1):
        if len(d) <= h:
            break
        same = rid[h:] == rid[:-h]
        d0, d1 = d[:-h][same], d[h:][same]
        b = np.searchsorted(np.asarray(C.CASCADE_BUCKETS, "float64"), d0, side="left")  # <=15 -> 0, ...
        cnt = lambda w=None: np.bincount(b, weights=w, minlength=nb)
        rows.append(pd.DataFrame({"h": h, "bucket": np.arange(nb), "n": cnt(), "sum_delay0": cnt(d0),
                                  "sum_delay_h": cnt(d1), "n_late_h": cnt((d1 > LATE).astype(float)),
                                  "n_lower_h": cnt((d1 < d0).astype(float))}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def partial_profiles(clean: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All profile pieces for one group of trains (complete runs)."""
    df = prepare(clean)
    obs = df[df["delay"].notna()]
    fl = _flags(obs)
    station = obs["station_name"].astype("string")
    delay_bin = obs["delay"].round().clip(0, C.HIST_MAX_DELAY_MIN).astype("int16")

    seg_rows = obs["stops_back"].eq(1) & obs["change"].notna()
    seg = fl[seg_rows].assign(sum_sched_min=obs.loc[seg_rows, "seg_sched_min"].to_numpy())
    seg_keys = [obs.loc[seg_rows, "prev_station"].astype("string").rename("from_station"),
                station[seg_rows].rename("to_station")]

    cal = fl[["n", "sum_delay", "n_late"]]
    return {
        "train_stops": train_stop_profile(df, obs),
        "trains": train_profile(df, obs),
        "station_sums": fl.groupby(station.rename("station_name")).sum(),
        "station_hist": delay_bin.groupby([station.rename("station_name"), delay_bin.rename("delay_min")]).size(),
        "station_meta": obs.groupby(station.rename("station_name"))[["station_full_name", "station_zone"]]
                           .first().astype("string"),
        "segment_sums": seg.groupby(seg_keys).sum(),
        "by_month_zone": cal.groupby([obs["month"].rename("month"),
                                      obs["station_zone"].astype("string").rename("station_zone")]).sum(),
        "by_dow": cal.groupby(obs["dow"].rename("dow")).sum(),
        "by_hour": cal.groupby(obs["hour"].rename("hour")).sum(),
        "by_date": cal.groupby(obs["date"].rename("date")).sum(),
        "cascade": cascade_partial(obs),
    }


# --------------------------------------------------------------------------------------------------
# Merge partials into final tables
# --------------------------------------------------------------------------------------------------
def hist_quantiles(hist: pd.Series, qs=(0.5, 0.9)) -> pd.DataFrame:
    """hist: counts indexed by (key, value) -> per key, the smallest value with cumulative share >= q."""
    h = hist.sort_index()
    key = h.index.get_level_values(0)
    val = h.index.get_level_values(1).to_numpy().astype("int64")
    cum = h.groupby(level=0, sort=False).cumsum().to_numpy()
    total = h.groupby(level=0, sort=False).transform("sum").to_numpy()
    out = {}
    for q in qs:
        first = pd.Series(np.where(cum >= q * total, val, np.iinfo(np.int64).max), index=key)
        out[f"p{int(q * 100)}_delay"] = first.groupby(level=0, sort=False).min()
    return pd.DataFrame(out)


def _rates(s: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    out["n_obs"] = s["n"]
    out["mean_delay"] = s["sum_delay"] / s["n"]
    out["late_pct"] = 100 * s["n_late"] / s["n"]
    if "n_change" in s:
        out["early_pct"] = 100 * s["n_early"] / s["n"]
        out["mean_change"] = s["sum_change"] / s["n_change"].where(s["n_change"] > 0)
        out["gain_pct"] = 100 * s["n_gain"] / s["n_change"].where(s["n_change"] > 0)
        out["n_late_upstream"] = s["n_late_up"]
        out["recovery_pct_when_late"] = 100 * s["n_recovered"] / s["n_late_up"].where(s["n_late_up"] > 0)
        out["mean_change_when_late"] = s["sum_change_late_up"] / s["n_late_up"].where(s["n_late_up"] > 0)
    return out


def _sum(parts: list[dict], name: str):
    frames = [p[name] for p in parts if len(p[name])]
    x = pd.concat(frames)
    return x.groupby(level=list(range(x.index.nlevels))).sum()


def merge_partials(parts: list[dict]) -> dict[str, pd.DataFrame]:
    stations = _rates(_sum(parts, "station_sums")).join(hist_quantiles(_sum(parts, "station_hist")))
    meta = pd.concat([p["station_meta"] for p in parts])
    stations = meta[~meta.index.duplicated()].join(stations, how="right")

    seg_s = _sum(parts, "segment_sums")
    segments = _rates(seg_s)
    segments["mean_sched_min"] = seg_s["sum_sched_min"] / seg_s["n"]

    # Sawtooth check: a station whose recorded delay is systematically off (typically a wrong timetable time
    # there) shows a large change on arrival that is undone on the next segment. Such changes are artefacts,
    # not real delay or recovery, so the stations and the segments touching them are flagged.
    out_s = seg_s.groupby(level="from_station")[["sum_change", "n_change"]].sum()
    stations["outbound_mean_change"] = out_s["sum_change"] / out_s["n_change"].where(out_s["n_change"] > 0)
    inb, outb = stations["mean_change"], stations["outbound_mean_change"]
    stations["suspect_timetable"] = ((np.sign(inb) * np.sign(outb) < 0)
                                     & (np.minimum(inb.abs(), outb.abs()) >= C.SAWTOOTH_MIN_CHANGE)).fillna(False)
    suspect = set(stations.index[stations["suspect_timetable"]])
    segments["touches_suspect"] = [a in suspect or b in suspect for a, b in segments.index]
    # A neighbour of a suspect inherits the mirror image of its error (arrivals from a station whose recorded
    # delay is too low look like delay gained), so record how much of each station's arrival change comes
    # from segments touching a suspect.
    inbound = seg_s["n_change"].groupby(level="to_station")
    touch = seg_s["n_change"].where(segments["touches_suspect"], 0).groupby(level="to_station").sum()
    stations["inbound_suspect_share"] = (touch / inbound.sum()).reindex(stations.index).fillna(0.0)

    cas = _sum([{"c": p["cascade"].set_index(["h", "bucket"])} for p in parts if len(p["cascade"])], "c")
    cas = cas[cas["n"] > 0]
    labels = [f"<= {C.CASCADE_BUCKETS[0]}"] + [f"({a}, {b}]" for a, b in zip(C.CASCADE_BUCKETS, C.CASCADE_BUCKETS[1:])] \
        + [f"> {C.CASCADE_BUCKETS[-1]}"]
    cascade = pd.DataFrame({"n_pairs": cas["n"], "mean_delay0": cas["sum_delay0"] / cas["n"],
                            "mean_delay_h": cas["sum_delay_h"] / cas["n"],
                            "late_pct_h": 100 * cas["n_late_h"] / cas["n"],
                            "lower_pct_h": 100 * cas["n_lower_h"] / cas["n"]}).reset_index()
    cascade["initial_delay"] = [labels[b] for b in cascade["bucket"]]
    cascade["mean_change"] = cascade["mean_delay_h"] - cascade["mean_delay0"]

    by_dow = _rates(_sum(parts, "by_dow")).reset_index()
    by_dow["day"] = [DOW_NAMES[int(d)] for d in by_dow["dow"]]
    out = {
        "stations": stations.reset_index().sort_values("station_name", ignore_index=True),
        "segments": segments.reset_index(),
        "train_stops": pd.concat([p["train_stops"] for p in parts], ignore_index=True),
        "trains": pd.concat([p["trains"] for p in parts], ignore_index=True).sort_values("train_no", ignore_index=True),
        "by_month_zone": _rates(_sum(parts, "by_month_zone")).reset_index(),
        "by_dow": by_dow,
        "by_hour": _rates(_sum(parts, "by_hour")).reset_index(),
        "by_date": _rates(_sum(parts, "by_date")).reset_index(),
        "cascade": cascade[["h", "bucket", "initial_delay", "n_pairs", "mean_delay0", "mean_delay_h",
                            "mean_change", "late_pct_h", "lower_pct_h"]],
    }
    for t in out.values():   # categoricals from different groups concat to object; store plain strings
        for c in t.columns:
            if t[c].dtype == object or isinstance(t[c].dtype, pd.CategoricalDtype):
                t[c] = t[c].astype("string")
    return out


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def train_partitions(src, n: int) -> list[list[int]]:
    trains = np.sort(pc.unique(pq.read_table(src, columns=["train_no"]).column("train_no")).to_numpy())
    return [list(map(int, trains[i::n])) for i in range(n) if len(trains[i::n])]


def build(full: bool = False) -> dict:
    t0 = time.time()
    src = C.CLEAN_PARQUET if full else C.SAMPLE_PARQUET
    out_dir = C.PROFILES_DIR if full else C.PROFILES_SAMPLE_DIR
    groups = train_partitions(src, C.PROFILE_TRAIN_PARTITIONS if full else 1)
    parts, rows = [], 0
    for i, trains in enumerate(groups, 1):
        clean = pq.read_table(src, columns=READ_COLUMNS, filters=[("train_no", "in", trains)]).to_pandas()
        rows += len(clean)
        parts.append(partial_profiles(clean))
        del clean
        print(f"  group {i}/{len(groups)}: {len(trains):,} trains, {rows:,} stops so far ({time.time() - t0:.0f} s)")
    tables = merge_partials(parts)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_parquet(out_dir / f"{name}.parquet", index=False)
    report = summarize(tables, full)
    report["meta"] = {"source": src.name, "output_dir": str(out_dir.relative_to(C.ROOT)), "stops": rows,
                      "trains": len(tables["trains"]), "stations": len(tables["stations"]),
                      "segments": len(tables["segments"]), "elapsed_s": round(time.time() - t0, 1)}
    write_report(report)
    return report


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def _records(df: pd.DataFrame, cols: list[str], nd: int = 2) -> list[dict]:
    return [{c: (None if pd.isna(v) else round(float(v), nd) if isinstance(v, (float, np.floating)) else
                 int(v) if isinstance(v, (int, np.integer)) else str(v)) for c, v in r.items()}
            for r in df[cols].to_dict("records")]


def summarize(t: dict[str, pd.DataFrame], full: bool) -> dict:
    k = 1 if full else 0
    min_st, min_seg, min_tr = (C.PROFILE_MIN_OBS[x][k] for x in ("station", "segment", "train"))
    st = t["stations"][t["stations"]["n_obs"] >= min_st]
    seg = t["segments"][t["segments"]["n_late_upstream"] >= min_seg]
    suspects = st[st["suspect_timetable"]]
    st_clean = st[~st["suspect_timetable"] & (st["inbound_suspect_share"] < 0.5)]
    seg_clean = seg[~seg["touches_suspect"]]
    tr = t["trains"][t["trains"]["n_runs"] >= min_tr].astype({"train_no": "string"})
    st_cols = ["station_name", "station_full_name", "station_zone", "n_obs", "mean_delay", "p50_delay",
               "p90_delay", "late_pct", "mean_change", "recovery_pct_when_late"]
    seg_cols = ["from_station", "to_station", "n_obs", "mean_sched_min", "mean_change", "mean_change_when_late",
                "recovery_pct_when_late"]
    tr_cols = ["train_no", "train_name", "type_code", "n_runs", "route_km", "mean_origin_delay",
               "mean_terminus_delay", "mean_gained_en_route", "terminus_late_pct"]
    cas = t["cascade"]
    cas = cas[cas["h"].isin([1, 2, 3, 5, 10, 15, 20])]
    by_type = (t["trains"].groupby("type_code").apply(lambda g: pd.Series({
        "trains": len(g), "runs": int(g["n_runs"].sum()),
        "mean_terminus_delay": np.average(g["mean_terminus_delay"].fillna(0), weights=g["n_runs"]),
        "mean_gained_en_route": np.average(g["mean_gained_en_route"].fillna(0), weights=g["n_runs"])}))
        .reset_index().sort_values("runs", ascending=False).astype({"trains": int, "runs": int}))
    month = t["by_month_zone"].groupby("month")[["n_obs"]].sum()
    month["mean_delay"] = (t["by_month_zone"].assign(s=lambda d: d["mean_delay"] * d["n_obs"])
                           .groupby("month")["s"].sum() / month["n_obs"])
    month["late_pct"] = (t["by_month_zone"].assign(s=lambda d: d["late_pct"] * d["n_obs"])
                         .groupby("month")["s"].sum() / month["n_obs"])
    return {
        "min_support": {"station_obs": min_st, "segment_late_arrivals": min_seg, "train_runs": min_tr},
        "network": {"mean_delay": round(float(np.average(t["by_date"]["mean_delay"], weights=t["by_date"]["n_obs"])), 2),
                    "late_pct": round(float(np.average(t["by_date"]["late_pct"], weights=t["by_date"]["n_obs"])), 2)},
        "worst_stations": _records(st.nlargest(15, "mean_delay"), st_cols),
        "delay_adding_stations": _records(st_clean.nlargest(10, "mean_change"), st_cols),
        "segments_adding_delay": _records(seg_clean.nlargest(10, "mean_change_when_late"), seg_cols),
        "segments_recovering_delay": _records(seg_clean.nsmallest(10, "mean_change_when_late"), seg_cols),
        "suspect_stations": {"count": int(len(suspects)), "of_ranked": int(len(st)),
                             "segments_excluded": int(seg["touches_suspect"].sum()),
                             "top": _records(suspects.assign(gap=suspects["mean_change"].abs())
                                             .nlargest(10, "gap"),
                                             ["station_name", "station_full_name", "station_zone", "n_obs",
                                              "mean_change", "outbound_mean_change"])},
        "worst_trains": _records(tr.nlargest(15, "mean_terminus_delay"), tr_cols),
        "by_type": _records(by_type, ["type_code", "trains", "runs", "mean_terminus_delay", "mean_gained_en_route"]),
        "by_month": _records(month.reset_index(), ["month", "n_obs", "mean_delay", "late_pct"]),
        "by_dow": _records(t["by_dow"], ["day", "n_obs", "mean_delay", "late_pct"]),
        "by_hour": _records(t["by_hour"], ["hour", "n_obs", "mean_delay", "late_pct"]),
        "cascade": _records(cas, ["h", "initial_delay", "n_pairs", "mean_delay0", "mean_delay_h", "mean_change",
                                  "late_pct_h", "lower_pct_h"]),
    }


def write_report(r: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.PROFILES_JSON.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    C.PROFILES_MD.write_text(render_markdown(r), encoding="utf-8")
    m, n = r["meta"], r["network"]
    print(f"\n=== Profiles ({m['source']} -> {m['output_dir']}) ===")
    print(f"  {m['stops']:,} stops, {m['trains']:,} trains, {m['stations']:,} stations, {m['segments']:,} segments")
    print(f"  network mean delay {n['mean_delay']} min, {n['late_pct']}% late  ({m['elapsed_s']} s)")


def _md_table(rows: list[dict], headers: dict[str, str]) -> list[str]:
    if not rows:
        return ["(none with enough support)"]
    cols = list(headers)
    num = [isinstance(rows[0].get(c), (int, float)) for c in cols]
    L = ["| " + " | ".join(headers.values()) + " |", "|" + "|".join("---:" if x else "---" for x in num) + "|"]
    for r in rows:
        L.append("| " + " | ".join("" if r.get(c) is None else f"{r[c]:,}" if isinstance(r[c], int)
                                   else str(r[c]) for c in cols) + " |")
    return L


def render_markdown(r: dict) -> str:
    m, ms, n = r["meta"], r["min_support"], r["network"]
    H_ST = {"station_name": "Code", "station_full_name": "Station", "station_zone": "Zone", "n_obs": "Arrivals",
            "mean_delay": "Mean", "p50_delay": "Median", "p90_delay": "p90", "late_pct": "% late",
            "mean_change": "Mean change", "recovery_pct_when_late": "% late arrivals that recover"}
    H_SEG = {"from_station": "From", "to_station": "To", "n_obs": "Pairs", "mean_sched_min": "Sched min",
             "mean_change": "Mean change", "mean_change_when_late": "Mean change if late",
             "recovery_pct_when_late": "% late that recover"}
    H_TR = {"train_no": "Train", "train_name": "Name", "type_code": "Type", "n_runs": "Runs", "route_km": "Km",
            "mean_origin_delay": "Origin delay", "mean_terminus_delay": "Terminus delay",
            "mean_gained_en_route": "Gained en route", "terminus_late_pct": "% late at terminus"}
    L = ["# Delay and route profiles (Phase 4)", "",
         f"Source `{m['source']}`: {m['stops']:,} stops of {m['trains']:,} trains at {m['stations']:,} stations "
         f"({m['segments']:,} station pairs). Tables in `{m['output_dir']}/`. All delays in minutes.", "",
         f"- Network: mean arrival delay **{n['mean_delay']} min**, **{n['late_pct']}%** of arrivals late > {LATE} min.",
         "- *Change* = delay at the stop minus delay at the previous observed stop of the same run "
         "(negative = time recovered).",
         f"- Rankings need support: stations >= {ms['station_obs']:,} arrivals, segments >= "
         f"{ms['segment_late_arrivals']:,} late arrivals, trains >= {ms['train_runs']:,} runs.", "",
         "## How far does a delay cascade?", "",
         f"For every observed stop, the delay h observed stops later in the same run, by the delay at the start.", ""]
    L += _md_table(r["cascade"], {"h": "h", "initial_delay": "Initial delay", "n_pairs": "Pairs",
                                  "mean_delay0": "Mean initial", "mean_delay_h": "Mean after h",
                                  "mean_change": "Change", "late_pct_h": "% late after h",
                                  "lower_pct_h": "% lower after h"})
    L += ["", "## Stations with the highest mean arrival delay", ""] + _md_table(r["worst_stations"], H_ST)
    L += ["", "## Stations where trains gain the most delay (mean change on arrival)", ""] + \
        _md_table(r["delay_adding_stations"], H_ST)
    L += ["", "## Segments where late trains lose the most time", ""] + _md_table(r["segments_adding_delay"], H_SEG)
    L += ["", "## Segments where late trains recover the most time", ""] + \
        _md_table(r["segments_recovering_delay"], H_SEG)
    sus = r["suspect_stations"]
    L += ["", "## Suspected timetable errors (sawtooth stations)", "",
          f"{sus['count']:,} of {sus['of_ranked']:,} ranked stations show a change on arrival of at least "
          f"{C.SAWTOOTH_MIN_CHANGE} min that the next segment undoes (opposite sign, also >= "
          f"{C.SAWTOOTH_MIN_CHANGE} min). The recorded delay there is systematically off, most likely because the "
          f"timetable time at that station is wrong, so it is not real delay or recovery. These stations and the "
          f"{sus['segments_excluded']:,} ranked segments touching them are left out of the change-based rankings "
          "above, as are stations whose arrivals mostly come from such a station (they inherit the mirror-image "
          "error). Everything stays in the tables, flagged.", ""]
    L += _md_table(sus["top"], {"station_name": "Code", "station_full_name": "Station", "station_zone": "Zone",
                                "n_obs": "Arrivals", "mean_change": "Change on arrival",
                                "outbound_mean_change": "Change on next segment"})
    L += ["", "## Trains with the highest mean terminus delay", ""] + _md_table(r["worst_trains"], H_TR)
    L += ["", "## By train type (weighted by runs)", ""] + _md_table(
        r["by_type"], {"type_code": "Type", "trains": "Trains", "runs": "Runs",
                       "mean_terminus_delay": "Terminus delay", "mean_gained_en_route": "Gained en route"})
    cal = {"n_obs": "Arrivals", "mean_delay": "Mean delay", "late_pct": "% late"}
    L += ["", "## Calendar", "", "**Month** (of the scheduled arrival)", ""] + _md_table(r["by_month"], {"month": "Month", **cal})
    L += ["", "**Day of week**", ""] + _md_table(r["by_dow"], {"day": "Day", **cal})
    L += ["", "**Hour of scheduled arrival**", ""] + _md_table(r["by_hour"], {"hour": "Hour", **cal})
    L += ["", f"Elapsed: {m['elapsed_s']} s."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="build from clean.parquet instead of the dev sample")
    build(full=ap.parse_args().full)


if __name__ == "__main__":
    main()
