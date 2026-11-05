"""Phase 2: leak-free features for predicting the arrival delay at stop k of a train run.

Usage:  python -m trackcast.features [--full] [--horizon H]

  default   data/processed/sample.parquet -> features_sample.parquet  (development)
  --full    data/processed/clean.parquet  -> features.parquet         (one month in memory at a time)

Information rule: a row for stop k may use
  * delays observed at stops <= k - H of the same run (H = horizon, default 1 = "next stop" forecast);
  * anything in the timetable (scheduled times, distances, slack, congestion), which is published in advance;
  * the scheduled time of stop k (never the actual time, which is scheduled time + the target).
It never uses the delay at stop k or later, nor any statistic fitted on delays across runs
(station target encoding happens in Phase 3, inside the CV folds).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trackcast import config as C
from trackcast import data as D

RUN_SORT = ["run_id", "station_no"]

ID_COLUMNS = ["run_id", "train_no", "date", "station_no", "sched_event_ts"]
CATEGORICAL_FEATURES = ["station_name", "type_code", "station_zone"]
CASCADE_FEATURES = ["lag1", "lag2", "delay_velocity", "lag_stops_back", "lag_sched_min_back",
                    "lag_from_origin", "upstream_max_delay", "n_upstream_obs"]
SLACK_FEATURES = ["seg_km", "seg_sched_min", "seg_speed_kmph", "seg_n_trains", "run_slack_min",
                  "dwell_prev_min", "slack_since_lag"]
ROUTE_FEATURES = ["station_no", "distance_from_origin", "route_frac", "n_sched_stops", "is_terminus"]
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]
SEASON_FEATURES = ["fog_season", "fog_night", "monsoon"]
CONGESTION_FEATURES = ["station_trains_hour", "station_trains_day"]
NUMERIC_FEATURES = (CASCADE_FEATURES + SLACK_FEATURES + ROUTE_FEATURES + TIME_FEATURES
                    + SEASON_FEATURES + CONGESTION_FEATURES)
TARGETS = ["delay", "y_late", "y_log"]
OUTPUT_COLUMNS = list(dict.fromkeys(ID_COLUMNS + CATEGORICAL_FEATURES + NUMERIC_FEATURES + TARGETS))

SCHED_FEATURE_COLUMNS = ["seg_km", "seg_sched_min", "seg_speed_kmph", "seg_n_trains", "run_slack_min",
                         "dwell_prev_min", "cum_slack_min", "route_frac", "n_sched_stops",
                         "station_trains_hour", "station_trains_day"]


# --------------------------------------------------------------------------------------------------
# Timetable features (one row per scheduled stop; no delays involved)
# --------------------------------------------------------------------------------------------------
def schedule_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """Per (train_no, station_no): segment, slack, route and congestion features from the timetable.

    Slack is measured against the fastest *scheduled* alternative, so it needs no delay data:
      run_slack_min    scheduled run time on (prev station -> station) minus the minimum over all trains
                       scheduled on that same station pair;
      dwell_slack      scheduled dwell minus the minimum dwell any train has at that station;
      cum_slack_min    running total of the slack available before arriving at each stop, so the slack
                       between any two stops j < k is cum_slack_min[k] - cum_slack_min[j].
    """
    s = schedule.sort_values(["train_no", "station_no"], ignore_index=True).copy()
    s["station_name"] = s["station_name"].astype("string")
    g = s.groupby("train_no", sort=False)
    arr, dep = s["sched_arr_offset_min"], s["sched_dep_offset_min"]

    prev_station = g["station_name"].shift()
    s["seg_km"] = (s["distance_from_origin"] - g["distance_from_origin"].shift()).astype("float32")
    s["seg_sched_min"] = (arr - g["sched_dep_offset_min"].shift()).astype("float32")
    seg = s.groupby([prev_station, s["station_name"]])["seg_sched_min"]  # NaN prev (origin) -> dropped
    s["seg_n_trains"] = seg.transform("count").astype("float32")
    s["run_slack_min"] = (s["seg_sched_min"] - seg.transform("min")).clip(lower=0).astype("float32")
    s["seg_speed_kmph"] = (s["seg_km"] / s["seg_sched_min"].where(s["seg_sched_min"] > 0) * 60).astype("float32")

    dwell = (dep - arr).clip(lower=0)  # NaN at origin and terminus
    dwell_slack = (dwell - dwell.groupby(s["station_name"]).transform("min")).clip(lower=0)
    s["dwell_prev_min"] = dwell.groupby(s["train_no"]).shift().astype("float32")
    step = s["run_slack_min"].fillna(0) + dwell_slack.groupby(s["train_no"]).shift().fillna(0)
    s["cum_slack_min"] = step.groupby(s["train_no"]).cumsum().astype("float32")

    route_km = g["distance_from_origin"].transform("max")
    s["route_frac"] = (s["distance_from_origin"] / route_km.where(route_km > 0)).astype("float32")

    # Congestion: distinct trains with a scheduled arrival/departure at the station in the same clock hour.
    # The timetable has no running days, so every train counts as daily (an upper bound).
    event = arr.fillna(dep)
    hour = (event % 1440 // 60)
    s["station_trains_hour"] = (s.groupby([s["station_name"], hour])["train_no"].transform("nunique")
                                .astype("float32"))
    s["station_trains_day"] = s.groupby("station_name")["train_no"].transform("nunique").astype("float32")
    s["n_sched_stops"] = s["n_sched_stops"].astype("float32")
    return s[["train_no", "station_no", *SCHED_FEATURE_COLUMNS]]


# --------------------------------------------------------------------------------------------------
# Run-level features
# --------------------------------------------------------------------------------------------------
def _gather(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """values[idx] as float, NaN where idx < 0."""
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def lag_positions(run_start: np.ndarray, observed: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (or -1) of the last and second-last observed stops at or before row k - horizon,
    within the same run. Rows must be sorted by run and stop; run_start[i] = first row of i's run."""
    n = len(observed)
    pos = np.arange(n)
    last = np.maximum.accumulate(np.where(observed, pos, -1))   # last observed row <= i (any run)
    last = np.where(last >= run_start, last, -1)                 # ... restricted to i's run

    def back(src: np.ndarray, h: int) -> np.ndarray:             # src[i - h] if still in i's run
        out = np.full(n, -1)
        ok = pos - h >= run_start
        out[ok] = src[pos[ok] - h]
        return out

    j1 = back(last, horizon)
    prev_obs = back(last, 1)                                     # last observed row strictly before i
    j2 = np.where(j1 >= 0, prev_obs[np.maximum(j1, 0)], -1)
    return j1, j2


def build_features(clean: pd.DataFrame, sched_feats: pd.DataFrame, horizon: int = C.HORIZON_STOPS,
                   model_rows_only: bool = True) -> pd.DataFrame:
    """Features for every stop of complete runs in `clean` (Phase 1 schema).

    With model_rows_only, keep the rows a model can be trained and scored on: observed delay, not the
    origin (whose delay is a departure delay), and at least one observed upstream delay."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 (horizon 0 would use the target itself)")
    df = clean.sort_values(RUN_SORT, ignore_index=True)
    df = df.merge(sched_feats, on=["train_no", "station_no"], how="left", validate="many_to_one")

    run_start = (np.arange(len(df)) - df.groupby("run_id", sort=False).cumcount().to_numpy())
    delay = df["delay"].to_numpy(dtype="float64", na_value=np.nan)
    observed = ~np.isnan(delay)
    j1, j2 = lag_positions(run_start, observed, horizon)

    arr = df["sched_arr_offset_min"].to_numpy(dtype="float64", na_value=np.nan)
    dep = df["sched_dep_offset_min"].to_numpy(dtype="float64", na_value=np.nan)
    event = np.where(np.isnan(arr), dep, arr)          # scheduled time of stop k (arrival; origin: departure)
    leave = np.where(np.isnan(dep), arr, dep)          # scheduled time the train leaves the lag stop
    station_no = df["station_no"].to_numpy(dtype="float64")
    cum_slack = df["cum_slack_min"].to_numpy(dtype="float64", na_value=np.nan)
    upstream_max = df.groupby("run_id", sort=False)["delay"].cummax().to_numpy(dtype="float64", na_value=np.nan)
    n_obs = np.cumsum(observed) - np.where(run_start > 0, np.cumsum(observed)[np.maximum(run_start - 1, 0)], 0)

    f = df
    f["lag1"] = _gather(delay, j1)
    f["lag2"] = _gather(delay, j2)
    f["delay_velocity"] = f["lag1"] - f["lag2"]
    f["lag_stops_back"] = station_no - _gather(station_no, j1)
    f["lag_sched_min_back"] = event - _gather(leave, j1)
    f["lag_from_origin"] = _gather(df["is_origin"].to_numpy(dtype="float64"), j1)
    f["upstream_max_delay"] = _gather(upstream_max, j1)
    f["n_upstream_obs"] = _gather(n_obs.astype("float64"), j1)
    f["slack_since_lag"] = cum_slack - _gather(cum_slack, j1)

    ts = df["date"] + pd.to_timedelta(event, unit="min")
    f["sched_event_ts"] = ts
    hour = (event % 1440) / 60
    dow = ts.dt.day_of_week.to_numpy(dtype="float64", na_value=np.nan)
    month = ts.dt.month.to_numpy(dtype="float64", na_value=np.nan)
    for name, v, period in (("hour", hour, 24), ("dow", dow, 7), ("month", month - 1, 12)):
        f[f"{name}_sin"] = np.sin(2 * np.pi * v / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * v / period)
    f["is_weekend"] = dow >= 5

    zone = df["station_zone"].astype("string")
    month_s = pd.Series(month, index=df.index)
    lo, hi = C.FOG_NIGHT_HOURS
    f["fog_season"] = (month_s.isin(C.FOG_MONTHS) & zone.isin(C.FOG_ZONES).fillna(False)).to_numpy(bool)
    f["fog_night"] = f["fog_season"] & ((hour >= lo) | (hour < hi))
    f["monsoon"] = month_s.isin(C.MONSOON_MONTHS).to_numpy(bool)

    f["y_late"] = f["delay"] > C.SIGNIFICANT_DELAY_MIN
    f["y_log"] = np.log1p(f["delay"])

    if model_rows_only:
        keep = observed & ~df["is_origin"].to_numpy(bool) & (j1 >= 0)
        f = f.loc[keep]
    f = f[OUTPUT_COLUMNS].reset_index(drop=True)
    float_cols = [c for c in NUMERIC_FEATURES if f[c].dtype.kind == "f"] + ["y_log"]
    return f.astype({c: "float32" for c in float_cols})


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def _to_arrow(df: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    t = pa.Table.from_pandas(df, preserve_index=False)
    if schema is None:  # fix dictionary index width so months with different category counts concatenate
        schema = pa.schema([pa.field(f.name, pa.dictionary(pa.int32(), pa.string()))
                            if pa.types.is_dictionary(f.type) else f for f in t.schema])
    return t.cast(schema)


def iter_months(path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[start, end) month windows covering the `date` range of a parquet file (from row-group stats)."""
    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.get_field_index("date")
    lo = min(pf.metadata.row_group(i).column(col).statistics.min for i in range(pf.num_row_groups))
    hi = max(pf.metadata.row_group(i).column(col).statistics.max for i in range(pf.num_row_groups))
    starts = pd.date_range(pd.Timestamp(lo).to_period("M").to_timestamp(), pd.Timestamp(hi), freq="MS")
    return [(s, s + pd.offsets.MonthBegin()) for s in starts]


def build(full: bool = False, horizon: int = C.HORIZON_STOPS) -> dict:
    t0 = time.time()
    src = C.CLEAN_PARQUET if full else C.SAMPLE_PARQUET
    out = C.FEATURES_PARQUET if full else C.FEATURES_SAMPLE_PARQUET
    sched_feats = schedule_features(D.load_schedule(D.QualityStats()))
    total_src = pq.ParquetFile(src).metadata.num_rows
    rng = np.random.default_rng(C.RANDOM_STATE)
    p = min(1.0, C.REPORT_SUBSAMPLE_ROWS / max(total_src, 1))

    acc = {"source_rows": 0, "rows": 0, "nonnull": pd.Series(0, index=NUMERIC_FEATURES), "late": 0}
    keep_for_report = []
    tmp = out.with_name(out.name + ".tmp")
    writer = None
    windows = iter_months(src) if full else [(None, None)]
    try:
        for start, end in windows:
            if start is None:
                clean = pd.read_parquet(src)
            else:
                clean = pd.read_parquet(src, filters=[("date", ">=", start), ("date", "<", end)])
            if clean.empty:
                continue
            feats = build_features(clean, sched_feats, horizon)
            acc["source_rows"] += len(clean)
            acc["rows"] += len(feats)
            acc["late"] += int(feats["y_late"].sum())
            acc["nonnull"] += feats[NUMERIC_FEATURES].notna().sum()
            keep_for_report.append(feats[rng.random(len(feats)) < p])
            table = _to_arrow(feats, None if writer is None else writer.schema)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            if start is not None:
                print(f"  {start:%Y-%m}: {len(clean):,} stops -> {len(feats):,} feature rows")
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)

    report = summarize(pd.concat(keep_for_report, ignore_index=True), acc)
    report["meta"] = {"source": src.name, "output": out.name, "horizon_stops": horizon,
                      "elapsed_s": round(time.time() - t0, 1)}
    write_report(report)
    return report


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def persistence_baseline(f: pd.DataFrame) -> dict:
    """Predict 'delay at k = last observed upstream delay'. Any model in Phase 3 has to beat this."""
    pred_log = np.log1p(f["lag1"].astype("float64"))
    y = f["y_log"].astype("float64")
    rmsle = float(np.sqrt(np.mean((pred_log - y) ** 2)))
    r2 = float(1 - np.sum((y - pred_log) ** 2) / np.sum((y - y.mean()) ** 2))
    pred_late, late = f["lag1"] > C.SIGNIFICANT_DELAY_MIN, f["y_late"]
    tp = int((pred_late & late).sum())
    prec = tp / max(int(pred_late.sum()), 1)
    rec = tp / max(int(late.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"rmsle": round(rmsle, 4), "r2_log": round(r2, 4), "f1_late": round(f1, 4),
            "precision_late": round(prec, 4), "recall_late": round(rec, 4)}


def slack_recovery_table(f: pd.DataFrame) -> list[dict]:
    """Change in delay between the lag stop and stop k, by scheduled slack in between (trains already late)."""
    late = f[f["lag1"] > C.SIGNIFICANT_DELAY_MIN]
    bins = [-np.inf, 0, 2, 5, 10, 20, 40, np.inf]
    labels = ["0", "(0,2]", "(2,5]", "(5,10]", "(10,20]", "(20,40]", ">40"]
    b = pd.cut(late["slack_since_lag"], bins=bins, labels=labels)
    change = late["delay"] - late["lag1"]
    t = pd.DataFrame({"b": b, "change": change, "recovered": change < 0}).groupby("b", observed=False)
    out = t.agg(rows=("change", "size"), mean_change=("change", "mean"), median_change=("change", "median"),
                pct_recovered=("recovered", "mean"))
    out["pct_recovered"] *= 100
    return [{"slack_min": k, **{c: (None if pd.isna(v) else round(float(v), 2)) for c, v in row.items()}}
            for k, row in out.iterrows()]


def summarize(sub: pd.DataFrame, acc: dict) -> dict:
    num = sub[NUMERIC_FEATURES].astype("float64")
    spearman = num.corrwith(sub["y_log"].astype("float64"), method="spearman")
    desc = num.describe(percentiles=[0.5]).T
    features = {
        c: {"nonnull_pct": round(100 * acc["nonnull"][c] / max(acc["rows"], 1), 2),
            "mean": _r(desc.loc[c, "mean"]), "median": _r(desc.loc[c, "50%"]),
            "min": _r(desc.loc[c, "min"]), "max": _r(desc.loc[c, "max"]),
            "spearman_vs_y_log": _r(spearman.get(c))}
        for c in NUMERIC_FEATURES
    }
    return {
        "rows": {"source_stops": acc["source_rows"], "feature_rows": acc["rows"],
                 "late_pct": round(100 * acc["late"] / max(acc["rows"], 1), 2),
                 "report_subsample": len(sub)},
        "features": features,
        "groups": {"cascade": CASCADE_FEATURES, "slack": SLACK_FEATURES, "route": ROUTE_FEATURES,
                   "time": TIME_FEATURES, "season": SEASON_FEATURES, "congestion": CONGESTION_FEATURES,
                   "categorical": CATEGORICAL_FEATURES},
        "persistence_baseline": persistence_baseline(sub),
        "slack_recovery": slack_recovery_table(sub),
        "lag_gap": {"lag_stops_back_gt1_pct": round(100 * float((sub["lag_stops_back"] > 1).mean()), 2),
                    "lag2_missing_pct": round(100 * float(sub["lag2"].isna().mean()), 2)},
    }


def _r(v, nd: int = 4):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def write_report(r: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FEATURES_JSON.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    C.FEATURES_MD.write_text(render_markdown(r), encoding="utf-8")
    m, rows, b = r["meta"], r["rows"], r["persistence_baseline"]
    print(f"\n=== Features ({m['source']} -> {m['output']}, horizon {m['horizon_stops']}) ===")
    print(f"  {rows['source_stops']:,} stops -> {rows['feature_rows']:,} feature rows, "
          f"{rows['late_pct']}% late > {C.SIGNIFICANT_DELAY_MIN} min")
    print(f"  persistence baseline: RMSLE {b['rmsle']}, R2(log) {b['r2_log']}, F1(late) {b['f1_late']}")
    print(f"  report: {C.FEATURES_MD}  ({m['elapsed_s']} s)")


def render_markdown(r: dict) -> str:
    m, rows, b = r["meta"], r["rows"], r["persistence_baseline"]
    L = ["# Feature report (Phase 2)", "",
         f"Source `{m['source']}` -> `{m['output']}`, horizon = {m['horizon_stops']} stop(s): stop k is predicted "
         f"from delays observed at stops <= k - {m['horizon_stops']} plus the timetable.", "",
         f"- {rows['source_stops']:,} stops -> **{rows['feature_rows']:,} model rows** (observed delay, not the "
         f"origin, at least one observed upstream delay).",
         f"- Late > {C.SIGNIFICANT_DELAY_MIN} min: {rows['late_pct']}%.",
         f"- Lag taken from further back than the previous stop: {r['lag_gap']['lag_stops_back_gt1_pct']}% of rows; "
         f"lag2 missing: {r['lag_gap']['lag2_missing_pct']}%.",
         f"- Statistics below use a random subsample of {rows['report_subsample']:,} rows.", "",
         "## Persistence baseline (predict delay_k = lag1)", "",
         "| RMSLE | R2 (log1p) | F1 (late) | Precision | Recall |", "|---:|---:|---:|---:|---:|",
         f"| {b['rmsle']} | {b['r2_log']} | {b['f1_late']} | {b['precision_late']} | {b['recall_late']} |", "",
         "Delay is highly persistent stop to stop, so this naive forecast is strong. Phase 3 models are judged by "
         "how much they improve on it, not by their absolute R2.", "",
         "## Does schedule slack absorb delay?", "",
         f"Trains already late (lag1 > {C.SIGNIFICANT_DELAY_MIN} min), by scheduled slack between the lag stop and "
         "stop k. Change = delay_k - lag1 (negative = recovered).", "",
         "| Slack (min) | Rows | Mean change | Median change | % recovered |", "|---|---:|---:|---:|---:|"]
    L += [f"| {s['slack_min']} | {int(s['rows'] or 0):,} | {s['mean_change']} | {s['median_change']} | "
          f"{s['pct_recovered']} |" for s in r["slack_recovery"]]
    L += ["", "## Features", ""]
    for group, cols in r["groups"].items():
        if group == "categorical":
            L += [f"**categorical** (encoded in Phase 3): {', '.join(cols)}", ""]
            continue
        L += [f"**{group}**", "", "| Feature | Non-null % | Mean | Median | Min | Max | Spearman vs log1p(delay) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for c in cols:
            s = r["features"][c]
            L += [f"| {c} | {s['nonnull_pct']} | {s['mean']} | {s['median']} | {s['min']} | {s['max']} | "
                  f"{s['spearman_vs_y_log']} |"]
        L += [""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="build from clean.parquet instead of the dev sample")
    ap.add_argument("--horizon", type=int, default=C.HORIZON_STOPS, help="stops ahead to forecast (>= 1)")
    args = ap.parse_args()
    build(full=args.full, horizon=args.horizon)


if __name__ == "__main__":
    main()
