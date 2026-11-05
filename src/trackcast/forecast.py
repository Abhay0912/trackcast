"""Phase 5 helpers: forecast a train's delay at its next stops from the delays observed so far.

The app never re-implements feature logic: it builds a Phase 1-shaped run (timetable + the delays the user
knows, NaN elsewhere) and calls the same `features.build_features` used for training. So a forecast in the
app is computed exactly like a test-set prediction.

Route forecast: the model is a 1-stop-ahead (horizon 1) model, so stops further ahead are forecast
recursively, feeding each forecast back in as the next stop's upstream delay. Which minutes are fed forward
matters: the Phase 3 point forecast expm1(E[log1p delay]) is median-like, and chained 20 times it shrinks a
late train's delay towards 0. Feeding the expected minutes, P(late) * expm1(m_late) + (1 - P(late)) *
expm1(m_low), was the most accurate from 2 stops ahead on 1.5k simulated test runs (`python -m
trackcast.forecast` -> reports/route_forecast.md), so it is the default. Uncertainty is not propagated, and
far-ahead forecasts still run low on average.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trackcast import config as C
from trackcast import data as D
from trackcast import features as F

RUN_COLUMNS = ["train_no", "date", "station_no", "station_name", "sched_arr_offset_min", "sched_dep_offset_min",
               "distance_from_origin", "is_origin", "is_terminus", "type_code", "station_zone"]


@dataclass
class Timetable:
    schedule: pd.DataFrame      # prepared schedule + type_code, train_name, station_zone, station_full_name
    sched_feats: pd.DataFrame   # features.schedule_features over the whole timetable (slack needs all trains)

    @classmethod
    def from_frames(cls, schedule: pd.DataFrame, trains: pd.DataFrame, stations: pd.DataFrame) -> "Timetable":
        s = (schedule.merge(trains, on="train_no", how="left", validate="many_to_one")
             .merge(stations, on="station_name", how="left", validate="many_to_one"))
        return cls(schedule=s.sort_values(["train_no", "station_no"], ignore_index=True),
                   sched_feats=F.schedule_features(schedule))

    @classmethod
    def load(cls) -> "Timetable":
        stats = D.QualityStats()
        return cls.from_frames(D.load_schedule(stats), D.load_trains(stats), D.load_stations(stats))

    def route(self, train_no: int) -> pd.DataFrame:
        r = self.schedule[self.schedule["train_no"] == train_no]
        if r.empty:
            raise KeyError(f"train {train_no} is not in the timetable")
        return r.reset_index(drop=True)


def run_frame(tt: Timetable, train_no: int, date, delays: dict[int, float]) -> pd.DataFrame:
    """A Phase 1-shaped run of `train_no` starting on `date`: known delays by station_no, NaN elsewhere."""
    r = tt.route(train_no).assign(date=pd.Timestamp(date).normalize())[RUN_COLUMNS].copy()
    r["delay"] = r["station_no"].map({int(k): float(v) for k, v in delays.items()}).astype("float32")
    r["run_id"] = D.make_run_id(r["train_no"].astype("int64"), r["date"])
    for c in ["station_name", "type_code", "station_zone"]:
        r[c] = r[c].astype("string").fillna("NA").astype("category")
    return r


def stop_features(tt: Timetable, train_no: int, date, delays: dict[int, float]) -> pd.DataFrame:
    """Feature rows for every stop of the run (no model-row filter), exactly as in training."""
    sf = tt.sched_feats[tt.sched_feats["train_no"] == train_no]   # slack minima already span all trains
    return F.build_features(run_frame(tt, train_no, date, delays), sf, horizon=1, model_rows_only=False)


def point_forecast(pred: pd.DataFrame, strategy: str, threshold: float) -> np.ndarray:
    """Minutes to report and to feed forward, from a HurdleModel prediction frame.

    median  expm1(E[log1p delay]): the RMSLE-optimal forecast for one stop (what Phase 3 scored)
    mean    P(late) * expm1(m_late) + (1 - P(late)) * expm1(m_low): expected delay in minutes
    gate    expm1(m_late) if P(late) >= threshold else expm1(m_low): the more likely branch
    """
    if strategy == "median":
        v = pred["pred_delay_min"].to_numpy("float64")
    else:
        p = pred["p_late"].to_numpy("float64")
        late, low = np.expm1(pred["m_late_log"].to_numpy("float64")), np.expm1(pred["m_low_log"].to_numpy("float64"))
        if strategy == "mean":
            v = p * late + (1 - p) * low
        elif strategy == "gate":
            v = np.where(p >= threshold, late, low)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    return np.clip(v, 0, None)


def forecast_route(model, tt: Timetable, train_no: int, date, observed: dict[int, float],
                   max_stops: int | None = None, strategy: str = C.ROUTE_FORECAST_STRATEGY) -> pd.DataFrame:
    """Forecast every stop after the last observed one (recursively, 1 stop ahead at a time).

    observed: {station_no: delay in minutes} for stops already passed (at least one).
    Returns one row per scheduled stop: observed delay, or forecast (p_late, pred_delay_min)."""
    if not observed:
        raise ValueError("need at least one observed delay")
    route = tt.route(train_no)
    last = max(observed)
    ahead = route.loc[route["station_no"] > last, "station_no"].tolist()[:max_stops]
    delays = {int(k): max(float(v), 0.0) for k, v in observed.items()}
    threshold = getattr(model, "threshold", 0.5)
    rows = []
    for k in ahead:
        feats = stop_features(tt, train_no, date, delays)
        row = feats[feats["station_no"] == k]
        pred = model.predict(row)
        delays[k] = float(point_forecast(pred, strategy, threshold)[0])
        rows.append({"station_no": k, "p_late": float(pred["p_late"].iloc[0]), "pred_delay_min": delays[k],
                     "slack_since_lag": float(row["slack_since_lag"].iloc[0])})
    out = route[["station_no", "station_name", "station_full_name", "distance_from_origin",
                 "sched_arr_offset_min", "sched_dep_offset_min"]].copy()
    out["observed_delay"] = out["station_no"].map(observed).astype("float64")
    fc = pd.DataFrame(rows, columns=["station_no", "p_late", "pred_delay_min", "slack_since_lag"])
    out = out.merge(fc, on="station_no", how="left")
    out["delay"] = out["observed_delay"].fillna(out["pred_delay_min"])
    out["kind"] = np.where(out["observed_delay"].notna(), "observed",
                           np.where(out["pred_delay_min"].notna(), "forecast", "not used"))
    return out


def sched_clock(offset_min) -> str:
    """'HH:MM (+d)' for a schedule offset in minutes from the origin day's midnight."""
    if offset_min is None or pd.isna(offset_min):
        return ""
    day, minute = divmod(int(round(float(offset_min))), 1440)
    return f"{minute // 60:02d}:{minute % 60:02d}" + (f" (+{day}d)" if day else "")


# --------------------------------------------------------------------------------------------------
# Route-forecast evaluation: simulate the app on real test-period runs
#   python -m trackcast.forecast   ->   reports/route_forecast.md (+ .json)
# --------------------------------------------------------------------------------------------------
SIM_COLUMNS = ["run_id", "train_no", "date", "station_no", "station_name", "delay", "sched_arr_offset_min",
               "sched_dep_offset_min", "distance_from_origin", "is_origin", "is_terminus", "type_code",
               "station_zone"]


def simulate_routes(model, clean: pd.DataFrame, sched_feats: pd.DataFrame, steps: int, strategy: str,
                    seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """For each run: pick a random observed start stop, hide every other delay (as in the app, only the start
    is known), forecast `steps` stops ahead recursively and return (run_id, h, initial, actual, pred)."""
    df = clean.sort_values(["run_id", "station_no"], ignore_index=True)
    pos = df.groupby("run_id", sort=False).cumcount().to_numpy()
    n = df.groupby("run_id", sort=False)["station_no"].transform("size").to_numpy()
    ok = df["delay"].notna().to_numpy() & (pos >= 1) & (pos <= n - 1 - steps)
    starts = df[ok].groupby("run_id", sort=False).sample(1, random_state=seed)
    s0 = df["run_id"].map(pd.Series(pos[starts.index], index=starts["run_id"]))
    keep = s0.notna().to_numpy()
    df, pos, s0 = df[keep].reset_index(drop=True), pos[keep], s0[keep].to_numpy().astype(int)
    actual = df["delay"].to_numpy("float64")
    df["delay"] = np.where(pos == s0, df["delay"], np.nan).astype("float32")
    initial = pd.Series(actual[pos == s0], index=df.loc[pos == s0, "run_id"].to_numpy())
    threshold = getattr(model, "threshold", 0.5)
    out = []
    for h in range(1, steps + 1):
        feats = F.build_features(df, sched_feats, horizon=1, model_rows_only=False)
        if not np.array_equal(feats["station_no"].to_numpy(), df["station_no"].to_numpy()):
            raise RuntimeError("feature rows are not aligned with the simulated runs")
        rows = pos == s0 + h
        pred = point_forecast(model.predict(feats.loc[rows]), strategy, threshold)
        df.loc[rows, "delay"] = pred.astype("float32")
        rid = df.loc[rows, "run_id"].to_numpy()
        out.append(pd.DataFrame({"run_id": rid, "h": h, "initial": initial.loc[rid].to_numpy(),
                                 "actual": actual[rows], "pred": pred}))
    return pd.concat(out, ignore_index=True)


def _path_metrics(r: pd.DataFrame, pred_col: str) -> dict:
    r = r[r["actual"].notna()]
    err = r[pred_col] - r["actual"]
    sl = np.log1p(r[pred_col].clip(lower=0)) - np.log1p(r["actual"])
    return {"n": int(len(r)), "mae_min": float(err.abs().mean()), "bias_min": float(err.mean()),
            "rmsle": float(np.sqrt((sl ** 2).mean()))}


def evaluate_route_forecasts(n_runs: int = C.ROUTE_EVAL_RUNS, steps: int = C.ROUTE_EVAL_STEPS) -> dict:
    import json
    import time

    import joblib
    import pyarrow.parquet as pq

    t0 = time.time()
    full = C.MODEL_PATH.exists() and C.CLEAN_PARQUET.exists()
    model = joblib.load(C.MODEL_PATH if full else C.MODEL_SAMPLE_PATH)
    src = C.CLEAN_PARQUET if full else C.SAMPLE_PARQUET
    test_start = pd.Timestamp(C.TEST_START_DATE)
    keys = pq.read_table(src, columns=["run_id", "date"], filters=[("date", ">=", test_start)]).to_pandas()
    size = keys.groupby("run_id").size()
    long_runs = size[size >= steps + 3].index.to_numpy()
    u = pd.util.hash_array(long_runs.astype("int64")) / 2.0 ** 64
    chosen = long_runs[np.argsort(u)[:n_runs]]
    clean = pq.read_table(src, columns=SIM_COLUMNS, filters=[("run_id", "in", chosen.tolist())]).to_pandas()
    sched_feats = F.schedule_features(D.load_schedule(D.QualityStats()))
    sched_feats = sched_feats[sched_feats["train_no"].isin(clean["train_no"].unique())]

    sims = {}
    for s in ("median", "mean", "gate"):
        sims[s] = simulate_routes(model, clean, sched_feats, steps, s)
        print(f"  simulated '{s}': {len(sims[s]):,} forecasts ({time.time() - t0:.0f} s)")
    base = sims["median"][["run_id", "h", "initial", "actual"]].copy()
    base["persistence"] = base["initial"]
    prof = C.PROFILES_DIR / "cascade.parquet"
    if prof.exists():
        cas = pd.read_parquet(prof).set_index(["h", "bucket"])["mean_change"]
        bucket = np.searchsorted(np.asarray(C.CASCADE_BUCKETS, "float64"), base["initial"].to_numpy(), side="left")
        idx = pd.MultiIndex.from_arrays([base["h"].to_numpy(), bucket])
        base["cascade_profile"] = np.clip(base["initial"].to_numpy() + cas.reindex(idx).to_numpy(), 0, None)

    horizons = [h for h in (1, 2, 3, 5, 10, 15, 20) if h <= steps]
    table = []
    for h in horizons:
        row = {"h": h}
        for s, r in sims.items():
            row[s] = _path_metrics(r[r["h"] == h], "pred")
        b = base[base["h"] == h]
        for col in ("persistence", "cascade_profile"):
            if col in b:
                row[col] = _path_metrics(b, col)
        table.append(row)
    report = {"meta": {"source": src.name, "model": (C.MODEL_PATH if full else C.MODEL_SAMPLE_PATH).name,
                       "runs": int(clean["run_id"].nunique()), "steps": steps, "test_start": C.TEST_START_DATE,
                       "elapsed_s": round(time.time() - t0, 1)}, "by_horizon": table}
    C.ROUTE_FORECAST_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    C.ROUTE_FORECAST_MD.write_text(_render_route_report(report), encoding="utf-8")
    print(_render_route_report(report))
    return report


def _render_route_report(r: dict) -> str:
    m = r["meta"]
    names = [k for k in r["by_horizon"][0] if k != "h"]
    L = ["# Route-forecast evaluation (Phase 5)", "",
         f"The app's multi-stop forecast, simulated on {m['runs']:,} real runs from the test period (starting on or "
         f"after {m['test_start']}, never seen in training; `{m['source']}`, model `{m['model']}`). Each run starts "
         f"at a random observed stop; only that stop's delay is known (as in the app), and the next "
         f"{m['steps']} stops are forecast recursively. Strategies differ only in which minutes are fed forward:",
         "", "- *median*: expm1(E[log1p delay]) (the Phase 3 point forecast)",
         "- *mean*: P(late) x expm1(m_late) + (1 - P(late)) x expm1(m_low)",
         "- *gate*: the branch the calibrated classifier picks (m_late if P(late) >= threshold, else m_low)",
         "- *persistence*: the start delay, unchanged",
         "- *cascade_profile*: start delay + the network-average change after h stops for its delay bucket "
         "(Phase 4; computed on all dates, so this baseline has seen the test months)", ""]
    for metric, title in (("mae_min", "MAE (minutes)"), ("bias_min", "Bias: mean forecast - actual (minutes)"),
                          ("rmsle", "RMSLE")):
        L += [f"**{title}**", "", "| h | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
        for row in r["by_horizon"]:
            vals = [row[n][metric] for n in names]
            best = min(vals, key=abs) if metric == "bias_min" else min(vals)
            L.append(f"| {row['h']} | " + " | ".join((f"**{v:.2f}**" if v == best else f"{v:.2f}")
                                                     for v in vals) + " |")
        L.append("")
    L.append(f"Rows per horizon: {r['by_horizon'][0][names[0]]['n']:,} (h = 1). Elapsed: {m['elapsed_s']} s.")
    return "\n".join(L)


if __name__ == "__main__":
    evaluate_route_forecasts()
