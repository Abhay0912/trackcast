"""Phase 7: results per train class and route, schedule recovery, and bottleneck junctions.

Usage:  python -m trackcast.results [--full]

  default   dev sample: test_predictions_sample, profiles_sample/, eda_sample/, sample.parquet
  --full    full year:  test_predictions,        profiles/,        eda/,        clean.parquet

Run after `train`, `profiles` and `eda` with the same flag. Writes data/processed/results[_sample]/*.parquet,
reports/results.md / .json and reports/figures/results_*.png.

What it answers:
  1. Results per train class and per route. The Phase 3 model's test-set scores (RMSLE, MAE, Stage 1 F1 and
     the skill over persistence with a day-block bootstrap interval) per train class, corridor, station
     group and train, next to what that class or corridor does over the year (Phase 6: delay, recovery
     rate, congestion sensitivity).
  2. Bottleneck junctions. For every train passing straight through a station (the previous and the next
     scheduled stop both observed in the same run), the change in delay from the stop before to the stop
     after: approach, platform dwell and exit together. Arrival-only change (Phase 4 / 6) misleads at big
     stations, because timetables pad the approach: a train "recovers" on arrival and loses it again
     leaving. Junctions and major stations are ranked by the delay-minutes they add per day.

The clean table is streamed one group of trains at a time (as in Phase 4); the through-station sums are
additive, so group partials merge exactly. Nothing here is a model input, so nothing can leak.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.figure import Figure

from trackcast import config as C
from trackcast import eda as E
from trackcast import profiles as P

LATE = C.SIGNIFICANT_DELAY_MIN
THROUGH_COLUMNS = ["run_id", "train_no", "station_no", "station_name", "delay"]
THROUGH_SUMS = ["n", "sum_arr", "sum_dep", "sum_thr", "n_late_in", "n_recovered", "n_ontime_in", "n_made_late"]
STATION_GROUPS = ["Bottlenecks", "Absorbers", "Other junctions and major stations", "Minor halts"]
TABLES = ["through", "scores_class", "scores_corridor", "scores_station_group", "scores_train"]


# --------------------------------------------------------------------------------------------------
# Through-station delay change (one group of trains, complete runs)
# --------------------------------------------------------------------------------------------------
def through_partial(clean: pd.DataFrame, suspect: set[str] | frozenset = frozenset()) -> pd.DataFrame:
    """Per station, sums over trains passing straight through it.

    a, b, c = delay at the previous scheduled stop, at the station, and at the next scheduled stop, all
    observed in the same run (station_no k-1, k, k+1). arr = b - a (approach), dep = c - b (dwell + exit),
    thr = c - a (both). Triples touching a suspected timetable error (Phase 4 sawtooth) are dropped: the
    recorded delay there is off, so any change across it is an artefact."""
    obs = clean.loc[clean["delay"].notna(), ["run_id", "station_no", "station_name", "delay"]]
    obs = obs.sort_values(["run_id", "station_no"], ignore_index=True)
    if len(obs) < 3:
        return pd.DataFrame(columns=THROUGH_SUMS, index=pd.Index([], name="station_name", dtype="string"))
    rid = obs["run_id"].to_numpy()
    no = obs["station_no"].to_numpy("int64")
    d = obs["delay"].to_numpy("float64")
    stn = obs["station_name"].astype("string").to_numpy(dtype=object, na_value="")
    ok = ((rid[:-2] == rid[1:-1]) & (rid[2:] == rid[1:-1])
          & (no[1:-1] - no[:-2] == 1) & (no[2:] - no[1:-1] == 1))
    if suspect:
        sus = np.isin(stn, list(suspect))
        ok &= ~(sus[:-2] | sus[1:-1] | sus[2:])
    a, b, c = d[:-2][ok], d[1:-1][ok], d[2:][ok]
    late_in = a > LATE
    sums = pd.DataFrame({
        "n": 1, "sum_arr": b - a, "sum_dep": c - b, "sum_thr": c - a,
        "n_late_in": late_in.astype(int), "n_recovered": (late_in & (c < a)).astype(int),
        "n_ontime_in": (~late_in).astype(int), "n_made_late": (~late_in & (c > LATE)).astype(int)})
    return sums.groupby(pd.Series(stn[1:-1][ok], name="station_name", dtype="string")).sum()


def merge_through(parts: list[pd.DataFrame]) -> pd.DataFrame:
    x = pd.concat([p for p in parts if len(p)])
    return x.groupby(level=0).sum()


def through_rates(s: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Rates from the through sums. minutes_per_day = total delay added by trains passing through, per day."""
    n = s["n"].where(s["n"] > 0)
    return pd.DataFrame({
        "n_through": s["n"], "through_per_day": s["n"] / n_days,
        "arrival_change": s["sum_arr"] / n, "departure_change": s["sum_dep"] / n,
        "through_change": s["sum_thr"] / n,
        "recovery_pct": 100 * s["n_recovered"] / s["n_late_in"].where(s["n_late_in"] > 0),
        "made_late_pct": 100 * s["n_made_late"] / s["n_ontime_in"].where(s["n_ontime_in"] > 0),
        "minutes_per_day": s["sum_thr"] / n_days}, index=s.index)


def worst_segments(segments: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    """Per station: the inbound and the outbound segment adding the most delay-minutes over the year
    (mean change x pairs). Segments touching a suspected timetable error are skipped."""
    seg = segments[~segments["touches_suspect"].astype(bool) & (segments["n_obs"] >= min_pairs)].copy()
    seg["total"] = seg["mean_change"] * seg["n_obs"]
    out = {}
    for end, other, name in (("to_station", "from_station", "approach"), ("from_station", "to_station", "exit")):
        top = seg.sort_values("total", ascending=False).drop_duplicates(end).set_index(end)
        out[f"{name}_from" if name == "approach" else f"{name}_to"] = top[other]
        out[f"{name}_change"] = top["mean_change"]
    return pd.DataFrame(out).rename_axis("station_name")


def station_through_table(through: pd.DataFrame, n_days: int, p4_stations: pd.DataFrame, ctx_stations: pd.DataFrame,
                          segments: pd.DataFrame, min_pairs: int) -> pd.DataFrame:
    """Every station with through-traffic: rates, class, traffic, names, and its worst approach / exit."""
    r = through_rates(through, n_days)
    meta = p4_stations.set_index("station_name")[["station_full_name", "station_zone"]]
    cls = ctx_stations.set_index("station_name")[["station_class", "trains_day"]]
    out = r.join(meta).join(cls).join(worst_segments(segments, min_pairs))
    return out.rename_axis("station_name").reset_index()


def rank_bottlenecks(st: pd.DataFrame, min_through: int, top: int = C.BOTTLENECK_TOP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bottlenecks, absorbers): junctions and major stations adding / removing the most delay-minutes a day."""
    cand = st[st["station_class"].isin(C.BOTTLENECK_CLASSES) & (st["n_through"] >= min_through)]
    adds = cand[cand["minutes_per_day"] > 0].nlargest(top, "minutes_per_day")
    absorbs = cand[cand["minutes_per_day"] < 0].nsmallest(top, "minutes_per_day")
    return adds.reset_index(drop=True), absorbs.reset_index(drop=True)


def through_by_class(through: pd.DataFrame, st: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Through-station rates pooled per station class (sums, not averages of station rates)."""
    cls = st.set_index("station_name")["station_class"].reindex(through.index)
    s = through.groupby(cls.rename("station_class")).sum()
    r = through_rates(s, n_days).drop(columns=["minutes_per_day"])
    r["stations"] = cls.value_counts()
    return r.reindex([c for c in E.CLASS_ORDER if c in r.index]).reset_index()


# --------------------------------------------------------------------------------------------------
# Model results per group (test set)
# --------------------------------------------------------------------------------------------------
def direct_column(pred: pd.DataFrame) -> str | None:
    cols = [c for c in pred.columns if c.startswith("direct_") and c.endswith("_log")]
    return cols[0] if cols else None


def group_scores(pred: pd.DataFrame, key: pd.Series, min_rows: int, n_boot: int = C.BOOTSTRAP_RESAMPLES,
                 seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Per group with >= min_rows test stops: RMSLE and MAE (persistence, direct single-stage, selected
    hurdle), Stage 1 F1 (hurdle vs 'late if lag1 > 15') and the skill over persistence,
    100 x (1 - RMSLE_selected / RMSLE_persistence), with a 95% day-block bootstrap interval.

    All groups share the same resampled days, as in Phase 3: errors on one day (fog, a blocked line) are
    correlated, so a row bootstrap would be far too narrow."""
    key = pd.Series(key, index=pred.index).astype("string")
    keep_rows = key.notna().to_numpy()
    df, key = pred[keep_rows], key[keep_rows]
    counts = key.value_counts()
    groups = counts.index[counts >= min_rows]
    cols = ["group", "rows", "runs", "late_pct", "rmsle_persistence", "rmsle_direct", "rmsle_selected",
            "rmsle_gain", "skill_pct", "skill_lo", "skill_hi", "mae_persistence", "mae_selected", "f1_persistence", "f1_selected"]
    if not len(groups):
        return pd.DataFrame(columns=cols)
    m = key.isin(groups).to_numpy()
    df, key = df[m], key[m]

    y = df["y_log"].to_numpy("float64")
    delay = df["delay"].to_numpy("float64")
    late = df["y_late"].to_numpy(bool)
    sel, per = df["pred_log"].to_numpy("float64"), df["persistence_log"].to_numpy("float64")
    dcol = direct_column(df)
    dirp = df[dcol].to_numpy("float64") if dcol else np.full(len(df), np.nan)
    hat_sel = df["late_pred"].to_numpy(bool)
    hat_per = df["lag1"].to_numpy("float64") > LATE
    e = pd.DataFrame({
        "g": key.to_numpy(), "run": df["run_id"].to_numpy(), "late": late,
        "se_sel": (y - sel) ** 2, "se_per": (y - per) ** 2, "se_dir": (y - dirp) ** 2,
        "ae_sel": np.abs(np.expm1(sel) - delay), "ae_per": np.abs(np.expm1(per) - delay),
        "tp_sel": hat_sel & late, "fp_sel": hat_sel & ~late, "fn_sel": ~hat_sel & late,
        "tp_per": hat_per & late, "fp_per": hat_per & ~late, "fn_per": ~hat_per & late})
    g = e.groupby("g")
    s = g.sum(numeric_only=True)
    n = g.size()
    f1 = lambda k: 2 * s[f"tp_{k}"] / (2 * s[f"tp_{k}"] + s[f"fp_{k}"] + s[f"fn_{k}"]).where(lambda v: v > 0)
    out = pd.DataFrame({
        "rows": n, "runs": g["run"].nunique(), "late_pct": 100 * s["late"] / n,
        "rmsle_persistence": np.sqrt(s["se_per"] / n), "rmsle_direct": np.sqrt(s["se_dir"] / n),
        "rmsle_selected": np.sqrt(s["se_sel"] / n),
        "mae_persistence": s["ae_per"] / n, "mae_selected": s["ae_sel"] / n,
        "f1_persistence": f1("per"), "f1_selected": f1("sel")})
    # Skill is undefined when persistence is already perfect (a train that never runs late).
    out["skill_pct"] = 100 * (1 - out["rmsle_selected"] / out["rmsle_persistence"].where(out["rmsle_persistence"] > 0))
    out["rmsle_gain"] = out["rmsle_persistence"] - out["rmsle_selected"]

    # Day-block bootstrap of the skill: per (group, day) sums of squared errors, days drawn with replacement.
    gi = out.index.get_indexer(e["g"])
    day = pd.factorize(pd.to_datetime(df["date"]).dt.normalize())[0]
    nk, nd = len(out), int(day.max()) + 1
    mat = lambda w: np.bincount(gi * nd + day, weights=w, minlength=nk * nd).reshape(nk, nd)
    s_sel, s_per = mat(e["se_sel"].to_numpy()), mat(e["se_per"].to_numpy())
    w = np.random.default_rng(seed).multinomial(nd, np.full(nd, 1 / nd), size=n_boot).astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = 100 * (1 - np.sqrt((w @ s_sel.T) / (w @ s_per.T)))   # the row counts cancel in the ratio
    skill[~np.isfinite(skill)] = np.nan
    lo, hi = np.full(nk, np.nan), np.full(nk, np.nan)
    ok = ~np.isnan(skill).all(axis=0)
    if ok.any():
        lo[ok], hi[ok] = np.nanpercentile(skill[:, ok], [2.5, 97.5], axis=0)
    out["skill_lo"], out["skill_hi"] = lo, hi
    return out.rename_axis("group").reset_index()[cols].sort_values("rows", ascending=False, ignore_index=True)


def station_groups(station: pd.Series, st: pd.DataFrame, bottlenecks: pd.DataFrame,
                   absorbers: pd.DataFrame) -> pd.Series:
    """Test stop -> Bottlenecks | Absorbers | Other junctions and major stations | Minor halts."""
    s = station.astype("string")
    cls = s.map(st.set_index("station_name")["station_class"]).astype("string")
    out = pd.Series(np.where(cls.isin(C.BOTTLENECK_CLASSES).fillna(False), STATION_GROUPS[2],
                             np.where(cls.notna(), STATION_GROUPS[3], None)), index=station.index, dtype="string")
    out[s.isin(set(bottlenecks["station_name"])).fillna(False).to_numpy()] = STATION_GROUPS[0]
    out[s.isin(set(absorbers["station_name"])).fillna(False).to_numpy()] = STATION_GROUPS[1]
    return out


def class_scorecard(scores: pd.DataFrame, eda_profile_type: pd.DataFrame, p4_trains: pd.DataFrame) -> pd.DataFrame:
    """Model results per train class next to the class's year profile."""
    prof = E.profile_rates(eda_profile_type)[["n_runs", "mean_delay", "recovery_pct_when_late",
                                              "congestion_slope_per10"]]
    tr = p4_trains.dropna(subset=["type_code"])
    w = tr["n_runs"].astype("float64")
    gained = (tr["mean_gained_en_route"].fillna(0) * w).groupby(tr["type_code"].astype("string")).sum() \
        / w.groupby(tr["type_code"].astype("string")).sum()
    prof = prof.join(gained.rename("mean_gained_en_route"))
    out = scores.merge(prof.rename(columns={"n_runs": "year_runs"}), left_on="group", right_index=True, how="left")
    out.insert(1, "label", out["group"].map(E.TYPE_LABELS).fillna(out["group"]))
    return out


def corridor_scorecard(scores: pd.DataFrame, eda_profile_corridor: pd.DataFrame) -> pd.DataFrame:
    prof = E.profile_rates(eda_profile_corridor)[["n_trains", "mean_delay", "recovery_pct_when_late",
                                                  "congestion_slope_per10"]]
    return scores.merge(prof, left_on="group", right_index=True, how="left")


def train_scorecard(scores: pd.DataFrame, p4_trains: pd.DataFrame, ctx_trains: pd.DataFrame) -> pd.DataFrame:
    meta = p4_trains[["train_no", "train_name", "type_code", "route_km", "origin", "terminus",
                      "mean_terminus_delay"]].merge(ctx_trains[["train_no", "corridor"]], on="train_no", how="left")
    meta = meta.assign(group=meta["train_no"].astype("string")).drop(columns="train_no")
    return scores.merge(meta, on="group", how="left").rename(columns={"group": "train_no"})


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def _paths(full: bool) -> dict:
    return {"src": C.CLEAN_PARQUET if full else C.SAMPLE_PARQUET,
            "pred": C.PREDICTIONS_PARQUET if full else C.PREDICTIONS_SAMPLE_PARQUET,
            "profiles": C.PROFILES_DIR if full else C.PROFILES_SAMPLE_DIR,
            "eda": C.EDA_DIR if full else C.EDA_SAMPLE_DIR,
            "out": C.RESULTS_DIR if full else C.RESULTS_SAMPLE_DIR}


def build(full: bool = False) -> dict:
    t0 = time.time()
    p = _paths(full)
    flag = " --full" if full else ""
    for path, step in ((p["pred"], "train"), (p["profiles"] / "stations.parquet", "profiles"),
                       (p["eda"] / "profile_type.parquet", "eda")):
        if not path.exists():
            raise SystemExit(f"{path} is missing: run `python -m trackcast.{step}{flag}` first")
    k = 1 if full else 0
    min_st, min_seg = C.PROFILE_MIN_OBS["station"][k], C.PROFILE_MIN_OBS["segment"][k]
    min_group, min_train = C.RESULTS_MIN_TEST_ROWS["group"][k], C.RESULTS_MIN_TEST_ROWS["train"][k]

    ctx = E.load_context()
    p4 = {n: pd.read_parquet(p["profiles"] / f"{n}.parquet") for n in ("stations", "segments", "trains", "by_date")}
    suspect = frozenset(p4["stations"].loc[p4["stations"]["suspect_timetable"].astype(bool), "station_name"])
    n_days = int(len(p4["by_date"]))

    groups = P.train_partitions(p["src"], C.PROFILE_TRAIN_PARTITIONS if full else 1)
    parts, rows = [], 0
    for i, trains in enumerate(groups, 1):
        clean = pq.read_table(p["src"], columns=THROUGH_COLUMNS, filters=[("train_no", "in", trains)]).to_pandas()
        rows += len(clean)
        parts.append(through_partial(clean, suspect))
        del clean
        print(f"  group {i}/{len(groups)}: {rows:,} stops so far ({time.time() - t0:.0f} s)")
    through = merge_through(parts)
    st = station_through_table(through, n_days, p4["stations"], ctx["stations"], p4["segments"], min_seg)
    bottlenecks, absorbers = rank_bottlenecks(st, min_st)

    pred = pd.read_parquet(p["pred"])
    corridor = pred["train_no"].map(ctx["trains"].set_index("train_no")["corridor"])
    t = load_eda(p["eda"])
    tables = {
        "through": st,
        "scores_class": class_scorecard(group_scores(pred, pred["type_code"], min_group), t["profile_type"],
                                        p4["trains"]),
        "scores_corridor": corridor_scorecard(group_scores(pred, corridor, min_group), t["profile_corridor"]),
        "scores_station_group": group_scores(pred, station_groups(pred["station_name"], st, bottlenecks, absorbers),
                                             min_group),
        "scores_train": train_scorecard(group_scores(pred, pred["train_no"], min_train), p4["trains"], ctx["trains"]),
    }
    p["out"].mkdir(parents=True, exist_ok=True)
    for name, tab in tables.items():
        tab.to_parquet(p["out"] / f"{name}.parquet", index=False)

    report = summarize(tables, through, pred, full, n_days)
    report["meta"] = {"source": p["src"].name, "predictions": p["pred"].name,
                      "output_dir": str(p["out"].relative_to(C.ROOT)).replace("\\", "/"), "stops": rows,
                      "days": n_days, "test_rows": int(len(pred)), "suspect_stations": len(suspect),
                      "figures": save_figures(report), "elapsed_s": round(time.time() - t0, 1)}
    write_report(report)
    return report


def load_eda(eda_dir) -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(eda_dir / f"{n}.parquet").set_index(E.INDEX[n])
            for n in ("profile_type", "profile_corridor")}


def load(full: bool = False) -> dict[str, pd.DataFrame]:
    """The saved Phase 7 tables (for the app and the notebook)."""
    d = _paths(full)["out"]
    return {n: pd.read_parquet(d / f"{n}.parquet") for n in TABLES}


# --------------------------------------------------------------------------------------------------
# Summary and report
# --------------------------------------------------------------------------------------------------
TRAIN_COLS = ["train_no", "train_name", "type_code", "corridor", "rows", "late_pct", "rmsle_persistence",
              "rmsle_selected", "rmsle_gain", "skill_pct", "mae_persistence", "mae_selected"]
SCORE_COLS = ["rows", "runs", "late_pct", "rmsle_persistence", "rmsle_direct", "rmsle_selected", "skill_pct",
              "skill_lo", "skill_hi", "mae_persistence", "mae_selected", "f1_persistence", "f1_selected"]
ST_COLS = ["station_name", "station_full_name", "station_zone", "station_class", "trains_day", "through_per_day",
           "arrival_change", "departure_change", "through_change", "recovery_pct", "made_late_pct",
           "minutes_per_day", "approach_from", "approach_change", "exit_to", "exit_change"]


DECIMALS = {"late_pct": 1, "skill_pct": 1, "skill_lo": 1, "skill_hi": 1, "recovery_pct": 1, "made_late_pct": 1,
            "recovery_pct_when_late": 1, "mean_delay": 1, "mean_gained_en_route": 1, "congestion_slope_per10": 1,
            "through_per_day": 1, "minutes_per_day": 1, "trains_day": 0, "arrival_change": 1,
            "departure_change": 1, "through_change": 1, "approach_change": 1, "exit_change": 1,
            "mae_persistence": 2, "mae_selected": 2, "route_km": 0, "mean_terminus_delay": 1}


def _rec(df: pd.DataFrame, cols: list[str], nd: int = 3) -> list[dict]:
    """Records rounded per column (RMSLE and F1 to 3 decimals, rates and minutes coarser)."""
    rows = P._records(df, [c for c in cols if c in df.columns], 4)
    return [{c: (round(v, DECIMALS.get(c, nd)) if isinstance(v, float) else v) for c, v in r.items()} for r in rows]


def summarize(t: dict[str, pd.DataFrame], through: pd.DataFrame, pred: pd.DataFrame, full: bool, n_days: int) -> dict:
    k = 1 if full else 0
    min_st = C.PROFILE_MIN_OBS["station"][k]
    st = t["through"]
    bottlenecks, absorbers = rank_bottlenecks(st, min_st)
    overall = group_scores(pred, pd.Series("all", index=pred.index), 1)
    trains = t["scores_train"]
    by_class = through_by_class(through, st, n_days)
    cand = st[st["station_class"].isin(C.BOTTLENECK_CLASSES) & (st["n_through"] >= min_st)]
    return {
        "overall": _rec(overall, SCORE_COLS),
        "class": _rec(t["scores_class"], ["group", "label", *SCORE_COLS, "year_runs", "mean_delay",
                                          "recovery_pct_when_late", "congestion_slope_per10",
                                          "mean_gained_en_route"]),
        "corridor": _rec(t["scores_corridor"].head(C.RESULTS_TOP_CORRIDORS),
                         ["group", *SCORE_COLS, "n_trains", "mean_delay", "recovery_pct_when_late",
                          "congestion_slope_per10"]),
        "station_group": _rec(t["scores_station_group"].set_index("group")
                              .reindex([g for g in STATION_GROUPS if g in set(t["scores_station_group"]["group"])])
                              .reset_index(), ["group", *SCORE_COLS]),
        "trains_scored": int(len(trains)),
        "trains_below_persistence": int((trains["rmsle_gain"] < 0).sum()),
        "trains_ci_below_0": int((trains["skill_hi"] < 0).sum()),
        "trains_skill_quantiles": {f"p{q}": round(float(trains["skill_pct"].quantile(q / 100)), 1)
                                   for q in (10, 25, 50, 75, 90)} if len(trains) else {},
        "trains_best": _rec(trains.nlargest(10, "rmsle_gain"), TRAIN_COLS),
        "trains_worst": _rec(trains.nsmallest(10, "rmsle_gain"), TRAIN_COLS),
        "through_by_class": _rec(by_class, ["station_class", "stations", "n_through", "arrival_change",
                                            "departure_change", "through_change", "recovery_pct", "made_late_pct"]),
        "bottlenecks": _rec(bottlenecks, ST_COLS),
        "absorbers": _rec(absorbers, ST_COLS),
        "candidates": {"stations": int(len(cand)), "adding": int((cand["minutes_per_day"] > 0).sum()),
                       "min_through": min_st,
                       "top_share_pct": round(float(100 * bottlenecks["minutes_per_day"].sum()
                                                    / cand.loc[cand["minutes_per_day"] > 0, "minutes_per_day"].sum()), 1)
                       if len(bottlenecks) else None},
        "thresholds": {"min_group_rows": C.RESULTS_MIN_TEST_ROWS["group"][k],
                       "min_train_rows": C.RESULTS_MIN_TEST_ROWS["train"][k], "top": C.BOTTLENECK_TOP,
                       "bootstrap": C.BOOTSTRAP_RESAMPLES},
    }


def write_report(r: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULTS_JSON.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    C.RESULTS_MD.write_text(render_markdown(r), encoding="utf-8")
    m, o = r["meta"], r["overall"][0]
    print(f"\n=== Results ({m['source']} + {m['predictions']} -> {m['output_dir']}, {len(m['figures'])} figures) ===")
    print(f"  test: skill {o['skill_pct']:.1f}% [{o['skill_lo']:.1f}, {o['skill_hi']:.1f}] over persistence; "
          f"{r['trains_scored']:,} trains scored, {r['trains_below_persistence']} below persistence")
    if r["bottlenecks"]:
        b = r["bottlenecks"][0]
        print(f"  top bottleneck: {b['station_name']} {b['station_full_name']} (+{b['minutes_per_day']:,.0f} min/day)"
              f"  ({m['elapsed_s']} s)")


def _ci(rows: list[dict]) -> list[dict]:
    """Skill with its interval as one display string."""
    return [{**x, "skill": (f"{x['skill_pct']:.1f} [{x['skill_lo']:.1f}, {x['skill_hi']:.1f}]"
                            if x.get("skill_lo") is not None else f"{x['skill_pct']:.1f}")} for x in rows]


H_SCORE = {"rows": "Test stops", "late_pct": "% late", "rmsle_persistence": "RMSLE persistence",
           "rmsle_selected": "RMSLE hurdle", "skill": "Skill % [95% CI]", "mae_persistence": "MAE persistence",
           "mae_selected": "MAE hurdle", "f1_persistence": "F1 persistence", "f1_selected": "F1 hurdle"}
H_ST = {"station_name": "Code", "station_full_name": "Station", "station_zone": "Zone", "trains_day": "Trains/day",
        "through_per_day": "Through/day", "arrival_change": "Arrival", "departure_change": "Departure",
        "through_change": "Through", "recovery_pct": "% late that recover", "made_late_pct": "% on time made late",
        "minutes_per_day": "Min/day", "approach": "Worst approach", "exit": "Worst exit"}


def _seg_text(rows: list[dict]) -> list[dict]:
    f = lambda a, b, ch: "" if a is None or ch is None else f"{a}→{b} ({ch:+.1f})"
    return [{**x, "approach": f(x.get("approach_from"), x["station_name"], x.get("approach_change")),
             "exit": f(x["station_name"], x.get("exit_to"), x.get("exit_change"))} for x in rows]


def render_markdown(r: dict) -> str:
    m, th, o, cand = r["meta"], r["thresholds"], r["overall"][0], r["candidates"]
    md = P._md_table
    L = ["# Results per train class and route; bottleneck junctions (Phase 7)", "",
         f"Model scores on the Phase 3 test set (`{m['predictions']}`, {m['test_rows']:,} stops); through-station "
         f"delay from `{m['source']}` ({m['stops']:,} scheduled stops, {m['days']} days). Tables in "
         f"`{m['output_dir']}/`. All delays in minutes; *late* = more than {LATE} min.", "",
         f"- **Skill** = 100 x (1 - RMSLE hurdle / RMSLE persistence): how much of persistence's error the model "
         f"removes. The interval is a 95% day-block bootstrap ({th['bootstrap']:,} resamples of whole test days, "
         "shared by all groups). Persistence = \"the delay at the previous stop carries over\".",
         f"- Groups need >= {th['min_group_rows']:,} test stops; trains >= {th['min_train_rows']:,}.",
         f"- All test stops: RMSLE {o['rmsle_persistence']:.3f} (persistence) vs {o['rmsle_selected']:.3f} (hurdle), "
         f"skill **{o['skill_pct']:.1f}%** [{o['skill_lo']:.1f}, {o['skill_hi']:.1f}]; F1 {o['f1_persistence']:.3f} "
         f"vs {o['f1_selected']:.3f}; MAE {o['mae_persistence']:.2f} vs {o['mae_selected']:.2f} min.", "",
         "## 1. Results per train class", "", "![results_skill](figures/results_skill.png)", "",
         "Model columns are test-set scores (Dec to Feb); the last columns describe the class over the whole "
         "year (Phase 6): mean delay, % of late arrivals that recover on arrival, congestion sensitivity "
         "(minutes per +10 scheduled trains/h) and mean delay gained from origin to terminus (Phase 4).", ""]
    L += md(_ci(r["class"]), {"label": "Class", **H_SCORE, "mean_delay": "Year mean delay",
                              "recovery_pct_when_late": "Year % recover",
                              "congestion_slope_per10": "Congestion slope",
                              "mean_gained_en_route": "Gained en route"})
    L += ["", "## 2. Results per route", "",
          f"**By corridor** (zones of origin and terminus, both directions; the {len(r['corridor'])} with the most "
          "test stops). Year columns from Phase 6.", ""]
    L += md(_ci(r["corridor"]), {"group": "Corridor", **H_SCORE, "n_trains": "Trains", "mean_delay": "Year mean delay",
                                 "recovery_pct_when_late": "Year % recover",
                                 "congestion_slope_per10": "Congestion slope"})
    H_TR = {"train_no": "Train", "train_name": "Name", "type_code": "Type", "corridor": "Corridor",
            "rows": "Test stops", "late_pct": "% late", "rmsle_persistence": "RMSLE persistence",
            "rmsle_selected": "RMSLE hurdle", "rmsle_gain": "RMSLE gain", "skill_pct": "Skill %",
            "mae_persistence": "MAE persistence", "mae_selected": "MAE hurdle"}
    q = r["trains_skill_quantiles"]
    L += ["", f"**By train** ({r['trains_scored']:,} trains with >= {th['min_train_rows']:,} test stops). Skill "
          f"per train: median {q.get('p50')}%, middle half {q.get('p25')}% to {q.get('p75')}%, 10th to 90th "
          f"percentile {q.get('p10')}% to {q.get('p90')}%. The hurdle has a higher RMSLE than persistence on "
          f"{r['trains_below_persistence']:,} trains, and on {r['trains_ci_below_0']:,} the whole 95% interval is "
          "below zero. Ranked by *RMSLE gain* = RMSLE persistence - RMSLE hurdle (a relative skill explodes for "
          "trains whose persistence error is already near zero). Largest gains:", ""] + md(r["trains_best"], H_TR)
    L += ["", "Largest losses:", ""] + md(r["trains_worst"], H_TR)
    L += ["", "## 3. Schedule recovery through stations", "",
          "For every train passing straight through a station (previous and next scheduled stops both observed "
          "in the run): *arrival* = change from the stop before to the station, *departure* = change from the "
          "station to the stop after (platform dwell + exit), *through* = both. *% late that recover* = trains "
          f"late (> {LATE} min) at the stop before that are less late at the stop after; *% on time made late* = "
          f"trains on time at the stop before that are late at the stop after. Triples touching one of the "
          f"{m['suspect_stations']:,} suspected timetable errors (Phase 4) are left out.", ""]
    L += md(r["through_by_class"], {"station_class": "Class", "stations": "Stations", "n_through": "Trains through",
                                    "arrival_change": "Arrival", "departure_change": "Departure",
                                    "through_change": "Through", "recovery_pct": "% late that recover",
                                    "made_late_pct": "% on time made late"})
    L += ["", "## 4. Bottleneck junctions", "", "![results_bottlenecks](figures/results_bottlenecks.png)", "",
          f"Candidates: junctions and major stations (Phase 6 classes) with >= {cand['min_through']:,} trains "
          f"through ({cand['stations']:,} stations, {cand['adding']:,} of them add delay on balance). Ranked by "
          "*Min/day* = delay-minutes added per day by all trains passing through (mean through change x trains "
          "through per day)."
          + (f" The top {len(r['bottlenecks'])} add {cand['top_share_pct']:.0f}% of the minutes added by all "
             "delay-adding candidates." if cand["top_share_pct"] is not None else "")
          + " *Worst approach / exit* = the inbound / outbound segment adding the most delay-minutes over the "
          "year (mean change per train in brackets).", ""]
    L += md(_seg_text(r["bottlenecks"]), H_ST)
    L += ["", "**Delay absorbers**: junctions and major stations where trains passing through recover the most "
          "delay-minutes per day (negative = recovered).", ""] + md(_seg_text(r["absorbers"]), H_ST)
    L += ["", "**Does the model cope at bottlenecks?** Test-set scores by the arrival station's group.", ""]
    L += md(_ci(r["station_group"]), {"group": "Stations", **H_SCORE})
    L += ["", "Caveats: through change is observed, not causal, and it is measured against the timetable. A strong "
          "absorber usually marks timetable padding rather than a well-run station: compare the scheduled "
          "minutes of its approach with the reverse direction in the Phase 4 segments table. A bad segment "
          "between two stations shows up in the departure change of one and the arrival change of the other, "
          "so the worst approach / exit columns say where to look. Scheduled traffic counts every train as "
          "daily (an upper bound).", "",
          f"Elapsed: {m['elapsed_s']} s."]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------------
# Figures (Phase 6 style and palette: one blue series per panel, so no legend; tables in results.md)
# --------------------------------------------------------------------------------------------------
def _hfig(ncols: int, w: float, h: float):
    fig, axes = E._fig(ncols, w=w, h=h)
    for ax in axes:
        ax.grid(False)
        ax.grid(axis="x", color=E.GRID, linewidth=0.8)
        ax.spines["bottom"].set_visible(False)
    return fig, axes


def _dots(ax, rows: list[dict], label_key: str, overall: float | None):
    rows = sorted(rows, key=lambda x: x["skill_pct"])
    y = np.arange(len(rows))
    lo = [x["skill_pct"] - x["skill_lo"] for x in rows]
    hi = [x["skill_hi"] - x["skill_pct"] for x in rows]
    ax.errorbar([x["skill_pct"] for x in rows], y, xerr=[lo, hi], fmt="none", ecolor=E.ORDINAL[0], elinewidth=2,
                capsize=0)
    ax.scatter([x["skill_pct"] for x in rows], y, s=40, color=E.BLUE, edgecolor=E.SURFACE, linewidth=1.5, zorder=3)
    ax.set_yticks(y, [str(x[label_key]) for x in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if overall is not None:
        ax.axvline(overall, color=E.MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"all test stops {overall:.0f}%", (overall, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -2), textcoords="offset points", fontsize=7.5, color=E.INK2, va="top")


def fig_skill(r: dict) -> Figure:
    corr = r["corridor"][:10]
    n = max(len(r["class"]), len(corr), 1)
    fig, (a, b) = _hfig(2, w=8.6, h=0.32 * n + 1.4)
    overall = r["overall"][0]["skill_pct"] if r["overall"] else None
    _dots(a, r["class"], "label", overall)
    _dots(b, corr, "group", overall)
    E._titles(a, "Skill over persistence by train class", "% lower RMSLE than 'delay carries over'; 95% CI",
              "Skill (%)")
    E._titles(b, "By corridor (most test stops)", "Zones of origin and terminus; 95% CI", "Skill (%)")
    return fig


def fig_bottlenecks(r: dict) -> Figure:
    rows_a = r["bottlenecks"]
    rows_b = r["absorbers"]
    n = max(len(rows_a), len(rows_b), 1)
    fig, (a, b) = _hfig(2, w=10.0, h=0.27 * n + 1.3)
    for ax, rows, sign in ((a, rows_a, 1), (b, rows_b, -1)):
        rows = rows[::-1]                    # largest at the top
        y = np.arange(len(rows))
        v = [sign * x["minutes_per_day"] for x in rows]
        ax.barh(y, v, height=E._bar_height(n, 0.27 * n), color=E.BLUE, edgecolor=E.SURFACE, linewidth=0.6)
        ax.set_yticks(y, [f"{x['station_name']} · {str(x['station_full_name']).title()}" for x in rows])
        ax.set_ylim(-0.7, max(len(rows), 1) - 0.3)
    E._titles(a, "Bottlenecks: delay added", "By trains passing through; junctions and major stations",
              "Delay-minutes added per day")
    E._titles(b, "Absorbers: delay recovered", "By trains passing through; junctions and major stations",
              "Delay-minutes recovered per day")
    return fig


FIGURES = {"results_skill": fig_skill, "results_bottlenecks": fig_bottlenecks}


def save_figures(r: dict) -> list[str]:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fn in FIGURES.items():
        path = C.FIGURES_DIR / f"{name}.png"
        fn(r).savefig(path, dpi=200, facecolor=E.SURFACE)
        out.append(str(path.relative_to(C.ROOT)).replace("\\", "/"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="use the --full predictions, profiles, eda and clean.parquet")
    build(full=ap.parse_args().full)


if __name__ == "__main__":
    main()
