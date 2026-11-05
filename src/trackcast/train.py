"""Phase 3: tune, fit and evaluate the two-stage hurdle model on a temporal holdout.

Usage:  python -m trackcast.train [--full] [--train-rows N] [--test-rows N]

  default   features_sample.parquet -> models/hurdle_sample.joblib, test_predictions_sample.parquet
  --full    features.parquet        -> models/hurdle.joblib,        test_predictions.parquet
            (train/test capped at --train-rows/--test-rows, sampled by whole run, one month in memory at a time)

Protocol
  1. Split by run date: test = runs starting on/after TEST_START_DATE; a short embargo before the cutoff.
  2. Forward-chaining CV inside the training period (each of the last CV_VAL_MONTHS months validates once)
     picks hyper-parameters per family and part, the Stage 1 F1 threshold, and the classifier x regressor pair.
  3. Refit the chosen configurations on the whole training period; score the test period exactly once.
Report: reports/models.md (+ .json).
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trackcast import config as C
from trackcast import evaluation as E
from trackcast import features as F
from trackcast import models as M

PREDICTION_ID_COLUMNS = ["run_id", "train_no", "date", "station_no", "station_name", "type_code", "station_zone",
                         "sched_event_ts", "lag1", "slack_since_lag", "delay", "y_late", "y_log"]


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def run_uniform(run_id) -> np.ndarray:
    """A deterministic U[0, 1) draw per run id, so sampling keeps or drops whole runs."""
    return pd.util.hash_array(np.asarray(run_id, dtype="int64")) / 2.0 ** 64


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    for c in F.CATEGORICAL_FEATURES:  # months carry different category sets; concat falls back to object
        df[c] = df[c].astype("category")
    return df.sort_values(["date", "run_id", "station_no"], ignore_index=True)


def load_split(src, train_max_rows: int, test_max_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    test_start = pd.Timestamp(C.TEST_START_DATE)
    dates = pq.read_table(src, columns=["date"]).column("date").to_pandas()
    tr, te = E.temporal_split(dates, test_start)
    n_tr, n_te, n_all = int(tr.sum()), int(te.sum()), len(dates)
    del dates, tr, te
    if not n_tr or not n_te:
        raise SystemExit(f"empty split around {test_start.date()}: {n_tr} train / {n_te} test rows")
    p_tr, p_te = min(1.0, train_max_rows / n_tr), min(1.0, test_max_rows / n_te)

    # Stream one parquet row group (~1M rows) at a time and filter in Arrow before converting to pandas:
    # reading a whole month (~3M rows x 42 columns) ran out of memory on a 5.8 GB laptop.
    import pyarrow as pa
    pf = pq.ParquetFile(src)
    train_parts, test_parts = [], []
    for i in range(pf.num_row_groups):
        keys = pf.read_row_group(i, columns=["run_id", "date"]).to_pandas()
        u = run_uniform(keys["run_id"])
        tr, te = E.temporal_split(keys["date"], test_start)
        keep_tr, keep_te = tr & (u < p_tr), te & (u < p_te)
        if not (keep_tr.any() or keep_te.any()):
            continue
        table = pf.read_row_group(i)
        for keep, parts in ((keep_tr, train_parts), (keep_te, test_parts)):
            if keep.any():
                parts.append(table.filter(pa.array(keep)).to_pandas())
        del table
    train, test = _concat(train_parts), _concat(test_parts)
    info = {"source": src.name, "source_rows": n_all, "test_start": str(test_start.date()),
            "embargo_days": C.EMBARGO_DAYS, "train_period_rows": n_tr, "test_period_rows": n_te,
            "train_rows": len(train), "test_rows": len(test),
            "train_sample_frac": round(p_tr, 4), "test_sample_frac": round(p_te, 4),
            "train_dates": [str(train["date"].min().date()), str(train["date"].max().date())],
            "test_dates": [str(test["date"].min().date()), str(test["date"].max().date())],
            "train_runs": int(train["run_id"].nunique()), "test_runs": int(test["run_id"].nunique()),
            "train_late_pct": round(100 * float(train["y_late"].mean()), 2),
            "test_late_pct": round(100 * float(test["y_late"].mean()), 2)}
    return train, test, info


# --------------------------------------------------------------------------------------------------
# Cross-validation (training period only)
# --------------------------------------------------------------------------------------------------
def _parts(name: str) -> tuple[str, ...]:
    return ("clf",) if M.FAMILIES[name].task == "clf" else M.REG_PARTS


def _fit(name: str, part: str, params: dict, X: pd.DataFrame, y_late: np.ndarray, y_log: np.ndarray,
         rows: np.ndarray):
    if part != "clf":
        rows = rows[M.part_mask(y_late[rows], part)]
    est = M.make_estimator(name, params, len(rows))
    est.fit(X.iloc[rows], y_late[rows] if part == "clf" else y_log[rows])
    return est


def _predict(est, part: str, X: pd.DataFrame) -> np.ndarray:
    return est.predict_proba(X)[:, 1] if part == "clf" else est.predict(X)


def cross_validate(train: pd.DataFrame) -> dict:
    folds = E.month_folds(train["date"])
    X = M.model_frame(train)
    y_late = train["y_late"].to_numpy(bool)
    y_log = train["y_log"].to_numpy("float64")
    oof: dict[tuple, np.ndarray] = {}
    scores = []
    for tr, va, label in folds:
        print(f"  CV fold {label}: fit on {len(tr):,} rows, validate on {len(va):,}")
        for name, fam in M.FAMILIES.items():
            for part in _parts(name):
                for ci, params in enumerate(fam.grid):
                    t0 = time.time()
                    est = _fit(name, part, params, X, y_late, y_log, tr)
                    pred = _predict(est, part, X.iloc[va])
                    if part == "clf":
                        cm = E.classification_metrics(y_late[va], pred)
                        metric, extra = cm["log_loss"], {"f1_at_0.5": cm["f1"], "roc_auc": cm["roc_auc"]}
                    else:
                        sel = M.part_mask(y_late[va], part)
                        metric, extra = E.rmse(y_log[va][sel], pred[sel]), {}
                    oof.setdefault((name, part, ci), np.full(len(train), np.nan))[va] = pred
                    scores.append({"family": name, "part": part, "config": ci, "params": params, "fold": label,
                                   "metric": metric, **extra, "fit_s": round(time.time() - t0, 1)})
                    print(f"    {name:6s} {part:6s} {params}: {metric:.4f}  ({time.time() - t0:.1f} s)")

    s = pd.DataFrame(scores)
    summary = (s.groupby(["family", "part", "config"], sort=False)
               .agg(metric_mean=("metric", "mean"), metric_std=("metric", "std"), fit_s=("fit_s", "sum"))
               .reset_index())
    summary["params"] = [M.FAMILIES[f].grid[c] for f, c in zip(summary["family"], summary["config"])]
    best = {(r.family, r.part): int(r.config)
            for r in summary.loc[summary.groupby(["family", "part"], sort=False)["metric_mean"].idxmin()]
            .itertuples()}

    # Calibrate each classifier on its pooled out-of-fold predictions (validation months only), then tune the
    # F1 threshold and the hurdle pair on calibrated probabilities. The test set is not involved.
    val = np.concatenate([va for _, va, _ in folds])
    calibrators, p_cal = {}, {}
    for c in M.CLASSIFIERS:
        raw = oof[(c, "clf", best[(c, "clf")])]
        calibrators[c] = M.fit_calibrator(raw[val], y_late[val])
        p_cal[c] = np.full(len(train), np.nan)
        p_cal[c][val] = calibrators[c].predict(raw[val])
    thresholds = {c: E.best_f1_threshold(y_late[val], p_cal[c][val]) for c in M.CLASSIFIERS}

    # Which classifier x regressor pair makes the best hurdle? Judged on the same validation months.
    base = E.baseline_predictions(train)
    combo_rows = []
    for tr, va, label in folds:
        y = y_log[va]
        row = {"fold": label, **{b: E.rmse(y, p[va]) for b, p in base.items()}}
        for r in M.REGRESSORS:
            row[f"direct_{r}"] = E.rmse(y, oof[(r, "direct", best[(r, "direct")])][va])
            for c in M.CLASSIFIERS:
                h = M.combine(p_cal[c][va], oof[(r, "late", best[(r, "late")])][va],
                              oof[(r, "low", best[(r, "low")])][va], thresholds[c])
                row[f"hurdle_{c}+{r}"] = E.rmse(y, h["pred_log"])
        combo_rows.append(row)
    combos = pd.DataFrame(combo_rows).set_index("fold")
    hurdle_cols = [c for c in combos.columns if c.startswith("hurdle_")]
    selected = combos[hurdle_cols].mean().idxmin().removeprefix("hurdle_").split("+")
    return {"folds": [{"month": label, "train_rows": len(tr), "val_rows": len(va)} for tr, va, label in folds],
            "summary": summary.to_dict("records"), "best": best, "thresholds": thresholds,
            "calibrators": calibrators,
            "combos": combos.reset_index().to_dict("records"),
            "combos_mean": combos.mean().to_dict(), "selected": {"classifier": selected[0],
                                                                 "regressor": selected[1]}}


# --------------------------------------------------------------------------------------------------
# Final fit + test evaluation
# --------------------------------------------------------------------------------------------------
def fit_final(train: pd.DataFrame, best: dict) -> dict:
    X = M.model_frame(train)
    y_late, y_log = train["y_late"].to_numpy(bool), train["y_log"].to_numpy("float64")
    rows = np.arange(len(train))
    fitted = {}
    for (name, part), ci in best.items():
        t0 = time.time()
        fitted[(name, part)] = _fit(name, part, M.FAMILIES[name].grid[ci], X, y_late, y_log, rows)
        print(f"  final {name:6s} {part:6s}: {time.time() - t0:.1f} s")
    return fitted


def evaluate_test(test: pd.DataFrame, fitted: dict, cv: dict) -> tuple[dict, M.HurdleModel, pd.DataFrame]:
    X = M.model_frame(test)
    y_late, y_log, delay = test["y_late"].to_numpy(bool), test["y_log"].to_numpy("float64"), test["delay"]
    p_raw = {c: _predict(fitted[(c, "clf")], "clf", X) for c in M.CLASSIFIERS}
    p = {c: cv["calibrators"][c].predict(p_raw[c]) for c in M.CLASSIFIERS}
    m = {(r, part): fitted[(r, part)].predict(X) for r in M.REGRESSORS for part in M.REG_PARTS}
    thr = cv["thresholds"]

    base = E.baseline_predictions(test)
    stage1 = {"persistence (lag1 > 15)": E.label_metrics(y_late, test["lag1"] > C.SIGNIFICANT_DELAY_MIN)}
    for c in M.CLASSIFIERS:
        stage1[f"{c} raw @ 0.5"] = E.classification_metrics(y_late, p_raw[c], 0.5)
        stage1[f"{c} calibrated @ CV-tuned {thr[c]:.3f}"] = E.classification_metrics(y_late, p[c], thr[c])

    stage2 = {}
    for part in ("late", "low"):
        sel = M.part_mask(y_late, part)
        stage2[f"persistence ({part} rows)"] = E.regression_metrics(y_log[sel], base["persistence"][sel], delay[sel])
        for r in M.REGRESSORS:
            stage2[f"{r} ({part} rows)"] = E.regression_metrics(y_log[sel], m[(r, part)][sel], delay[sel])

    preds = dict(base)
    for r in M.REGRESSORS:
        preds[f"direct_{r}"] = m[(r, "direct")]
        for c in M.CLASSIFIERS:
            preds[f"hurdle_{c}+{r}"] = M.combine(p[c], m[(r, "late")], m[(r, "low")], thr[c])["pred_log"].to_numpy()
    c_sel, r_sel = cv["selected"]["classifier"], cv["selected"]["regressor"]
    sel_name = f"hurdle_{c_sel}+{r_sel}"
    preds[f"hurdle_perfect_stage1+{r_sel}"] = np.where(y_late, m[(r_sel, "late")], m[(r_sel, "low")])
    overall = {k: E.regression_metrics(y_log, v, delay) for k, v in preds.items()}
    pers = overall["persistence"]["rmsle"]
    for k in overall:
        overall[k]["skill_vs_persistence_pct"] = 100 * (1 - overall[k]["rmsle"] / pers)

    sel_pred = preds[sel_name]
    boot = {"persistence - selected": E.bootstrap_rmsle_diff(test["date"], y_log, base["persistence"], sel_pred),
            f"direct_{r_sel} - selected": E.bootstrap_rmsle_diff(test["date"], y_log, preds[f"direct_{r_sel}"],
                                                                 sel_pred)}

    lag_bucket = pd.cut(test["lag1"], [-np.inf, 15, 60, 180, np.inf],
                        labels=["lag1 <= 15", "15-60", "60-180", "> 180"])
    slack_bucket = pd.cut(test["slack_since_lag"], [-np.inf, 0, 5, 20, np.inf],
                          labels=["slack 0", "(0,5]", "(5,20]", "> 20"])
    cmp = {"persistence": base["persistence"], f"direct_{r_sel}": preds[f"direct_{r_sel}"], "selected": sel_pred}
    breakdowns = {"RMSLE by month": E.breakdown(test, test["date"].dt.to_period("M").astype(str), cmp),
                  "RMSLE by train type": E.breakdown(test, test["type_code"].astype(str), cmp),
                  "RMSLE by upstream delay (lag1, min)": E.breakdown(test, lag_bucket, cmp),
                  "MAE (minutes) by upstream delay (lag1, min)": E.breakdown(test, lag_bucket, cmp, "mae"),
                  "RMSLE by slack since lag stop (min)": E.breakdown(test, slack_bucket, cmp)}

    best = cv["best"]
    model = M.HurdleModel(
        classifier=fitted[(c_sel, "clf")], calibrator=cv["calibrators"][c_sel],
        reg_late=fitted[(r_sel, "late")], reg_low=fitted[(r_sel, "low")],
        threshold=thr[c_sel],
        meta={"classifier": c_sel, "regressor": r_sel, "threshold": thr[c_sel],
              "params": {"classifier": M.FAMILIES[c_sel].grid[best[(c_sel, "clf")]],
                         "reg_late": M.FAMILIES[r_sel].grid[best[(r_sel, "late")]],
                         "reg_low": M.FAMILIES[r_sel].grid[best[(r_sel, "low")]]},
              "test_start": C.TEST_START_DATE, "significant_delay_min": C.SIGNIFICANT_DELAY_MIN,
              "horizon_stops": C.HORIZON_STOPS, "input_columns": M.MODEL_INPUT_COLUMNS})

    rng = np.random.default_rng(C.RANDOM_STATE)
    sub = test.iloc[np.sort(rng.choice(len(test), min(C.IMPORTANCE_ROWS, len(test)), replace=False))]
    importance = E.group_permutation_importance(lambda d: model.predict(d)["pred_log"].to_numpy(), sub,
                                                E.importance_groups())

    out = test[PREDICTION_ID_COLUMNS].copy()
    h = M.combine(p[c_sel], m[(r_sel, "late")], m[(r_sel, "low")], thr[c_sel])
    for col in h.columns:
        out[col] = h[col].to_numpy()
    out["persistence_log"] = base["persistence"]
    out[f"direct_{r_sel}_log"] = preds[f"direct_{r_sel}"]

    result = {"selected": sel_name, "stage1": stage1, "stage2": stage2, "overall": overall, "bootstrap": boot,
              "breakdowns": breakdowns, "importance": importance}
    return result, model, out


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def run(full: bool = False, train_max_rows: int = C.TRAIN_MAX_ROWS, test_max_rows: int = C.TEST_MAX_ROWS) -> dict:
    t0 = time.time()
    src = C.FEATURES_PARQUET if full else C.FEATURES_SAMPLE_PARQUET
    model_path = C.MODEL_PATH if full else C.MODEL_SAMPLE_PATH
    pred_path = C.PREDICTIONS_PARQUET if full else C.PREDICTIONS_SAMPLE_PARQUET

    train, test, info = load_split(src, train_max_rows, test_max_rows)
    print(f"Train {info['train_rows']:,} rows {info['train_dates']} | test {info['test_rows']:,} rows "
          f"{info['test_dates']}  ({time.time() - t0:.0f} s)")
    cv = cross_validate(train)
    print(f"CV selected: {cv['selected']}  ({time.time() - t0:.0f} s)")
    fitted = fit_final(train, cv["best"])
    test_result, model, preds = evaluate_test(test, fitted, cv)

    model.meta.update({"source": info["source"], "train_rows": info["train_rows"],
                       "train_dates": info["train_dates"]})
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path, compress=3)
    preds.to_parquet(pred_path, index=False)

    cv_out = {k: v for k, v in cv.items() if k != "calibrators"}
    cv_out["best"] = [{"family": f, "part": p, "config": c, "params": M.FAMILIES[f].grid[c]}
                      for (f, p), c in cv["best"].items()]
    report = {"meta": {**info, "model_path": str(model_path.relative_to(C.ROOT)),
                       "predictions_path": str(pred_path.relative_to(C.ROOT)),
                       "model_mb": round(model_path.stat().st_size / 2 ** 20, 1),
                       "horizon_stops": C.HORIZON_STOPS, "elapsed_s": round(time.time() - t0, 1)},
              "cv": cv_out, "test": test_result}
    write_report(report)
    return report


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
DECIMALS = {"late_pct": 1, "fit_s": 1, "mae_min": 2, "skill_vs_persistence_pct": 1, "threshold": 3}


def _f(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return f"{v:,.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)


def write_report(r: dict) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_JSON.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    C.MODELS_MD.write_text(render_markdown(r), encoding="utf-8")
    t, m = r["test"], r["meta"]
    sel, pers = t["overall"][t["selected"]], t["overall"]["persistence"]
    print(f"\n=== Models ({m['source']}: train {m['train_rows']:,} / test {m['test_rows']:,} rows) ===")
    print(f"  selected {t['selected']}: RMSLE {sel['rmsle']:.4f} vs persistence {pers['rmsle']:.4f} "
          f"({sel['skill_vs_persistence_pct']:.1f}% better), R2(log) {sel['r2_log']:.4f}")
    for k, v in t["stage1"].items():
        print(f"  stage 1 {k}: F1 {v['f1']:.4f}")
    print(f"  model: {m['model_path']} ({m['model_mb']} MB)  report: {C.MODELS_MD}  ({m['elapsed_s']} s)")


def _table(rows: list[dict], cols: list[str], headers: list[str] | None = None, nd: int = 4) -> list[str]:
    """Markdown table; text columns left-aligned, numbers right-aligned (decimals from DECIMALS, else nd)."""
    headers = headers or cols
    is_text = [all(isinstance(r.get(c), str) or r.get(c) is None for r in rows) for c in cols]
    L = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if t else "---:" for t in is_text) + "|"]
    L += ["| " + " | ".join(_f(row.get(c), DECIMALS.get(c, nd)) for c in cols) + " |" for row in rows]
    return L


def render_markdown(r: dict) -> str:
    m, cv, t = r["meta"], r["cv"], r["test"]
    sel = cv["selected"]
    sel_name = t["selected"]
    L = ["# Model report (Phase 3)", "",
         f"Source `{m['source']}`, horizon = {m['horizon_stops']} stop(s). "
         f"Temporal holdout: train on runs starting {m['train_dates'][0]} to {m['train_dates'][1]}, test on runs "
         f"starting {m['test_dates'][0]} to {m['test_dates'][1]} (cutoff {m['test_start']}, "
         f"{m['embargo_days']}-day embargo before it).", "",
         f"- Train: {m['train_rows']:,} rows / {m['train_runs']:,} runs ({m['train_late_pct']}% late > "
         f"{C.SIGNIFICANT_DELAY_MIN} min), {100 * m['train_sample_frac']:.1f}% of the training period sampled by run.",
         f"- Test: {m['test_rows']:,} rows / {m['test_runs']:,} runs ({m['test_late_pct']}% late), "
         f"{100 * m['test_sample_frac']:.1f}% of the test period sampled by run.",
         f"- Selected on CV: **Stage 1 = {sel['classifier']}, Stage 2 = {sel['regressor']}**; saved to "
         f"`{m['model_path']}` ({m['model_mb']} MB). Test predictions: `{m['predictions_path']}`.", ""]

    o = t["overall"]
    L += ["## Test: overall forecast of log1p(delay) (all test rows)", "",
          "Hurdle = P(late) x m_late + (1 - P(late)) x m_low. *Direct* = one regressor on all rows. "
          "*Perfect Stage 1* uses the true class to pick the regressor: an upper bound on what a better "
          "classifier could add.", ""]
    rows = [{"model": (f"**{k}** (selected)" if k == sel_name else k), **v} for k, v in o.items()]
    L += _table(rows, ["model", "rmsle", "r2_log", "mae_min", "skill_vs_persistence_pct"],
                ["Model", "RMSLE", "R2 (log1p)", "MAE (min)", "RMSLE skill vs persistence (%)"])
    L += ["", "Day-block bootstrap (95% CI) of the RMSLE difference; positive = the selected model is better:", ""]
    L += [f"- {k}: **{v['diff']:+.4f}** [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] over {v['days']} test days"
          for k, v in t["bootstrap"].items()]

    L += ["", "## Test: Stage 1 (late > 15 min)", "",
          "*Raw* = the classifier's own probabilities. *Calibrated* = after an isotonic map fitted on the pooled "
          "CV validation months; the F1 threshold is tuned there too, never on the test set. The hurdle uses the "
          "calibrated probabilities.", ""]
    rows = [{"model": k, **v} for k, v in t["stage1"].items()]
    L += _table(rows, ["model", "f1", "precision", "recall", "roc_auc", "pr_auc", "brier", "log_loss"],
                ["Model", "F1", "Precision", "Recall", "ROC-AUC", "PR-AUC", "Brier", "Log loss"])

    L += ["", "## Test: Stage 2 (log1p delay, conditional on the true class)", ""]
    rows = [{"model": k, **v} for k, v in t["stage2"].items()]
    L += _table(rows, ["model", "n", "rmsle", "r2_log", "mae_min"], ["Model", "Rows", "RMSLE", "R2 (log1p)", "MAE (min)"])

    L += ["", "## Where the model helps (test set by segment)", ""]
    for name, rows in t["breakdowns"].items():
        cols = [c for c in rows[0] if c not in ("segment", "rows", "late_pct")]
        L += [f"**{name}**", ""] + _table(rows, ["segment", "rows", "late_pct", *cols],
                                          ["Segment", "Rows", "Late %", *cols],
                                          nd=2 if name.startswith("MAE") else 4) + [""]

    L += ["## Feature-group importance (selected model, test subsample)", "",
          "Increase in RMSLE when the group's columns are shuffled together (no refit).", ""]
    L += _table(t["importance"], ["group", "rmsle_increase", "std"], ["Group", "RMSLE increase", "Std"])

    L += ["", "## Cross-validation (training period only)", "",
          "Forward-chaining folds: " + ", ".join(f"{f['month']} (fit {f['train_rows']:,} / val {f['val_rows']:,})"
                                                 for f in cv["folds"]) + ".",
          "Metric: log loss for classifiers; RMSE of log1p(delay) on the part's rows for regressors.", ""]
    best = {(b["family"], b["part"]): b["config"] for b in cv["best"]}
    rows = [{**s, "params": json.dumps(s["params"]),
             "chosen": "yes" if best.get((s["family"], s["part"])) == s["config"] else ""} for s in cv["summary"]]
    L += _table(rows, ["family", "part", "params", "metric_mean", "metric_std", "fit_s", "chosen"],
                ["Family", "Part", "Params", "Mean", "Std", "Fit s (all folds)", "Chosen"])
    L += ["", "CV RMSLE of each hurdle pair, the direct regressors and the baselines (mean over folds):", ""]
    rows = [{"model": k, "rmsle": v} for k, v in sorted(cv["combos_mean"].items(), key=lambda kv: kv[1])]
    L += _table(rows, ["model", "rmsle"], ["Model", "CV RMSLE"])
    L += ["", "CV-tuned Stage 1 thresholds: " + ", ".join(f"{k} {v:.3f}" for k, v in cv["thresholds"].items()) + ".",
          "", f"Elapsed: {m['elapsed_s']} s."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="use features.parquet instead of the dev sample")
    ap.add_argument("--train-rows", type=int, default=C.TRAIN_MAX_ROWS, help="cap on training rows")
    ap.add_argument("--test-rows", type=int, default=C.TEST_MAX_ROWS, help="cap on test rows")
    args = ap.parse_args()
    run(full=args.full, train_max_rows=args.train_rows, test_max_rows=args.test_rows)


if __name__ == "__main__":
    main()
