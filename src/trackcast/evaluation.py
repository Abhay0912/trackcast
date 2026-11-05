"""Phase 3 evaluation: temporal splits, metrics, baselines, bootstrap intervals and diagnostics."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from trackcast import config as C
from trackcast import features as F


# --------------------------------------------------------------------------------------------------
# Splits (by date, never by random row)
# --------------------------------------------------------------------------------------------------
def temporal_split(dates: pd.Series, test_start, embargo_days: int = C.EMBARGO_DAYS):
    """(train_mask, test_mask). Runs starting in [test_start - embargo, test_start) are in neither: a
    multi-day run that starts just before the cutoff shares days (weather, network state) with test runs."""
    dates = pd.to_datetime(dates)
    test_start = pd.Timestamp(test_start)
    train = (dates < test_start - pd.Timedelta(days=embargo_days)).to_numpy()
    test = (dates >= test_start).to_numpy()
    return train, test


def month_folds(dates: pd.Series, n_folds: int = C.CV_VAL_MONTHS, embargo_days: int = C.EMBARGO_DAYS):
    """Forward-chaining CV over the last `n_folds` calendar months: each validates once, trained on all
    earlier dates (minus the embargo). Returns [(train_idx, val_idx, 'YYYY-MM')]."""
    dates = pd.to_datetime(dates).reset_index(drop=True)
    months = dates.dt.to_period("M")
    folds = []
    for m in sorted(months.unique())[-n_folds:]:
        start = m.to_timestamp()
        tr = np.flatnonzero((dates < start - pd.Timedelta(days=embargo_days)).to_numpy())
        va = np.flatnonzero((months == m).to_numpy())
        if len(tr) and len(va):
            folds.append((tr, va, str(m)))
    if not folds:
        raise ValueError("no usable CV folds: need at least two months of training data")
    return folds


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------
def rmse(y, pred) -> float:
    y, pred = np.asarray(y, "float64"), np.asarray(pred, "float64")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def regression_metrics(y_log, pred_log, delay=None) -> dict:
    """RMSLE (= RMSE on log1p scale), R2 on log1p, and MAE in minutes of expm1(pred)."""
    y, p = np.asarray(y_log, "float64"), np.asarray(pred_log, "float64")
    out = {"n": int(len(y)), "rmsle": rmse(y, p),
           "r2_log": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else np.nan}
    if delay is not None:
        out["mae_min"] = float(np.mean(np.abs(np.expm1(p) - np.asarray(delay, "float64"))))
    return out


def best_f1_threshold(y, p) -> float:
    """Threshold on P(late) maximising F1 (to be chosen on validation predictions only)."""
    prec, rec, thr = precision_recall_curve(np.asarray(y, bool), p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y, bool)
    p = np.clip(np.asarray(p, "float64"), 1e-7, 1 - 1e-7)
    hat = p >= threshold
    return {"n": int(len(y)), "threshold": float(threshold),
            "f1": float(f1_score(y, hat, zero_division=0)),
            "precision": float(precision_score(y, hat, zero_division=0)),
            "recall": float(recall_score(y, hat, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[False, True]))}


def label_metrics(y, hat) -> dict:
    """F1/precision/recall of hard labels (for the persistence rule 'late if lag1 > 15')."""
    y, hat = np.asarray(y, bool), np.asarray(hat, bool)
    return {"n": int(len(y)), "f1": float(f1_score(y, hat, zero_division=0)),
            "precision": float(precision_score(y, hat, zero_division=0)),
            "recall": float(recall_score(y, hat, zero_division=0))}


# --------------------------------------------------------------------------------------------------
# Baselines (no fitting at all)
# --------------------------------------------------------------------------------------------------
def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """log1p predictions of two rules: persistence (delay_k = lag1) and persistence minus the scheduled
    slack between the lag stop and k (the train makes up all its slack, never going below 0)."""
    lag1 = df["lag1"].to_numpy("float64")
    slack = np.nan_to_num(df["slack_since_lag"].to_numpy("float64"), nan=0.0)
    return {"persistence": np.log1p(lag1), "persistence_minus_slack": np.log1p(np.clip(lag1 - slack, 0, None))}


# --------------------------------------------------------------------------------------------------
# Uncertainty and diagnostics
# --------------------------------------------------------------------------------------------------
def bootstrap_rmsle_diff(dates, y_log, pred_a, pred_b, n_resamples: int = C.BOOTSTRAP_RESAMPLES,
                         seed: int = C.RANDOM_STATE) -> dict:
    """RMSLE(a) - RMSLE(b) with a 95% day-block bootstrap interval (positive = b is better).

    Resamples whole days, not rows: errors of trains on the same day (fog, a blocked line) are correlated,
    so a row bootstrap would give intervals that are far too narrow."""
    y = np.asarray(y_log, "float64")
    day = pd.factorize(pd.to_datetime(pd.Series(dates)).dt.normalize())[0]
    n_days = day.max() + 1
    cnt = np.bincount(day, minlength=n_days).astype("float64")
    sa = np.bincount(day, (y - np.asarray(pred_a, "float64")) ** 2, minlength=n_days)
    sb = np.bincount(day, (y - np.asarray(pred_b, "float64")) ** 2, minlength=n_days)
    point = np.sqrt(sa.sum() / cnt.sum()) - np.sqrt(sb.sum() / cnt.sum())
    rng = np.random.default_rng(seed)
    w = rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=n_resamples).astype("float64")  # days drawn
    n = w @ cnt
    diffs = np.sqrt((w @ sa) / n) - np.sqrt((w @ sb) / n)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi), "days": int(n_days),
            "rmsle_a": float(np.sqrt(sa.sum() / cnt.sum())), "rmsle_b": float(np.sqrt(sb.sum() / cnt.sum()))}


def breakdown(df: pd.DataFrame, by: pd.Series, preds: dict[str, np.ndarray], metric: str = "rmsle") -> list[dict]:
    """Per segment: rows, late share and, for each log1p prediction, RMSLE or MAE in minutes."""
    if metric == "rmsle":
        y = df["y_log"].to_numpy("float64")
        err = {name: (y - np.asarray(p, "float64")) ** 2 for name, p in preds.items()}
    elif metric == "mae":
        d = df["delay"].to_numpy("float64")
        err = {name: np.abs(np.expm1(np.asarray(p, "float64")) - d) for name, p in preds.items()}
    else:
        raise ValueError(metric)
    t = pd.DataFrame({"seg": pd.Series(by).array, "late": df["y_late"].to_numpy(bool), **err})
    out = []
    for seg, part in t.groupby("seg", observed=True, sort=True):   # categorical order is kept
        row = {"segment": str(seg), "rows": int(len(part)), "late_pct": round(100 * float(part["late"].mean()), 1)}
        row.update({name: float(np.sqrt(part[name].mean()) if metric == "rmsle" else part[name].mean())
                    for name in preds})
        out.append(row)
    return out


def importance_groups() -> dict[str, list[str]]:
    """Feature groups permuted together (correlated columns are only meaningful as a block)."""
    return {"cascade": F.CASCADE_FEATURES, "slack": F.SLACK_FEATURES, "route": F.ROUTE_FEATURES,
            "time": F.TIME_FEATURES, "season": F.SEASON_FEATURES, "congestion": F.CONGESTION_FEATURES,
            "station (code, zone)": ["station_name", "station_zone"], "train (number, type)": ["train_no", "type_code"]}


def group_permutation_importance(predict_log: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame,
                                 groups: dict[str, list[str]], n_repeats: int = C.IMPORTANCE_REPEATS,
                                 seed: int = C.RANDOM_STATE) -> list[dict]:
    """Increase in RMSLE when one group's columns are shuffled together across rows (model not refitted)."""
    y = df["y_log"].to_numpy("float64")
    base = rmse(y, predict_log(df))
    rng = np.random.default_rng(seed)
    out = []
    for name, cols in groups.items():
        incs = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(df))
            shuffled = df.copy()
            for c in cols:
                shuffled[c] = df[c].iloc[perm].set_axis(df.index)
            incs.append(rmse(y, predict_log(shuffled)) - base)
        out.append({"group": name, "rmsle_increase": float(np.mean(incs)), "std": float(np.std(incs))})
    return sorted(out, key=lambda r: -r["rmsle_increase"])
