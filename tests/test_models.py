"""Tests for trackcast.models / evaluation / train on synthetic feature frames (never the real data)."""
import numpy as np
import pandas as pd
import pytest

from trackcast import config as C
from trackcast import evaluation as E
from trackcast import features as F
from trackcast import models as M
from trackcast import train as T

BOOL_COLS = {"is_terminus", "is_weekend", "fog_season", "fog_night", "monsoon"}


def synthetic_features(n_runs: int = 500, stops: int = 8, start="2025-06-01", end="2025-12-31", seed: int = 0):
    """A Phase 2-shaped frame where delay_k = lag1 - 0.8 * slack + noise (so slack matters, as in the data)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(start, end)
    n = n_runs * stops
    run = np.repeat(np.arange(n_runs), stops)
    df = pd.DataFrame({"run_id": run.astype("int64"),
                       "train_no": rng.integers(1000, 1040, n_runs)[run].astype("int32"),
                       "date": days[rng.integers(0, len(days), n_runs)][run],
                       "station_no": np.tile(np.arange(2, 2 + stops), n_runs).astype("int16")})
    df["sched_event_ts"] = df["date"]
    df["station_name"] = pd.Categorical(rng.choice([f"S{i}" for i in range(30)], n))
    df["type_code"] = pd.Categorical(rng.choice(["EXP-TRAINS", "PASS-TRAINS", "SF-TRAINS"], n))
    df["station_zone"] = pd.Categorical(rng.choice(["NR", "SCR", "CR"], n))
    for c in F.NUMERIC_FEATURES:
        df[c] = rng.random(n) < 0.3 if c in BOOL_COLS else rng.normal(10, 3, n).astype("float32")
    df["lag1"] = rng.exponential(40, n).astype("float32")
    df["slack_since_lag"] = rng.uniform(0, 30, n).astype("float32")
    df.loc[rng.random(n) < 0.1, "lag2"] = np.nan
    delay = np.clip(df["lag1"] - 0.8 * df["slack_since_lag"] + rng.normal(0, 4, n), 0, None)
    df["delay"] = delay.astype("float32")
    df["y_late"] = df["delay"] > C.SIGNIFICANT_DELAY_MIN
    df["y_log"] = np.log1p(df["delay"]).astype("float32")
    return df[F.OUTPUT_COLUMNS]


# --------------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------------
def test_temporal_split_with_embargo():
    dates = pd.Series(pd.date_range("2025-11-20", "2025-12-10"))
    tr, te = E.temporal_split(dates, "2025-12-01", embargo_days=3)
    assert dates[tr].max() == pd.Timestamp("2025-11-27")       # 11-28 .. 11-30 embargoed
    assert dates[te].min() == pd.Timestamp("2025-12-01")
    assert not (tr & te).any() and (~tr & ~te).sum() == 3


def test_month_folds_forward_chaining():
    dates = pd.Series(pd.date_range("2025-06-01", "2025-11-30"))
    folds = E.month_folds(dates, n_folds=3, embargo_days=2)
    assert [f[2] for f in folds] == ["2025-09", "2025-10", "2025-11"]
    for tr, va, label in folds:
        start = pd.Period(label).to_timestamp()
        assert dates.iloc[tr].max() < start - pd.Timedelta(days=1)     # embargo respected
        assert set(dates.iloc[va].dt.to_period("M").astype(str)) == {label}
    assert len(folds[0][0]) < len(folds[1][0]) < len(folds[2][0])      # expanding window


# --------------------------------------------------------------------------------------------------
# Target encoding: cross-fitting must be grouped by date
# --------------------------------------------------------------------------------------------------
def test_grouped_target_encoder_never_uses_same_date_targets():
    # One category on two dates: delays are 100 on day 1 and 0 on day 2. With date-grouped folds, a day-1
    # row may only be encoded from day-2 rows (and vice versa); a random-row KFold would mix the two days.
    X = pd.DataFrame({"station_name": ["A"] * 20, "date": [1] * 10 + [2] * 10})
    y = np.r_[np.linspace(95, 105, 10), np.linspace(-5, 5, 10)]     # day-1 mean 100, day-2 mean 0
    enc = M.GroupedTargetEncoder(n_splits=2)
    z = enc.fit_transform(X, y).ravel()
    assert np.allclose(z[:10], 0.0) and np.allclose(z[10:], 100.0)
    assert np.allclose(enc.transform(X).ravel(), 50.0)   # at predict time: fitted on all training rows

    with pytest.raises(ValueError):
        M.GroupedTargetEncoder().fit_transform(X.assign(date=1), y)


def test_model_inputs_exclude_targets():
    assert not set(F.TARGETS) & set(M.MODEL_INPUT_COLUMNS)
    X = M.model_frame(synthetic_features(n_runs=20))
    assert list(X.columns) == M.MODEL_INPUT_COLUMNS
    assert X.select_dtypes(include="bool").empty


# --------------------------------------------------------------------------------------------------
# Hurdle maths and metrics
# --------------------------------------------------------------------------------------------------
def test_combine_is_expectation():
    h = M.combine(np.array([1.0, 0.0, 0.25]), np.array([4.0, 4.0, 4.0]), np.array([1.0, 1.0, -2.0]), 0.5)
    assert np.allclose(h["pred_log"], [4.0, 1.0, 0.0])   # third: 0.25*4 + 0.75*(-2) = -0.5, clipped at 0
    assert list(h["late_pred"]) == [True, False, False]
    assert np.allclose(h["pred_delay_min"], np.expm1(h["pred_log"]))
    h2 = M.combine(np.array([0.25]), np.array([4.0]), np.array([2.0]), 0.5)
    assert h2["pred_log"].iloc[0] == pytest.approx(0.25 * 4 + 0.75 * 2)


def test_persistence_offset_regressor():
    from sklearn.dummy import DummyRegressor
    X = pd.DataFrame({"lag1": [0.0, 10.0, 100.0]})
    y = np.log1p(X["lag1"]) + 0.5                                   # every train gains the same log-ratio
    reg = M.PersistenceOffsetRegressor(DummyRegressor()).fit(X, y)
    assert reg.estimator_.constant_.ravel()[0] == pytest.approx(0.5)  # learnt the residual, not the level
    assert np.allclose(reg.predict(X), y)


def test_best_f1_threshold():
    y = np.array([0, 0, 0, 1, 1, 1], bool)
    p = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
    t = E.best_f1_threshold(y, p)
    assert 0.35 < t <= 0.4
    assert E.classification_metrics(y, p, t)["f1"] == 1.0


def test_bootstrap_rmsle_diff():
    rng = np.random.default_rng(0)
    dates = pd.Series(np.repeat(pd.date_range("2025-12-01", periods=30), 50))
    y = rng.normal(3, 1, len(dates))
    good, bad = y + rng.normal(0, 0.1, len(y)), y + rng.normal(0, 1.0, len(y))
    same = E.bootstrap_rmsle_diff(dates, y, good, good, n_resamples=200)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    b = E.bootstrap_rmsle_diff(dates, y, bad, good, n_resamples=200)
    assert b["days"] == 30 and 0 < b["ci_low"] <= b["diff"] <= b["ci_high"]


def test_run_uniform_keeps_whole_runs():
    run_id = np.repeat(np.arange(100), 5)
    u = T.run_uniform(run_id)
    assert ((u >= 0) & (u < 1)).all()
    assert (pd.Series(u).groupby(run_id).nunique() == 1).all()
    assert np.array_equal(u, T.run_uniform(run_id))   # deterministic


# --------------------------------------------------------------------------------------------------
# End to end on synthetic data
# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained():
    mp = pytest.MonkeyPatch()
    mp.setattr(C, "RF_TREES", 20)
    mp.setattr(C, "IMPORTANCE_ROWS", 300)
    mp.setattr(C, "IMPORTANCE_REPEATS", 1)
    mp.setattr(C, "BOOTSTRAP_RESAMPLES", 50)
    small = {name: M.Family(f.task, f.style, f.grid[:2]) for name, f in M.FAMILIES.items()}
    mp.setattr(M, "FAMILIES", small)
    df = synthetic_features()
    tr, te = E.temporal_split(df["date"], "2025-12-01")
    train, test = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
    cv = T.cross_validate(train)
    fitted = T.fit_final(train, cv["best"])
    result, model, preds = T.evaluate_test(test, fitted, cv)
    yield train, test, cv, result, model, preds
    mp.undo()


def test_pipeline_end_to_end(trained):
    train, test, cv, result, model, preds = trained
    assert [f["month"] for f in cv["folds"]] == ["2025-09", "2025-10", "2025-11"]
    assert set(cv["selected"]) == {"classifier", "regressor"}
    assert len(preds) == len(test) and preds["pred_log"].ge(0).all()
    o = result["overall"]
    # delay = lag1 - 0.8 * slack + noise: a fitted model must beat pure persistence
    assert o[result["selected"]]["rmsle"] < o["persistence"]["rmsle"]
    assert {r["group"] for r in result["importance"]} == set(E.importance_groups())
    assert result["importance"][0]["group"] in {"cascade", "slack"}


def test_predictions_do_not_depend_on_targets(trained):
    """The fitted model must ignore the target columns of the rows it scores."""
    _, test, _, _, model, _ = trained
    p0 = model.predict(test)
    scrambled = test.assign(delay=0.0, y_late=False, y_log=0.0)
    pd.testing.assert_frame_equal(p0, model.predict(scrambled))


def test_report_renders(trained):
    train, test, cv, result, model, preds = trained
    cv_out = dict(cv, best=[{"family": f, "part": p, "config": c, "params": M.FAMILIES[f].grid[c]}
                            for (f, p), c in cv["best"].items()])
    meta = {"source": "synthetic", "train_dates": ["a", "b"], "test_dates": ["c", "d"], "test_start": "x",
            "embargo_days": 3, "train_rows": len(train), "train_runs": 1, "train_late_pct": 50.0,
            "train_sample_frac": 1.0, "test_rows": len(test), "test_runs": 1, "test_late_pct": 50.0,
            "test_sample_frac": 1.0, "model_path": "m", "model_mb": 0.1, "predictions_path": "p",
            "horizon_stops": 1, "elapsed_s": 0.0}
    md = T.render_markdown({"meta": meta, "cv": cv_out, "test": result})
    assert "Model report (Phase 3)" in md and "(selected)" in md
