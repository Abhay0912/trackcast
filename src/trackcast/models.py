"""Phase 3 building blocks: preprocessing, estimator families and the two-stage hurdle model.

Stage 1  P(late) = P(delay_k > 15 min)              LogisticRegression | RandomForest (+ isotonic calibration)
Stage 2  E[log1p delay | late], E[log1p delay | not late]   Ridge | HistGradientBoosting (one fit per part)
Hurdle   E[log1p delay] = P(late) * m_late + (1 - P(late)) * m_low   (law of total expectation)

Stage 2 regressors learn the residual log1p(delay_k) - log1p(lag1) (PersistenceOffsetRegressor), so the
persistence forecast is built in and the models learn how much delay is gained or recovered.

Everything fitted on delays (target encoders, scalers, imputers) lives inside the sklearn Pipeline, so it is
re-fitted on the training rows of every CV fold and never sees validation or test targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

from trackcast import config as C
from trackcast import features as F

TE_COLUMNS = ["station_name", "train_no"]       # high cardinality (8k stations, 7k trains): target-encoded
ONEHOT_COLUMNS = ["type_code", "station_zone"]  # low cardinality: one-hot
GROUP_COLUMN = "date"                           # run date; only used to group the encoder's cross-fitting folds
MODEL_INPUT_COLUMNS = F.NUMERIC_FEATURES + TE_COLUMNS + ONEHOT_COLUMNS + [GROUP_COLUMN]

# Heavy-tailed, non-negative inputs that linear models see on a log scale (the target is log1p(delay)).
LOG_TERMS = ["lag1", "lag2", "upstream_max_delay", "slack_since_lag", "run_slack_min", "lag_sched_min_back",
             "seg_sched_min", "dwell_prev_min", "seg_km", "seg_n_trains", "distance_from_origin",
             "station_trains_hour", "station_trains_day"]


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The model's input columns from a Phase 2 feature frame, with stable dtypes."""
    X = df[MODEL_INPUT_COLUMNS].copy()
    for c in F.NUMERIC_FEATURES:
        if X[c].dtype.kind == "b":
            X[c] = X[c].astype("float32")
    for c in ["station_name", *ONEHOT_COLUMNS]:
        X[c] = X[c].astype("string").fillna("NA").astype(object)
    X[GROUP_COLUMN] = pd.to_datetime(X[GROUP_COLUMN]).to_numpy().astype("datetime64[D]").astype("int64")
    return X


# --------------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------------
class GroupedTargetEncoder(TransformerMixin, BaseEstimator):
    """sklearn TargetEncoder whose cross-fitting folds are grouped by the last input column (the run date).

    During fit_transform, TargetEncoder encodes each training row from the *other* folds. With its default
    random-row KFold, the row (run r, stop k) would be encoded from later stops of the same run r (their delays
    are almost the target) and from same-day trains at the same station. Training rows would then carry an
    encoding far more informative than any test row gets, and the model would over-trust it. Grouping the
    folds by date removes both channels. transform() uses the encoding fitted on all training rows.
    """

    def __init__(self, n_splits: int = C.TE_INNER_FOLDS):
        self.n_splits = n_splits

    @staticmethod
    def _split(X):
        X = pd.DataFrame(X)
        return X.iloc[:, :-1], X.iloc[:, -1].to_numpy()

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        cats, groups = self._split(X)
        n = min(self.n_splits, len(np.unique(groups)))
        if n < 2:
            raise ValueError("GroupedTargetEncoder needs rows from at least 2 dates")
        cv = list(GroupKFold(n_splits=n).split(cats, y, groups))
        self.encoder_ = TargetEncoder(cv=cv)
        return self.encoder_.fit_transform(cats, y)

    def transform(self, X):
        """Same output as TargetEncoder.transform, via a cached category -> row lookup.

        sklearn rebuilds its category lookup (~8k stations, ~7k trains) on every call: ~30 ms even for one
        row, and the app's recursive route forecast calls the model once per stop (Phase 8). Fitted
        encodings are untouched: known category -> encodings_[j][code], unknown -> target_mean_."""
        cats = self._split(X)[0]
        enc = self.encoder_
        if enc.target_type_ == "multiclass" or cats.shape[1] != len(enc.categories_):
            return enc.transform(cats)
        lookups = self.__dict__.get("_lookups")
        if lookups is None:   # built lazily, so models pickled before this method existed get it too
            lookups = self._lookups = [pd.Index(c) for c in enc.categories_]
        out = np.empty(cats.shape, dtype="float64")
        for j, (index, encoding) in enumerate(zip(lookups, enc.encodings_)):
            code = index.get_indexer(cats.iloc[:, j])
            out[:, j] = np.where(code >= 0, encoding[np.maximum(code, 0)], enc.target_mean_)
        return out

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lookups", None)   # a cache: rebuilt on first use, never pickled
        return state

    def get_feature_names_out(self, input_features=None):
        return self.encoder_.get_feature_names_out()


def linear_terms(X: pd.DataFrame) -> pd.DataFrame:
    """Extra columns for linear models: log scales, a lateness flag and the lag x slack interaction.

    Trees are invariant to monotone transforms and find interactions themselves; Ridge/LogReg are not.
    Phase 2 showed slack acts on the *change* in delay, so the linear models need log(lag1) * log(slack)."""
    X = X.astype("float64")
    for c in LOG_TERMS:
        X[f"log_{c}"] = np.log1p(X[c].clip(lower=0))
    v = X["delay_velocity"]
    X["slog_delay_velocity"] = np.sign(v) * np.log1p(v.abs())
    X["lag1_late"] = (X["lag1"] > C.SIGNIFICANT_DELAY_MIN).astype("float64")
    X["log_lag1_x_log_slack"] = X["log_lag1"] * X["log_slack_since_lag"]
    X["lag1_late_x_log_slack"] = X["lag1_late"] * X["log_slack_since_lag"]
    X["seg_speed_kmph"] = X["seg_speed_kmph"].clip(upper=C.MAX_SEG_SPEED_KMPH)
    return X


def preprocessor(style: str) -> ColumnTransformer:
    """style: 'linear' (log terms + impute + scale), 'impute' (RandomForest) or 'native' (HGB handles NaN)."""
    if style == "linear":
        num = Pipeline([("terms", FunctionTransformer(linear_terms)),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
        te = Pipeline([("te", GroupedTargetEncoder()), ("scale", StandardScaler())])
    elif style == "impute":
        num, te = SimpleImputer(strategy="median", add_indicator=True), GroupedTargetEncoder()
    elif style == "native":
        num, te = "passthrough", GroupedTargetEncoder()
    else:
        raise ValueError(f"unknown preprocessing style {style!r}")
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, F.NUMERIC_FEATURES),
                              ("te", te, TE_COLUMNS + [GROUP_COLUMN]),
                              ("onehot", onehot, ONEHOT_COLUMNS)], remainder="drop")


# --------------------------------------------------------------------------------------------------
# Estimator families
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    task: str                 # "clf" (Stage 1) or "reg" (Stage 2)
    style: str                # preprocessing style
    grid: tuple[dict, ...]    # small hyper-parameter grid searched with forward-chaining CV


FAMILIES = {
    "logreg": Family("clf", "linear", ({"C": 0.01}, {"C": 0.1}, {"C": 1.0})),
    "rf": Family("clf", "impute", ({"min_samples_leaf": 5}, {"min_samples_leaf": 25})),
    "ridge": Family("reg", "linear", ({"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}, {"alpha": 100.0})),
    "hgb": Family("reg", "native", ({"max_leaf_nodes": 31, "min_samples_leaf": 20},
                                    {"max_leaf_nodes": 127, "min_samples_leaf": 100})),
}
CLASSIFIERS = [n for n, f in FAMILIES.items() if f.task == "clf"]
REGRESSORS = [n for n, f in FAMILIES.items() if f.task == "reg"]
REG_PARTS = ("late", "low", "direct")  # fitted on late rows, on not-late rows, on all rows (single-stage)


def part_mask(y_late: np.ndarray, part: str) -> np.ndarray:
    """Rows a Stage 2 part is trained/evaluated on."""
    y_late = np.asarray(y_late, dtype=bool)
    return {"late": y_late, "low": ~y_late, "direct": np.ones_like(y_late)}[part]


class PersistenceOffsetRegressor(RegressorMixin, BaseEstimator):
    """Regress the residual y - log1p(lag1) and add log1p(lag1) back at predict time.

    Delay is persistent stop to stop, so log1p(delay_k) ~ log1p(lag1) + small change. A tree fitted on the
    level puts e.g. lag1 = 300..500 min into one leaf with one constant output, an error of up to ~100 min;
    on the residual, the same leaf only has to learn the (nearly constant) gain or recovery. Linear models
    are unaffected apart from regularisation (the offset is a feature with coefficient fixed at 1).
    Inside the wrapped Pipeline the target encoder then encodes "how much delay trains gain at this station"."""

    def __init__(self, estimator):
        self.estimator = estimator

    @staticmethod
    def offset(X: pd.DataFrame) -> np.ndarray:
        return np.log1p(np.clip(np.nan_to_num(np.asarray(X["lag1"], dtype="float64")), 0, None))

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y, dtype="float64") - self.offset(X))
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + self.offset(X)


def make_estimator(name: str, params: dict, n_rows: int):
    """Stage 1: a Pipeline with predict_proba. Stage 2: a Pipeline wrapped in PersistenceOffsetRegressor."""
    if name == "logreg":
        model = LogisticRegression(max_iter=3000, **params)
    elif name == "rf":
        model = RandomForestClassifier(n_estimators=C.RF_TREES, max_features="sqrt", n_jobs=-1,
                                       max_samples=min(1.0, C.RF_MAX_SAMPLES / max(n_rows, 1)),
                                       random_state=C.RANDOM_STATE, **params)
    elif name == "ridge":
        model = Ridge(**params)
    elif name == "hgb":
        # Early stopping holds out random rows of the *training* data only; test data is never involved.
        model = HistGradientBoostingRegressor(learning_rate=0.1, max_iter=500, early_stopping=True,
                                              validation_fraction=0.1, n_iter_no_change=20,
                                              random_state=C.RANDOM_STATE, **params)
    else:
        raise ValueError(f"unknown family {name!r}")
    pipe = Pipeline([("prep", preprocessor(FAMILIES[name].style)), ("model", model)])
    return pipe if FAMILIES[name].task == "clf" else PersistenceOffsetRegressor(pipe)


# --------------------------------------------------------------------------------------------------
# Hurdle model
# --------------------------------------------------------------------------------------------------
def combine(p_late: np.ndarray, m_late: np.ndarray, m_low: np.ndarray, threshold: float) -> pd.DataFrame:
    """Hurdle prediction. pred_log is the expected log1p(delay), the RMSLE-optimal point forecast when both
    parts are conditional means; pred_delay_min = expm1(pred_log) is closer to a median than a mean.
    The two parts (m_late_log, m_low_log) are returned too, for other point forecasts (forecast.py)."""
    p = np.asarray(p_late, dtype="float64")
    m_late, m_low = np.asarray(m_late, "float64"), np.asarray(m_low, "float64")
    log = np.clip(p * m_late + (1 - p) * m_low, 0, None)
    return pd.DataFrame({"p_late": p, "late_pred": p >= threshold, "pred_log": log,
                         "pred_delay_min": np.expm1(log), "m_late_log": m_late, "m_low_log": m_low})


def fit_calibrator(p_raw: np.ndarray, y_late: np.ndarray) -> IsotonicRegression:
    """Isotonic map raw P(late) -> calibrated P(late), fitted on out-of-fold (validation-month) predictions.

    The hurdle mixes the two parts in log space, so a miscalibrated P(late) costs a lot: a train 250 min late
    with P(late) = 0.97 instead of ~1.0 gets 3% weight on a ~8-min forecast and comes out ~25 min too low.
    Random forests are known to keep probabilities away from 0 and 1, so this matters most for them."""
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        np.asarray(p_raw, "float64"), np.asarray(y_late, "float64"))


def environment_meta() -> dict:
    """Library versions and time of a training run, saved in the model so serving can check its pickle."""
    import datetime as dt
    import sklearn

    return {"trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}}


@dataclass
class HurdleModel:
    classifier: Pipeline
    calibrator: IsotonicRegression
    reg_late: PersistenceOffsetRegressor
    reg_low: PersistenceOffsetRegressor
    threshold: float          # F1-optimal threshold on calibrated P(late), chosen on CV folds (never on test)
    meta: dict = field(default_factory=dict)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = model_frame(df)
        p = self.calibrator.predict(self.classifier.predict_proba(X)[:, 1])
        out = combine(p, self.reg_late.predict(X), self.reg_low.predict(X), self.threshold)
        out.index = df.index
        return out
