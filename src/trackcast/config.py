"""Central paths and constants, so no module hard-codes locations or thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Phase 1 outputs
INTERIM_BY_MONTH = DATA_INTERIM / "delay_by_month"
CLEAN_PARQUET = DATA_PROCESSED / "clean.parquet"
SAMPLE_PARQUET = DATA_PROCESSED / "sample.parquet"
DATA_QUALITY_MD = REPORTS_DIR / "data_quality.md"
DATA_QUALITY_JSON = REPORTS_DIR / "data_quality.json"

KAGGLE_DATASET = "rxydenxd/indian-railways-delay-dataset"

RAW_FILES = {
    "delay": DATA_RAW / "combined_delay.csv",
    "schedule": DATA_RAW / "combined_schedule.csv",
    "trains": DATA_RAW / "train_details.csv",
    "stations": DATA_RAW / "station_full_names.csv",
}

SIGNIFICANT_DELAY_MIN = 15  # Stage-1 hurdle threshold (minutes)
RANDOM_STATE = 42

# Phase 1 cleaning / sampling
DELAY_CHUNKSIZE = 2_000_000      # rows per CSV chunk in pass 1
MAX_PLAUSIBLE_DELAY_MIN = 1440   # delays above 24 h are treated as data errors (set to NaN)
MIN_OBSERVED_STOPS_PER_RUN = 3   # need k-1 and k-2 for lag + velocity
SAMPLE_TARGET_ROWS = 100_000     # dev sample size (whole runs only)
SAMPLE_OVERDRAW = 2.0            # draw this many x target as candidates, trim to target at the end

# Phase 2 features
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"            # from clean.parquet (--full)
FEATURES_SAMPLE_PARQUET = DATA_PROCESSED / "features_sample.parquet"  # from sample.parquet (dev)
FEATURES_MD = REPORTS_DIR / "features.md"
FEATURES_JSON = REPORTS_DIR / "features.json"
HORIZON_STOPS = 1          # predict stop k from delays observed at stops <= k - HORIZON_STOPS
REPORT_SUBSAMPLE_ROWS = 500_000  # rows kept for correlations / baselines in the feature report

# Season flags (the dataset has no weather columns and no station coordinates for an Open-Meteo join)
FOG_MONTHS = (12, 1, 2)
FOG_ZONES = ("NR", "NCR", "NER", "NWR", "ECR", "ER")  # Indo-Gangetic plain + Rajasthan
FOG_NIGHT_HOURS = (21, 10)  # [21:00, 10:00): when dense fog and visibility rules bite
MONSOON_MONTHS = (6, 7, 8, 9)

# Phase 3 models
TEST_START_DATE = "2025-12-01"   # temporal holdout: runs starting on/after this date are the test set
EMBARGO_DAYS = 3                 # drop training runs starting < this many days before a validation/test cutoff
CV_VAL_MONTHS = 3                # forward-chaining CV: each of the last N training months validates once
TE_INNER_FOLDS = 5               # target-encoder cross-fitting folds (grouped by run date)
TRAIN_MAX_ROWS = 1_000_000       # cap on training rows (sampled by whole run) so --full fits in laptop RAM
TEST_MAX_ROWS = 1_000_000        # cap on test rows (sampled by whole run)
RF_TREES = 200
RF_MAX_SAMPLES = 100_000         # rows bootstrapped per tree: caps RF size (~0.6 GB at leaf 5) on 5.8 GB RAM
MAX_SEG_SPEED_KMPH = 200         # linear models only: clip timetable-rounding artefacts (1-min segments)
IMPORTANCE_ROWS = 20_000         # test rows used for grouped permutation importance
IMPORTANCE_REPEATS = 3
BOOTSTRAP_RESAMPLES = 1000       # day-block bootstrap for confidence intervals on test metrics
MODEL_PATH = MODELS_DIR / "hurdle.joblib"                 # --full
MODEL_SAMPLE_PATH = MODELS_DIR / "hurdle_sample.joblib"   # dev sample
PREDICTIONS_PARQUET = DATA_PROCESSED / "test_predictions.parquet"
PREDICTIONS_SAMPLE_PARQUET = DATA_PROCESSED / "test_predictions_sample.parquet"
MODELS_MD = REPORTS_DIR / "models.md"
MODELS_JSON = REPORTS_DIR / "models.json"

# Phase 4 profiles
PROFILES_DIR = DATA_PROCESSED / "profiles"                 # --full (from clean.parquet)
PROFILES_SAMPLE_DIR = DATA_PROCESSED / "profiles_sample"   # dev (from sample.parquet)
PROFILES_MD = REPORTS_DIR / "profiles.md"
PROFILES_JSON = REPORTS_DIR / "profiles.json"
PROFILE_TRAIN_PARTITIONS = 12    # --full: process trains in this many groups (each holds all of its runs)
CASCADE_MAX_STOPS = 20           # cascade curve: delay h = 1..N observed stops after a stop
CASCADE_BUCKETS = (15, 60, 180)  # initial-delay buckets: <=15 (on time), (15,60], (60,180], >180
HIST_MAX_DELAY_MIN = 1440        # station delay histograms (1-min bins) for exact medians / p90
SAWTOOTH_MIN_CHANGE = 15         # station flagged when arrival change and next-segment change are both >= this,
                                 # with opposite signs (recorded delay there is off, e.g. a wrong timetable time)
# Phase 5 app: multi-stop (recursive) route forecasts
ROUTE_FORECAST_STRATEGY = "mean"     # minutes fed forward: median | mean | gate. "mean" wins from 2 stops
                                     # ahead on real test runs (reports/route_forecast.md); "median" decays to 0
ROUTE_EVAL_RUNS = 1500               # test-period runs simulated by `python -m trackcast.forecast`
ROUTE_EVAL_STEPS = 20                # stops forecast ahead per simulated run
ROUTE_FORECAST_MD = REPORTS_DIR / "route_forecast.md"
ROUTE_FORECAST_JSON = REPORTS_DIR / "route_forecast.json"

# Minimum support before an entity appears in the report's rankings: (dev sample, full data)
PROFILE_MIN_OBS = {"station": (30, 2000), "segment": (15, 500), "train": (3, 60)}
