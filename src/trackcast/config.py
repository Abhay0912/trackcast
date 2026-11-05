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
