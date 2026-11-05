"""Central paths and constants, so no module hard-codes locations or thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

KAGGLE_DATASET = "rxydenxd/indian-railways-delay-dataset"

RAW_FILES = {
    "delay": DATA_RAW / "combined_delay.csv",
    "schedule": DATA_RAW / "combined_schedule.csv",
    "trains": DATA_RAW / "train_details.csv",
    "stations": DATA_RAW / "station_full_names.csv",
}

SIGNIFICANT_DELAY_MIN = 15  # Stage-1 hurdle threshold (minutes)
RANDOM_STATE = 42
