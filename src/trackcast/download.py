"""Download the raw Kaggle dataset into data/raw.

Usage:  python -m trackcast.download
Needs a Kaggle API token at %USERPROFILE%\\.kaggle\\kaggle.json.
"""
from trackcast.config import DATA_RAW, KAGGLE_DATASET, RAW_FILES


def main() -> None:
    if all(p.exists() for p in RAW_FILES.values()):
        print("Raw files already present, skipping download.")
        return

    # Imported lazily: the kaggle package raises at import time if no token exists.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} (~1 GB) ...")
    api.dataset_download_files(KAGGLE_DATASET, path=DATA_RAW, unzip=True, quiet=False)

    for name, path in RAW_FILES.items():
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"{name:9s} {path.name:25s} {size_mb:8.1f} MB {'OK' if path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
