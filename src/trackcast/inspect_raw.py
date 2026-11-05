"""Print a compact schema summary of each raw table without loading the 1 GB file.

Usage:  python -m trackcast.inspect_raw
"""
import pandas as pd

from trackcast.config import RAW_FILES

PREVIEW_ROWS = 50_000  # enough to infer dtypes; avoids loading combined_delay.csv fully


def main() -> None:
    pd.set_option("display.width", 160, "display.max_columns", 30)
    for name, path in RAW_FILES.items():
        print(f"\n=== {name}: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        df = pd.read_csv(path, nrows=PREVIEW_ROWS)
        print(df.dtypes.to_string())
        print(df.head(3).to_string())


if __name__ == "__main__":
    main()
