"""Phase 1: raw delay telemetry -> joined, cleaned parquet + whole-run dev sample + data-quality report.

Usage:  python -m trackcast.data [--nrows N] [--chunksize N] [--keep-interim]

Two passes, so the 1 GB CSV is never held in memory:
  pass 1  stream combined_delay.csv in chunks, count raw quality issues, and split the rows into one
          interim parquet file per month. A run (train_no, date) is keyed on date, so it never spans files.
  pass 2  load one month at a time, join schedule / trains / stations, apply the cleaning rules, append to
          clean.parquet, and draw candidate runs for the sample.

Outputs: config.CLEAN_PARQUET, config.SAMPLE_PARQUET, config.DATA_QUALITY_MD / DATA_QUALITY_JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from trackcast import config as C

DELAY_KEYS = ["train_no", "date", "station_no"]
RUN_KEYS = ["train_no", "date"]
RUN_ID_EPOCH = pd.Timestamp("2000-01-01")
RUN_ID_MULT = 100_000  # run_id = train_no * RUN_ID_MULT + days since RUN_ID_EPOCH

START_DATE, CALENDAR_DATE, INCONCLUSIVE = "START_DATE", "CALENDAR_DATE", "INCONCLUSIVE"
MIN_PAIRS = 20  # minimum boundary pairs for the continuity test to be decisive

INTERIM_SCHEMA = pa.schema([
    ("train_no", pa.int32()),
    ("date", pa.timestamp("s")),
    ("station_no", pa.int16()),
    ("station_name", pa.string()),
    ("delay", pa.float32()),
])

_DICT = pa.dictionary(pa.int32(), pa.string())
CLEAN_SCHEMA = pa.schema([
    ("run_id", pa.int64()),
    ("train_no", pa.int32()),
    ("date", pa.timestamp("s")),
    ("station_no", pa.int16()),
    ("station_name", _DICT),
    ("delay_raw", pa.float32()),         # as given (negatives kept); > cap set to NaN
    ("delay", pa.float32()),             # delay_raw clipped at 0 -> modelling target
    ("delay_missing", pa.bool_()),
    ("arrival_day", pa.int8()),
    ("departure_day", pa.int8()),
    ("sched_arr_offset_min", pa.float32()),  # minutes from midnight of the run start date
    ("sched_dep_offset_min", pa.float32()),
    ("sched_arr_ts", pa.timestamp("s")),
    ("distance_from_origin", pa.float32()),
    ("is_origin", pa.bool_()),
    ("is_terminus", pa.bool_()),
    ("train_name", _DICT),
    ("type_code", _DICT),
    ("station_full_name", _DICT),
    ("station_zone", _DICT),
])
CLEAN_COLUMNS = CLEAN_SCHEMA.names


# --------------------------------------------------------------------------------------------------
# Quality accumulators
# --------------------------------------------------------------------------------------------------
@dataclass
class PairStats:
    """Streaming stats for (x, y) delay pairs: correlation and median |x - y|."""

    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0
    absdiff_hist: Counter = field(default_factory=Counter)

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        self.n += len(x)
        self.sx += x.sum()
        self.sy += y.sum()
        self.sxx += (x * x).sum()
        self.syy += (y * y).sum()
        self.sxy += (x * y).sum()
        vals, cnts = np.unique(np.round(np.abs(x - y)).astype(int), return_counts=True)
        self.absdiff_hist.update(dict(zip(vals.tolist(), cnts.tolist())))

    def corr(self) -> float:
        if self.n < 2:
            return float("nan")
        cov = self.sxy - self.sx * self.sy / self.n
        vx = self.sxx - self.sx**2 / self.n
        vy = self.syy - self.sy**2 / self.n
        return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else float("nan")

    def median_absdiff(self) -> float:
        q = _hist_quantile(self.absdiff_hist, 0.5)
        return float("nan") if q is None else float(q)


@dataclass
class QualityStats:
    meta: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    delay_hist: Counter = field(default_factory=Counter)  # raw delay (rounded min) -> rows
    missing_by_position: Counter = field(default_factory=Counter)  # (position, is_missing) -> rows
    missing_by_month: Counter = field(default_factory=Counter)  # (YYYY-MM, is_missing) -> rows
    coverage_hist: Counter = field(default_factory=Counter)  # coverage decile -> runs
    trains: set = field(default_factory=set)
    stations: set = field(default_factory=set)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    pairs_same: PairStats = field(default_factory=PairStats)
    pairs_next: PairStats = field(default_factory=PairStats)
    weekday_parts: list = field(default_factory=list)
    date_check: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        c, h = self.counts, self.delay_hist
        raw = c["raw_rows"]
        valid = raw - c["dropped_bad_key"] - c["dropped_bad_date"]
        dedup = valid - c["dropped_duplicates"]
        matched = dedup - c["dropped_no_schedule"]
        final = matched - c["dropped_short_run_rows"]
        neg = {k: v for k, v in h.items() if k < 0}
        pos = {
            p: {
                "rows": self.missing_by_position[(p, False)] + self.missing_by_position[(p, True)],
                "missing": self.missing_by_position[(p, True)],
            }
            for p in ("origin", "intermediate", "terminus")
        }
        for v in pos.values():
            v["missing_pct"] = _pct(v["missing"], v["rows"])
        months = sorted({m for m, _ in self.missing_by_month})
        by_month = {
            m: _pct(self.missing_by_month[(m, True)],
                    self.missing_by_month[(m, True)] + self.missing_by_month[(m, False)])
            for m in months
        }
        return {
            "meta": self.meta,
            "waterfall": [
                ["raw rows", raw],
                ["valid keys and date", valid],
                ["after dedupe", dedup],
                ["matched to schedule", matched],
                [f"runs with >= {C.MIN_OBSERVED_STOPS_PER_RUN} observed delays (final)", final],
            ],
            "raw_delay": {
                "rows": raw,
                "missing": c["raw_missing"],
                "missing_pct": _pct(c["raw_missing"], raw),
                "negative": c["raw_negative"],
                "negative_pct": _pct(c["raw_negative"], raw),
                "negative_min": min(neg) if neg else None,
                "negative_median": _hist_quantile(neg, 0.5),
                "above_cap": c["raw_above_cap"],
                "quantiles": {f"p{int(q * 100)}": _hist_quantile(h, q)
                              for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
                "max": max(h) if h else None,
            },
            "clean": {
                "rows": c["clean_rows"],
                "runs": c["clean_runs"],
                "trains": len(self.trains),
                "stations": len(self.stations),
                "date_min": str(self.date_min.date()) if self.date_min is not None else None,
                "date_max": str(self.date_max.date()) if self.date_max is not None else None,
                "missing": c["clean_missing"],
                "missing_pct": _pct(c["clean_missing"], c["clean_rows"]),
                "runs_with_any_missing": c["runs_with_any_missing"],
                "runs_with_any_missing_pct": _pct(c["runs_with_any_missing"], c["clean_runs"]),
                "negative_clipped_to_0": c["negative_clipped"],
                "above_cap_set_nan": c["delay_above_cap_set_nan"],
                "delay_gt_threshold_pct": _pct(c["clean_delay_gt_threshold"],
                                               c["clean_rows"] - c["clean_missing"]),
                "missing_by_position": pos,
                "missing_pct_by_month": by_month,
            },
            "joins": {
                "dropped_bad_key": c["dropped_bad_key"],
                "dropped_bad_date": c["dropped_bad_date"],
                "dropped_duplicates": c["dropped_duplicates"],
                "dropped_no_schedule": c["dropped_no_schedule"],
                "station_no_renumbered_to_schedule": c["station_no_renumbered"],
                "stop_order_inversions_vs_schedule": c["stop_order_inversions"],
                "rows_train_not_in_details": c["rows_train_not_in_details"],
                "rows_station_not_in_names": c["rows_station_not_in_names"],
                "dropped_short_runs": c["dropped_short_runs"],
                "dropped_short_run_rows": c["dropped_short_run_rows"],
            },
            "lookups": {
                "schedule_rows": c["schedule_rows"],
                "schedule_trains": c["schedule_trains"],
                "schedule_duplicates": c["schedule_duplicates"],
                "schedule_repeated_station": c["schedule_repeated_station"],
                "schedule_unparsed_times": c["schedule_unparsed_times"],
                "schedule_offset_violation_trains": c["schedule_offset_violation_trains"],
                "trains_duplicate_keys": c["trains_duplicate_keys"],
                "stations_duplicate_keys": c["stations_duplicate_keys"],
            },
            "run_coverage_deciles": {f"{k:.1f}": v for k, v in sorted(self.coverage_hist.items())},
            "date_check": self.date_check,
            "sample": {"rows": c["sample_rows"], "runs": c["sample_runs"]},
        }


def _pct(num: float, den: float) -> float | None:
    return round(100.0 * num / den, 3) if den else None


def _hist_quantile(hist: dict, q: float):
    if not hist:
        return None
    keys = sorted(hist)
    cum = np.cumsum([hist[k] for k in keys])
    idx = int(np.searchsorted(cum, q * cum[-1], side="left"))
    return keys[min(idx, len(keys) - 1)]


# --------------------------------------------------------------------------------------------------
# Lookup tables (small, loaded whole)
# --------------------------------------------------------------------------------------------------
def _hhmm_to_min(s: pd.Series) -> pd.Series:
    parts = s.astype("string").str.extract(r"^\s*(\d{1,2}):(\d{2})")
    return (pd.to_numeric(parts[0]) * 60 + pd.to_numeric(parts[1])).astype("float32")


def prepare_schedule(raw: pd.DataFrame, stats: QualityStats) -> pd.DataFrame:
    s = raw.dropna(subset=["train_no", "station_no"]).copy()
    s["station_name"] = s["station_name"].astype("string").str.strip()
    n = len(s)
    s = s.drop_duplicates(["train_no", "station_no"], keep="first")
    stats.counts["schedule_duplicates"] += n - len(s)
    n = len(s)  # a station visited twice by one train would make the (train_no, station_name) join ambiguous
    s = s.drop_duplicates(["train_no", "station_name"], keep="first")
    stats.counts["schedule_repeated_station"] += n - len(s)

    s["train_no"] = s["train_no"].astype("int32")
    s["station_no"] = s["station_no"].astype("int16")
    s["distance_from_origin"] = s["distance_from_origin"].astype("float32")
    s["arrival_day"] = pd.to_numeric(s["arrival_day"]).astype("Int8")
    s["departure_day"] = pd.to_numeric(s["departure_day"]).astype("Int8")

    arr_min, dep_min = _hhmm_to_min(s["arrival_time"]), _hhmm_to_min(s["departure_time"])
    unparsed = (s["arrival_time"].notna() & arr_min.isna()) | (s["departure_time"].notna() & dep_min.isna())
    stats.counts["schedule_unparsed_times"] += int(unparsed.sum())
    s["sched_arr_offset_min"] = ((s["arrival_day"].astype("float32") - 1) * 1440 + arr_min).astype("float32")
    s["sched_dep_offset_min"] = ((s["departure_day"].astype("float32") - 1) * 1440 + dep_min).astype("float32")

    s = s.sort_values(["train_no", "station_no"], ignore_index=True)
    g = s.groupby("train_no")["station_no"]
    s["is_origin"] = s["station_no"] == g.transform("min")
    s["is_terminus"] = s["station_no"] == g.transform("max")
    s["n_sched_stops"] = g.transform("size").astype("int16")
    s["stop_day"] = s["arrival_day"].fillna(s["departure_day"])

    prev_dep = s.groupby("train_no")["sched_dep_offset_min"].shift()
    viol = (s["sched_arr_offset_min"] < prev_dep) | (s["sched_dep_offset_min"] < s["sched_arr_offset_min"])
    stats.counts["schedule_offset_violation_trains"] = int(s.loc[viol, "train_no"].nunique())
    stats.counts["schedule_rows"] = len(s)
    stats.counts["schedule_trains"] = int(s["train_no"].nunique())
    return s[["train_no", "station_no", "station_name", "distance_from_origin", "arrival_day",
              "departure_day", "sched_arr_offset_min", "sched_dep_offset_min", "is_origin",
              "is_terminus", "n_sched_stops", "stop_day"]]


def prepare_lookup(raw: pd.DataFrame, key: str, cols: list[str], stats: QualityStats, name: str) -> pd.DataFrame:
    """Dedupe a lookup table on its key so a left join can never multiply rows."""
    df = raw[[key, *cols]].dropna(subset=[key]).copy()
    if key == "train_no":
        df[key] = df[key].astype("int32")
    else:
        df[key] = df[key].astype("string").str.strip()
    n = len(df)
    df = df.drop_duplicates(key, keep="first")
    stats.counts[f"{name}_duplicate_keys"] += n - len(df)
    for col in cols:
        df[col] = df[col].astype("string")
    return df


def load_schedule(stats: QualityStats) -> pd.DataFrame:
    return prepare_schedule(pd.read_csv(C.RAW_FILES["schedule"]), stats)


def load_trains(stats: QualityStats) -> pd.DataFrame:
    return prepare_lookup(pd.read_csv(C.RAW_FILES["trains"]), "train_no", ["train_name", "type_code"], stats, "trains")


def load_stations(stats: QualityStats) -> pd.DataFrame:
    raw = pd.read_csv(C.RAW_FILES["stations"], usecols=["station_name", "station_full_name", "station_zone"])
    return prepare_lookup(raw, "station_name", ["station_full_name", "station_zone"], stats, "stations")


# --------------------------------------------------------------------------------------------------
# Pass 1: stream the CSV, count raw issues, split by month
# --------------------------------------------------------------------------------------------------
def iter_delay_chunks(path: Path, chunksize: int, nrows: int | None = None):
    dtypes = {"date": "string", "station_name": "string", "station_no": "Int16",
              "train_no": "Int32", "delay": "float32"}
    yield from pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, chunksize=chunksize, nrows=nrows)


def raw_chunk_stats(chunk: pd.DataFrame, stats: QualityStats) -> pd.DataFrame:
    """Count raw issues in one chunk; return typed rows with valid keys and date."""
    c = stats.counts
    d = chunk["delay"]
    c["raw_rows"] += len(chunk)
    c["raw_missing"] += int(d.isna().sum())
    c["raw_negative"] += int((d < 0).sum())
    c["raw_above_cap"] += int((d > C.MAX_PLAUSIBLE_DELAY_MIN).sum())
    vals, cnts = np.unique(np.round(d.dropna().to_numpy()).astype(int), return_counts=True)
    stats.delay_hist.update(dict(zip(vals.tolist(), cnts.tolist())))

    bad_key = (chunk["train_no"].isna() | chunk["station_no"].isna()).to_numpy()
    dates = pd.to_datetime(chunk["date"], format="%Y-%m-%d", errors="coerce")
    bad_date = dates.isna().to_numpy() & ~bad_key
    c["dropped_bad_key"] += int(bad_key.sum())
    c["dropped_bad_date"] += int(bad_date.sum())

    keep = ~(bad_key | bad_date)
    return pd.DataFrame({
        "train_no": chunk["train_no"][keep].astype("int32"),
        "date": dates[keep].astype("datetime64[s]"),
        "station_no": chunk["station_no"][keep].astype("int16"),
        "station_name": chunk["station_name"][keep],
        "delay": d[keep],
    })


def split_by_month(path: Path, out_dir: Path, stats: QualityStats, chunksize: int,
                   nrows: int | None = None) -> dict[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    rows_by_month: Counter = Counter()
    try:
        for i, chunk in enumerate(iter_delay_chunks(path, chunksize, nrows), 1):
            chunk = raw_chunk_stats(chunk, stats)
            month = chunk["date"].dt.year * 100 + chunk["date"].dt.month
            for m, part in chunk.groupby(month, sort=False):
                m = int(m)
                if m not in writers:
                    writers[m] = pq.ParquetWriter(out_dir / f"{m}.parquet", INTERIM_SCHEMA)
                table = pa.Table.from_pandas(part[INTERIM_SCHEMA.names], preserve_index=False)
                writers[m].write_table(table.cast(INTERIM_SCHEMA))
                rows_by_month[m] += len(part)
            print(f"  chunk {i}: {stats.counts['raw_rows']:,} rows read")
    finally:
        for w in writers.values():
            w.close()
    return dict(rows_by_month)


# --------------------------------------------------------------------------------------------------
# Pass 2: clean one month (pure function of its inputs)
# --------------------------------------------------------------------------------------------------
def make_run_id(train_no: pd.Series, date: pd.Series) -> pd.Series:
    days = (date - RUN_ID_EPOCH).dt.days.astype("int64")
    return train_no.astype("int64") * RUN_ID_MULT + days


def clean_month(df: pd.DataFrame, schedule: pd.DataFrame, trains: pd.DataFrame,
                stations: pd.DataFrame, stats: QualityStats) -> pd.DataFrame:
    """Apply the Phase 1 cleaning rules to complete runs. Returns CLEAN_COLUMNS, sorted by run and stop."""
    c = stats.counts

    n = len(df)
    df = df.drop_duplicates(DELAY_KEYS, keep="first")
    c["dropped_duplicates"] += n - len(df)

    # Join on station *code*, not station_no: for ~40% of trains the delay file numbers stops differently
    # from the schedule (offsets of +-1..5), so a station_no join attaches the wrong stop's schedule.
    # (train_no, station_name) is unique in the schedule; the schedule's station_no becomes the stop order.
    df = df.rename(columns={"station_no": "station_no_raw"})
    df["station_name"] = df["station_name"].astype("string").str.strip()
    df = df.merge(schedule, on=["train_no", "station_name"], how="left", indicator=True, validate="many_to_one")
    unmatched = df["_merge"] == "left_only"
    c["dropped_no_schedule"] += int(unmatched.sum())
    df = df.loc[~unmatched].drop(columns="_merge")
    # the left join made these object/float while unmatched rows existed; restore dtypes
    df = df.astype({"station_no": "int16", "is_origin": bool, "is_terminus": bool, "n_sched_stops": "int16"})
    c["station_no_renumbered"] += int((df["station_no_raw"] != df["station_no"]).sum())
    n = len(df)
    df = df.drop_duplicates(DELAY_KEYS, keep="first")  # two raw stops mapping to one schedule stop
    c["dropped_duplicates"] += n - len(df)
    df = df.sort_values(DELAY_KEYS)
    back = df.groupby(RUN_KEYS)["station_no_raw"].diff() < 0
    c["stop_order_inversions"] += int(back.sum())
    df = df.drop(columns="station_no_raw")

    for lookup, key, name in ((trains, "train_no", "train"), (stations, "station_name", "station")):
        df = df.merge(lookup, on=key, how="left", indicator=True, validate="many_to_one")
        miss_col = "rows_train_not_in_details" if name == "train" else "rows_station_not_in_names"
        c[miss_col] += int((df["_merge"] == "left_only").sum())
        df = df.drop(columns="_merge")

    raw = df["delay"].astype("float32")
    above = raw > C.MAX_PLAUSIBLE_DELAY_MIN
    c["delay_above_cap_set_nan"] += int(above.sum())
    raw = raw.mask(above)
    c["negative_clipped"] += int((raw < 0).sum())
    df["delay_raw"] = raw
    df["delay"] = raw.clip(lower=0)
    df["delay_missing"] = raw.isna()

    observed = df.groupby(RUN_KEYS)["delay_raw"].transform("count")
    short = observed < C.MIN_OBSERVED_STOPS_PER_RUN
    c["dropped_short_run_rows"] += int(short.sum())
    c["dropped_short_runs"] += len(df.loc[short, RUN_KEYS].drop_duplicates())
    df = df.loc[~short].copy()

    df["run_id"] = make_run_id(df["train_no"], df["date"])
    df["sched_arr_ts"] = df["date"] + pd.to_timedelta(df["sched_arr_offset_min"], unit="min")
    df = df.sort_values(DELAY_KEYS, ignore_index=True)
    _accumulate_clean_stats(df, stats)
    return df[CLEAN_COLUMNS]


def _accumulate_clean_stats(df: pd.DataFrame, stats: QualityStats) -> None:
    if df.empty:
        return
    c = stats.counts
    c["clean_rows"] += len(df)
    c["clean_runs"] += df["run_id"].nunique()
    c["clean_missing"] += int(df["delay_missing"].sum())
    c["runs_with_any_missing"] += int(df.groupby("run_id")["delay_missing"].any().sum())
    c["clean_delay_gt_threshold"] += int((df["delay"] > C.SIGNIFICANT_DELAY_MIN).sum())

    position = pd.Series(np.select([df["is_origin"], df["is_terminus"]], ["origin", "terminus"],
                                   "intermediate"), index=df.index)
    for (p, miss), v in df.groupby([position, df["delay_missing"]]).size().items():
        stats.missing_by_position[(p, bool(miss))] += int(v)
    month = df["date"].dt.strftime("%Y-%m")
    for (m, miss), v in df.groupby([month, df["delay_missing"]]).size().items():
        stats.missing_by_month[(m, bool(miss))] += int(v)

    per_run = df.groupby("run_id").agg(obs=("delay_raw", "count"), sched=("n_sched_stops", "first"))
    cov = np.clip(np.floor(per_run["obs"] / per_run["sched"] * 10) / 10, 0, 1)
    stats.coverage_hist.update({float(k): int(v) for k, v in cov.value_counts().items()})

    stats.trains.update(df["train_no"].unique().tolist())
    stats.stations.update(df["station_name"].dropna().unique().tolist())
    lo, hi = df["date"].min(), df["date"].max()
    stats.date_min = lo if stats.date_min is None else min(stats.date_min, lo)
    stats.date_max = hi if stats.date_max is None else max(stats.date_max, hi)


# --------------------------------------------------------------------------------------------------
# Is `date` the run's start date?
# --------------------------------------------------------------------------------------------------
def weekday_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per (train_no, stop_day, weekday of `date`) -- input to date_check_weekday."""
    out = pd.DataFrame({
        "train_no": df["train_no"],
        "stop_day": df["arrival_day"].fillna(df["departure_day"]),
        "weekday": df["date"].dt.day_of_week,
    }).dropna()
    return out.groupby(["train_no", "stop_day", "weekday"], as_index=False).size().rename(columns={"size": "n"})


def date_check_weekday(counts: pd.DataFrame, min_share: float = 0.95, core_share: float = 0.05,
                       max_weekdays: int = 3) -> dict:
    """Test A. For non-daily multi-day trains, the weekdays seen at later stop-days must equal the day-1
    weekdays if `date` is the start date, or be shifted by (stop_day - 1) if it is the stop's calendar date."""
    counts = counts.groupby(["train_no", "stop_day", "weekday"], as_index=False)["n"].sum()
    res: Counter = Counter()
    labels = {(True, False): "start_only", (False, True): "calendar_only", (True, True): "both", (False, False): "neither"}
    for _, g in counts.groupby("train_no"):
        if g["stop_day"].max() < 2:
            continue
        d1 = g[g["stop_day"] == 1]
        if d1.empty:
            continue
        w1 = set(d1.loc[d1["n"] / d1["n"].sum() >= core_share, "weekday"])
        if len(w1) > max_weekdays:
            res["skipped_frequent"] += 1
            continue
        later = g[g["stop_day"] >= 2]
        total = later["n"].sum()
        start_ok = later.loc[later["weekday"].isin(w1), "n"].sum() / total >= min_share
        shifted = (later["weekday"] - (later["stop_day"].astype(int) - 1)) % 7
        cal_ok = later.loc[shifted.isin(w1), "n"].sum() / total >= min_share
        res[labels[(bool(start_ok), bool(cal_ok))]] += 1

    decisive = res["start_only"] + res["calendar_only"]
    if decisive == 0:
        verdict = INCONCLUSIVE
    elif res["start_only"] / decisive >= 0.8:
        verdict = START_DATE
    elif res["calendar_only"] / decisive >= 0.8:
        verdict = CALENDAR_DATE
    else:
        verdict = INCONCLUSIVE
    return {"verdict": verdict, **{k: int(res[k]) for k in
            ("start_only", "calendar_only", "both", "neither", "skipped_frequent")}}


def day_boundaries(schedule: pd.DataFrame) -> pd.DataFrame:
    """For multi-day trains: the last stop of stop-day d and the first stop of stop-day d+1."""
    s = schedule[["train_no", "station_no", "stop_day"]].dropna()
    g = s.groupby(["train_no", "stop_day"])["station_no"]
    last = g.max().rename("prev_station_no").reset_index()
    first = g.min().rename("next_station_no").reset_index()
    first["stop_day"] = first["stop_day"] - 1
    return last.merge(first, on=["train_no", "stop_day"])[["train_no", "prev_station_no", "next_station_no"]]


def continuity_pairs(df: pd.DataFrame, boundaries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Delay pairs across each midnight boundary: next stop on the same `date`, and on `date + 1`."""
    obs = df.loc[df["delay_raw"].notna(), ["train_no", "date", "station_no", "delay_raw"]]
    prev = obs.merge(boundaries, left_on=["train_no", "station_no"], right_on=["train_no", "prev_station_no"])
    prev = prev[["train_no", "date", "next_station_no", "delay_raw"]].rename(columns={"delay_raw": "x"})
    nxt = obs.rename(columns={"station_no": "next_station_no", "delay_raw": "y"})
    keys = ["train_no", "date", "next_station_no"]
    same = prev.merge(nxt, on=keys)
    shifted = prev.merge(nxt.assign(date=nxt["date"] - pd.Timedelta(days=1)), on=keys)
    return same[["x", "y"]], shifted[["x", "y"]]


def date_check_continuity(same: PairStats, nxt: PairStats) -> dict:
    """Test B. Delay is continuous along a run, so the pairing that keeps runs intact has the smaller
    |delay jump| and the higher correlation across midnight."""
    rs, rn, ms, mn = same.corr(), nxt.corr(), same.median_absdiff(), nxt.median_absdiff()
    if same.n >= MIN_PAIRS and (nxt.n < MIN_PAIRS or (ms < mn and rs > rn)):
        verdict = START_DATE
    elif nxt.n >= MIN_PAIRS and (same.n < MIN_PAIRS or (mn < ms and rn > rs)):
        verdict = CALENDAR_DATE
    else:
        verdict = INCONCLUSIVE
    r = lambda v: None if np.isnan(v) else round(v, 4)  # noqa: E731
    return {"verdict": verdict,
            "same_date": {"pairs": same.n, "corr": r(rs), "median_abs_jump_min": r(ms)},
            "next_date": {"pairs": nxt.n, "corr": r(rn), "median_abs_jump_min": r(mn)}}


def combine_verdicts(a: str, b: str) -> str:
    if a == b:
        return a
    if INCONCLUSIVE in (a, b):
        return b if a == INCONCLUSIVE else a
    return INCONCLUSIVE


# --------------------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------------------
def select_candidate_runs(df: pd.DataFrame, p: float, rng: np.random.Generator) -> pd.DataFrame:
    runs = df["run_id"].unique()
    keep = runs[rng.random(len(runs)) < p]
    return df[df["run_id"].isin(keep)]


def sample_runs(candidates: pd.DataFrame, target_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Random whole runs whose total size is <= target_rows."""
    sizes = candidates.groupby("run_id").size()
    order = rng.permutation(sizes.index.to_numpy())
    keep = order[sizes.loc[order].cumsum().to_numpy() <= target_rows]
    return candidates[candidates["run_id"].isin(keep)].sort_values(DELAY_KEYS, ignore_index=True)


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------
def to_arrow(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df[CLEAN_COLUMNS], preserve_index=False).cast(CLEAN_SCHEMA)


def build(nrows: int | None = None, chunksize: int = C.DELAY_CHUNKSIZE, keep_interim: bool = False) -> QualityStats:
    t0 = time.time()
    stats = QualityStats(meta={"nrows_limit": nrows, "chunksize": chunksize,
                               "max_plausible_delay_min": C.MAX_PLAUSIBLE_DELAY_MIN,
                               "min_observed_stops_per_run": C.MIN_OBSERVED_STOPS_PER_RUN,
                               "significant_delay_min": C.SIGNIFICANT_DELAY_MIN,
                               "random_state": C.RANDOM_STATE})
    schedule, trains, stations = load_schedule(stats), load_trains(stats), load_stations(stats)
    boundaries = day_boundaries(schedule)

    interim = C.INTERIM_BY_MONTH
    if interim.exists():
        shutil.rmtree(interim)
    print("Pass 1: splitting combined_delay.csv by month ...")
    months = split_by_month(C.RAW_FILES["delay"], interim, stats, chunksize, nrows)
    total = sum(months.values())
    p = min(1.0, C.SAMPLE_OVERDRAW * C.SAMPLE_TARGET_ROWS / max(total, 1))
    rng = np.random.default_rng(C.RANDOM_STATE)

    print(f"Pass 2: cleaning {len(months)} months ({total:,} rows) ...")
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp_clean = C.CLEAN_PARQUET.with_name(C.CLEAN_PARQUET.name + ".tmp")
    candidates = []
    with pq.ParquetWriter(tmp_clean, CLEAN_SCHEMA) as writer:
        for m in sorted(months):
            clean = clean_month(pq.read_table(interim / f"{m}.parquet").to_pandas(), schedule, trains, stations, stats)
            if clean.empty:
                continue
            writer.write_table(to_arrow(clean))
            stats.weekday_parts.append(weekday_counts(clean))
            same, nxt = continuity_pairs(clean, boundaries)
            stats.pairs_same.add(same["x"], same["y"])
            stats.pairs_next.add(nxt["x"], nxt["y"])
            candidates.append(select_candidate_runs(clean, p, rng))
            print(f"  {m}: {len(clean):,} clean rows")

    wd = pd.concat(stats.weekday_parts) if stats.weekday_parts else pd.DataFrame(
        columns=["train_no", "stop_day", "weekday", "n"])
    test_a = date_check_weekday(wd)
    test_b = date_check_continuity(stats.pairs_same, stats.pairs_next)
    stats.date_check = {"verdict": combine_verdicts(test_a["verdict"], test_b["verdict"]),
                        "test_a_weekday": test_a, "test_b_continuity": test_b}

    if stats.date_check["verdict"] == CALENDAR_DATE:
        tmp_clean.unlink(missing_ok=True)
        write_report(stats)
        raise SystemExit("`date` looks like the stop's calendar date, not the run start date. Run ids would "
                         "need date - (arrival_day - 1). No processed files written; see "
                         f"{C.DATA_QUALITY_MD}.")

    os.replace(tmp_clean, C.CLEAN_PARQUET)
    if candidates:
        sample = sample_runs(pd.concat(candidates, ignore_index=True), C.SAMPLE_TARGET_ROWS, rng)
        pq.write_table(to_arrow(sample), C.SAMPLE_PARQUET)
        stats.counts["sample_rows"] = len(sample)
        stats.counts["sample_runs"] = int(sample["run_id"].nunique())
    if not keep_interim:
        shutil.rmtree(interim, ignore_errors=True)
    stats.meta["elapsed_s"] = round(time.time() - t0, 1)
    write_report(stats)
    return stats


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------
def write_report(stats: QualityStats) -> None:
    d = stats.to_dict()
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.DATA_QUALITY_JSON.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
    C.DATA_QUALITY_MD.write_text(render_markdown(d), encoding="utf-8")
    print_summary(d)


def render_markdown(d: dict) -> str:
    rd, cl, dc = d["raw_delay"], d["clean"], d["date_check"]
    L = ["# Data-quality report (Phase 1)", ""]
    if d["meta"].get("nrows_limit"):
        L += [f"> **Partial run:** first {d['meta']['nrows_limit']:,} rows only.", ""]
    L += ["## Row waterfall", "", "| Step | Rows |", "|---|---:|"]
    L += [f"| {s} | {n:,} |" for s, n in d["waterfall"]]
    L += ["", f"Clean: **{cl['rows']:,} rows**, {cl['runs']:,} runs, {cl['trains']:,} trains, "
          f"{cl['stations']:,} stations, {cl['date_min']} to {cl['date_max']}.", ""]

    L += ["## Missing and negative delays", "", "| Metric | Raw | After cleaning |", "|---|---:|---:|",
          f"| Missing (NaN) | {rd['missing']:,} ({rd['missing_pct']}%) | {cl['missing']:,} ({cl['missing_pct']}%) |",
          f"| Negative (early) | {rd['negative']:,} ({rd['negative_pct']}%) | {cl['negative_clipped_to_0']:,} clipped to 0 |",
          f"| > {d['meta']['max_plausible_delay_min']} min | {rd['above_cap']:,} | {cl['above_cap_set_nan']:,} set to NaN |",
          "",
          f"- Most negative raw delay: {rd['negative_min']} min; median of negatives: {rd['negative_median']} min.",
          f"- Raw delay quantiles (min): {rd['quantiles']}, max {rd['max']}.",
          f"- Runs with at least one missing stop: {cl['runs_with_any_missing']:,} ({cl['runs_with_any_missing_pct']}%).",
          f"- Share of observed stops delayed > {d['meta']['significant_delay_min']} min: {cl['delay_gt_threshold_pct']}%.",
          "", "| Stop position | Rows | Missing | % |", "|---|---:|---:|---:|"]
    L += [f"| {p} | {v['rows']:,} | {v['missing']:,} | {v['missing_pct']} |" for p, v in cl["missing_by_position"].items()]
    L += ["", "Missing % by month: " + ", ".join(f"{m}: {v}" for m, v in cl["missing_pct_by_month"].items()), ""]

    L += ["## Is `date` the run's start date?", "", f"**Verdict: {dc.get('verdict')}**", ""]
    a, b = dc.get("test_a_weekday", {}), dc.get("test_b_continuity", {})
    if a:
        L += [f"- *Test A (weekday consistency, non-daily multi-day trains)*: {a['verdict']}. "
              f"Trains consistent only with start date: {a['start_only']}, only with calendar date: "
              f"{a['calendar_only']}, both: {a['both']}, neither: {a['neither']} "
              f"(skipped {a['skipped_frequent']} daily/frequent trains)."]
    if b:
        s, n = b["same_date"], b["next_date"]
        L += [f"- *Test B (delay continuity across midnight)*: {b['verdict']}. Pairing the next-day stop on the "
              f"**same** date: {s['pairs']:,} pairs, corr {s['corr']}, median |jump| {s['median_abs_jump_min']} min. "
              f"On **date + 1**: {n['pairs']:,} pairs, corr {n['corr']}, median |jump| {n['median_abs_jump_min']} min."]
    L += [""]

    L += ["## Join and schedule health", "", "| Check | Count |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in {**d["joins"], **d["lookups"]}.items()]
    L += ["", "## Run coverage (observed delays / scheduled stops)", "", "| Decile | Runs |", "|---|---:|"]
    L += [f"| {k} | {v:,} |" for k, v in d["run_coverage_deciles"].items()]
    L += ["", "## Sample", "", f"{d['sample']['rows']:,} rows from {d['sample']['runs']:,} whole runs.", ""]
    return "\n".join(L)


def print_summary(d: dict) -> None:
    rd, cl, dc = d["raw_delay"], d["clean"], d["date_check"]
    print("\n=== Data-quality summary ===")
    for s, n in d["waterfall"]:
        print(f"  {s:55s} {n:>12,}")
    print(f"  raw missing delays:  {rd['missing']:,} ({rd['missing_pct']}%)")
    print(f"  raw negative delays: {rd['negative']:,} ({rd['negative_pct']}%), min {rd['negative_min']}")
    print(f"  clean missing:       {cl['missing']:,} ({cl['missing_pct']}%)")
    print(f"  date is run start?   {dc.get('verdict')}")
    print(f"  sample:              {d['sample']['rows']:,} rows / {d['sample']['runs']:,} runs")
    print(f"  report:              {C.DATA_QUALITY_MD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows (smoke run)")
    ap.add_argument("--chunksize", type=int, default=C.DELAY_CHUNKSIZE)
    ap.add_argument("--keep-interim", action="store_true", help="keep data/interim/delay_by_month/")
    args = ap.parse_args()
    build(nrows=args.nrows, chunksize=args.chunksize, keep_interim=args.keep_interim)


if __name__ == "__main__":
    main()
