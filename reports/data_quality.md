# Data-quality report (Phase 1)

## Row waterfall

| Step | Rows |
|---|---:|
| raw rows | 38,428,703 |
| valid keys and date | 38,428,703 |
| after dedupe | 38,428,703 |
| matched to schedule | 37,133,436 |
| runs with >= 3 observed delays (final) | 36,500,515 |

Clean: **36,500,515 rows**, 1,830,284 runs, 6,908 trains, 8,142 stations, 2025-02-08 to 2026-02-07.

## Missing and negative delays

| Metric | Raw | After cleaning |
|---|---:|---:|
| Missing (NaN) | 1,874,271 (4.877%) | 474,709 (1.301%) |
| Negative (early) | 773,174 (2.012%) | 760,141 clipped to 0 |
| > 1440 min | 15,073 | 14,899 set to NaN |

- Most negative raw delay: -120 min; median of negatives: -10 min.
- Raw delay quantiles (min): {'p1': -11, 'p5': 0, 'p25': 4, 'p50': 16, 'p75': 41, 'p95': 139, 'p99': 315}, max 525592.
- Runs with at least one missing stop: 225,446 (12.318%).
- Share of observed stops delayed > 15 min: 50.588%.

| Stop position | Rows | Missing | % |
|---|---:|---:|---:|
| origin | 1,824,898 | 26,816 | 1.469 |
| intermediate | 32,851,849 | 416,908 | 1.269 |
| terminus | 1,823,768 | 30,985 | 1.699 |

Missing % by month: 2025-02: 2.388, 2025-03: 2.286, 2025-04: 2.217, 2025-05: 2.046, 2025-06: 1.839, 2025-07: 1.673, 2025-08: 1.234, 2025-09: 0.707, 2025-10: 0.451, 2025-11: 0.397, 2025-12: 0.559, 2026-01: 0.384, 2026-02: 0.453

## Is `date` the run's start date?

**Verdict: START_DATE**

- *Test A (weekday consistency, non-daily multi-day trains)*: START_DATE. Trains consistent only with start date: 1704, only with calendar date: 0, both: 0, neither: 0 (skipped 1084 daily/frequent trains).
- *Test B (delay continuity across midnight)*: START_DATE. Pairing the next-day stop on the **same** date: 539,873 pairs, corr 0.9464, median |jump| 8.0 min. On **date + 1**: 367,788 pairs, corr 0.5618, median |jump| 16.0 min.

## Join and schedule health

| Check | Count |
|---|---:|
| dropped_bad_key | 0 |
| dropped_bad_date | 0 |
| dropped_duplicates | 0 |
| dropped_no_schedule | 1,295,267 |
| station_no_renumbered_to_schedule | 6,634,445 |
| stop_order_inversions_vs_schedule | 40,277 |
| rows_train_not_in_details | 0 |
| rows_station_not_in_names | 0 |
| dropped_short_runs | 54,784 |
| dropped_short_run_rows | 632,921 |
| schedule_rows | 172,112 |
| schedule_trains | 8,673 |
| schedule_duplicates | 0 |
| schedule_repeated_station | 0 |
| schedule_unparsed_times | 0 |
| schedule_offset_violation_trains | 0 |
| trains_duplicate_keys | 272 |
| stations_duplicate_keys | 0 |

## Run coverage (observed delays / scheduled stops)

| Decile | Runs |
|---|---:|
| 0.0 | 821 |
| 0.1 | 284 |
| 0.2 | 411 |
| 0.3 | 1,836 |
| 0.4 | 1,115 |
| 0.5 | 3,219 |
| 0.6 | 6,289 |
| 0.7 | 7,842 |
| 0.8 | 44,000 |
| 0.9 | 193,039 |
| 1.0 | 1,571,428 |

## Sample

99,999 rows from 4,988 whole runs.
