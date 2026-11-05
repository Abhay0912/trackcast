# TrackCast: Interview Notes

For each phase: **what** was built, **why**, **alternatives rejected**, **trade-offs**, **what could go wrong**.

## Phase 0: Setup

**What:** Python venv, `src/` package layout installed in editable mode, git, data folders kept out of git, a Kaggle download script.

**Why**
- *src layout + editable install:* notebooks, the app and tests all import the same `trackcast` code, so logic is never copy-pasted between notebooks. Tests run against the code that is actually in use.
- *Single `config.py`:* paths and thresholds (e.g. the 15-min hurdle) are defined once; changing the threshold changes it everywhere.
- *Data not in git:* the raw file is ~1 GB, and anyone can rebuild it by running `python -m trackcast.download`. Git tracks code, not data.
- *Dataset choice:* `rxydenxd/indian-railways-delay-dataset` has delays per train, per day and per station, which is what lag and cascade features need. The alternative (naijilaji) only had per-station averages, with no sequence of stops for a single run, so no cascade could be modelled.

**What the raw data looks like**
- The delay table is in *long format*: one row per (train, date, stop), with the delay already given in minutes. There are no actual arrival timestamps, so equation (1) of the proposal is already computed upstream. We use it directly and clip negatives to 0.
- A train *run* = (train_no, date). Lag features are `groupby(run)` sorted by `station_no`, then `shift(1)` / `shift(2)`.
- The schedule gives arrival and departure per stop plus a day offset for multi-day trains. That lets us compute dwell time (dep − arr), scheduled run time between stops, and slack.
- Missing delays appear as NaN (not 0). Treating them as 0 would wrongly make trains look punctual.

**Trade-offs:** venv + requirements.txt is simpler than conda or Poetry, but it doesn't pin system libraries. That's fine for a pure-Python stack.

**Likely questions**
- *How would someone reproduce your results?* Clone the repo → create the venv → `pip install -r requirements.txt` → `pip install -e .` → `python -m trackcast.download` → run the pipeline.
- *Why not keep everything in notebooks?* Notebooks are hard to test, diff and reuse. Modules can be tested and imported by the Streamlit app.

## Phase 1: Loading, joining and cleaning (`src/trackcast/data.py`)

**What:** `python -m trackcast.data` streams `combined_delay.csv`, joins it with the schedule, trains and
stations tables, applies the cleaning rules, and writes three things:
- `data/processed/clean.parquet` (all clean rows);
- `data/processed/sample.parquet` (about 100k rows, whole runs only);
- a data-quality report (`reports/data_quality.md` and `.json`).

**Why it is built this way**
- *Two passes, split by month.* The CSV is 1 GB and its sort order is not guaranteed, so one run can be spread
  across many chunks.
  - Pass 1 streams the chunks, counts the raw issues and writes one interim parquet file per month.
  - A run is keyed on `(train_no, date)`, so each month file holds complete runs. Pass 2 can then dedupe,
    filter runs and sample with one month in memory, never the whole year.
- *Join on `(train_no, station_name)`, not `(train_no, station_no)`.* This was found during the smoke run.
  - For about 40% of trains the delay file numbers stops differently from the schedule (offsets of ±1 to 5).
    A `station_no` join attached the wrong stop's scheduled times to about 18% of rows, and it did so silently.
  - The station code is unique per train in both files, so the code join is safely many-to-one
    (`validate="many_to_one"` enforces it).
  - The schedule's `station_no` becomes the stop order, because it agrees with scheduled times and distance.
  - Lesson: check a join key by matching on a *second* attribute, not only by the match rate.
- *Lookups are deduped before the left joins.* `train_details` has repeated `train_no` values, which would have
  multiplied rows.
- *NaN delays are kept, never filled with 0.* Filling with 0 would make trains look punctual and would bias both
  the class balance and the lag features. Phase 2 decides how lag works across a gap; the data layer does not.
- *`delay` is clipped at 0, and `delay_raw` keeps the early arrivals.* The target is lateness, but early
  arrivals matter for the route profiles and for slack (a train that arrives early had spare time).
- *Delays above 24 h are set to NaN.* They are treated as recording errors, and the threshold is in config. The
  trade-off is that a genuine extreme fog delay above 24 h would be lost; the report counts how many there are.
- *Runs with fewer than 3 observed delays are dropped.* Delay velocity needs stops k-1 and k-2.
- *The sample is drawn by run, not by row.* Row sampling would break the stop-to-stop sequences that lag
  features need. Runs are drawn in every month, so a by-date train/test split still works on the sample.
- *Typed parquet with fixed dtypes.* Codes are stored as dictionary (category) columns, keys as int16/int32 and
  delays as float32, so it loads several times faster than the CSV and uses much less memory.

**Is `date` the run's start date? (this decides what a run is)** Two independent tests on multi-day trains:
- **A, weekdays.** A weekly train that starts on Mondays shows Monday on every one of its rows if `date` is
  the start date. If `date` were the stop's own calendar date, the rows for day-2 stops would show Tuesday.
- **B, continuity.** Delay barely changes between consecutive stops of one run. Pairing the last day-1 stop
  with the first day-2 stop on the *same* `date` should give a much higher correlation and a smaller jump than
  pairing it with the stop on `date + 1`.

If the verdict were `CALENDAR_DATE`, the build stops before writing outputs, rather than producing wrong run ids.

**Leakage note:** Phase 1 builds no features. It only attaches information known *before* the run: the
scheduled times, distance and train/station attributes. Lag and slack are built in Phase 2 with `shift()`
inside each run.

**Likely questions**
- *Why not load everything with pandas or Polars?* At about 25M rows times 5 columns it might fit, but not
  comfortably on a laptop next to the join outputs. Streaming caps memory at one chunk or one month.
- *How do you know the cleaning didn't remove something important?* Every rule is counted in the report's
  waterfall, and each rule has a unit test on synthetic data.

**What the full run found** (38.4M raw rows, 8.5 min; details in `reports/data_quality.md`)
- **Rows:** 36.5M clean rows from 1.83M runs, 6,908 trains and 8,142 stations, covering 2025-02-08 to
  2026-02-07. That is 95% of the raw rows kept.
- **Missing delays:** 4.9% of raw rows. After cleaning it is 1.3%, because most missing rows were stops not in
  the schedule or runs that were too short.
  - Missingness falls over the year, from 2.4% in Feb 2025 to 0.4% in Jan 2026, so collection improved.
    A temporal split will therefore see different missingness in train and test.
- **Negative delays:** 2.0% of rows (median −10 min, minimum −120 min). They are clipped to 0 in `delay`.
- **Implausible delays:** 15k rows were above 24 h, the largest at 525,592 min (about a year), a clear
  recording error. All of them were set to NaN.
- **Class balance:** 50.6% of observed stops are more than 15 min late, so Stage 1 is close to balanced.
- **`date` is the run start date.** Test A: 1,704 non-daily trains were consistent only with the start date and
  0 only with the calendar date. Test B: corr 0.95 vs 0.56 across midnight, median jump 8 vs 16 min.
- **Station renumbering:** 6.6M rows (18%) had their `station_no` renumbered to the schedule's. 40k rows (0.1%)
  are in a different stop order from the schedule; we follow the schedule's order.

## Phase 2: Feature engineering (`src/trackcast/features.py`)

**What:** `python -m trackcast.features` turns the sample (or `--full`: `clean.parquet`, one month at a time) into
one row per predictable stop, with about 35 features and three targets (`delay`, `y_late` = delay > 15,
`y_log` = log1p(delay)). It writes `features_sample.parquet` / `features.parquet` and `reports/features.md`.
The feature lists (`NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`) are exported so Phase 3 imports them, not copies.

**The information rule (the leakage contract).** A row for stop k may use delays observed at stops <= k - H of
the same run (H = horizon, default 1), plus anything in the timetable. It may not use the delay at k or later, the
*actual* time at k (actual = scheduled + target), or any statistic fitted on delays across runs. A property test
enforces it: random changes to the delays at stops >= k (including flipping them to missing) must leave every
feature of stop k bit-for-bit unchanged, for H = 1 and H = 2.

**Feature groups and why**
- *Cascade.* `lag1` and `lag2` are the last two **observed** upstream delays, `delay_velocity = lag1 - lag2`.
  - A missing stop is skipped, never treated as 0 (0 would read as "the train caught up"). Two features say
    how stale the lag is: `lag_stops_back` and `lag_sched_min_back` (scheduled minutes since the lag stop).
  - Implemented with position arithmetic (`np.maximum.accumulate` of observed row positions, cut at the run
    boundary), not `groupby().shift()`: one vectorised pass, and correct across gaps.
  - `lag_from_origin` flags that lag1 is the origin's *departure* delay, a slightly different quantity.
  - `upstream_max_delay` and `n_upstream_obs` describe the run so far.
- *Schedule slack.* Built from the timetable only, so no delay data is needed and it cannot leak.
  - `run_slack_min` is a train's scheduled time on a station pair minus the fastest train scheduled on the same
    pair. Dwell slack is the scheduled dwell minus the shortest dwell any train has at that station.
  - Both are summed into `cum_slack_min` along the route, so the slack between the lag stop j and stop k is just
    `cum_slack[k] - cum_slack[j]` (`slack_since_lag`). That makes it work for any horizon and across gaps.
  - "Minimum observed" in the proposal is read as the minimum *scheduled* time across trains, not the minimum
    *actual* run time. The actual-time version would need delays from training dates only, so it belongs inside
    the CV folds (Phase 3) or the route profiles (Phase 4).
- *Time.* Hour, day of week and month as sin/cos pairs of the **scheduled** time at k, so 23:00 and 00:00 are
  neighbours. The day of week is from the stop's own calendar date (day-2 stops roll over), not the run date.
- *Weather / season.* The dataset has no weather and no station coordinates, so an Open-Meteo join would first
  need 8k stations geocoded from free-text addresses. Instead: `fog_season` (Dec to Feb in the northern zones NR,
  NCR, NER, NWR, ECR and ER), `fog_night` (that, between 21:00 and 10:00) and `monsoon` (Jun to Sep). The zones
  and months are in config.
- *Congestion.* `station_trains_hour` counts the distinct trains scheduled at the station in the same clock hour,
  and `station_trains_day` counts all trains stopping there. The timetable has no running days, so each train
  counts as daily (an upper bound). The proposal says "through the station", but the timetable only lists
  *stops*, so trains passing through without stopping are invisible.
- *Categoricals.* `station_name`, `type_code` and `station_zone` stay raw. The target encoding of 8k stations is
  fitted inside each CV fold in Phase 3; doing it here over all rows would leak the test targets into the train
  features.

**Which rows are model rows:** the delay is observed, the stop is not the origin (its delay is a departure delay
with nothing upstream), and at least one upstream delay was observed. Rows without lag2 are kept with NaN, and
Phase 3 imputes per model (HistGradientBoosting handles NaN; Ridge/LogReg get an imputer + missing indicator).

**Horizon parameter.** `--horizon H` forecasts H stops ahead using only stops <= k - H. H = 1 is the proposal's
setting. Larger H is the more useful "how far does the delay cascade" question and costs nothing extra, because
every lag feature is defined relative to the lag stop j.

**What the sample showed** (93.6k model rows from the 100k-row sample)
- *Persistence is a strong baseline.* Predicting delay_k = lag1 gives F1 0.875 for "late > 15 min" and R2 0.59
  on log1p. Phase 3 has to report its gain over this. A high absolute R2 on its own would be misleading.
- *Slack is real signal.* Among trains already late, the share that recover by stop k rises from 31% with no
  slack in between, to 60% at 10 to 20 min, to 85% above 40 min (mean change from +0.7 to -29 min). Slack's
  overall rank correlation with the target is close to 0, because it acts on the *change* in delay, not its
  level. Linear models need the interaction (lag1 x slack); trees find it themselves.
- *Gaps are rare.* Only about 1% of rows take their lag from further back than the previous stop.

**What the full run found** (`--full`, 3.7 min, one month in memory at a time)
- 36.5M clean stops became **34.2M model rows** (`features.parquet`, 483 MB), with 52.6% late > 15 min.
- Persistence baseline on a 468k-row subsample: RMSLE 0.969, R2(log1p) 0.596, F1(late) 0.876. That is within
  0.01 of the sample, so the 100k-row sample is representative for development.
- The slack recovery curve is the same on full data: 31% recover with no slack and 86% with more than 40 min
  (mean change from +0.6 to -28.5 min).

**Trade-offs and what could go wrong**
- Slack is measured against the fastest scheduled train on the same station pair. A passenger train compared
  with a Rajdhani on the same pair shows "slack" that is really a lower speed limit. Per-type minima would fix
  this, but segments with a single train type would then always show 0.
- `seg_speed_kmph` reaches 900 km/h when two stops are scheduled 1 minute apart (the timetable is rounded to
  the minute). It is harmless for trees; for Ridge, clip it or use a robust scaler.
- Congestion counts every train as daily, which overstates congestion at stations served by weekly trains.

**Likely questions**
- *Why not `groupby('run_id').shift(1)` for the lag?* That returns NaN whenever the previous stop is missing,
  which throws away rows that still have a usable earlier observation. "Last observed" plus a staleness feature
  keeps them and tells the model how far back the lag is.
- *Isn't the timetable "future information" for stop k?* No. It is published before the train departs, so it is
  known at prediction time. Leakage means using the realised outcome, not using the plan.
- *How do you know there is no leakage?* By construction (every delay-derived feature is gathered at a position
  <= k - H) and by test (the perturbation test above). Cross-run statistics are deferred to fold-fitted
  encoders.

## Phase 3: Two-stage hurdle models (`src/trackcast/models.py`, `evaluation.py`, `train.py`)

**What:** `python -m trackcast.train [--full]` tunes, fits and scores the hurdle model on a temporal holdout and
writes `models/hurdle[_sample].joblib`, the test predictions (`data/processed/test_predictions[_sample].parquet`)
and `reports/models.md` / `.json`.
- Stage 1: P(delay > 15 min), LogisticRegression vs RandomForest, then isotonic calibration.
- Stage 2: log1p(delay), Ridge vs HistGradientBoosting, fitted separately on late rows (m_late) and on-time rows (m_low).
- Hurdle forecast: E[log1p delay] = p * m_late + (1 - p) * m_low (law of total expectation, the RMSLE-optimal
  point forecast when both parts are conditional means).

**Evaluation protocol (why each piece)**
- *Split by date, not rows or runs.* Test = runs starting on or after 2025-12-01 (Dec to Feb). Random rows would
  put stops of the same run on both sides, so the model would be graded on near-copies of its training rows.
  A run-level split would still share days: same-day trains share fog, a blocked line and network state. A date
  split is also the honest deployment question ("trained on the past, used on the future").
- *3-day embargo* before every cutoff. A multi-day run that starts on 29 Nov is still running on 1 Dec, the
  same days the test runs use.
- *Forward-chaining CV inside the training period* (Sep, Oct and Nov each validate once on an expanding
  window). It picks the hyper-parameters per family and part, the F1 threshold, the calibration map and the
  classifier x regressor pair. The test months are scored exactly once, at the end.
- *Baselines no model may ignore:* persistence (delay_k = lag1) and persistence minus slack. Every metric is
  reported as skill over persistence, with a **day-block bootstrap** CI. Errors on the same day are correlated,
  so a row bootstrap would give intervals that are far too narrow.
- *Ablations in the same table:* a single-stage ("direct") regressor of each family, and a "perfect Stage 1"
  hurdle that picks m_late / m_low using the true class. That bound shows how much a better classifier could add.

**Three design decisions that came from looking at the errors**
1. *Target encoding with folds grouped by date (`GroupedTargetEncoder`).* sklearn's TargetEncoder cross-fits
   with random-row KFold. Row (run r, stop k) would then be encoded from later stops of the *same run*, whose
   delays are almost the target, and from same-day trains. Training rows would carry a far more informative
   encoding than any test row gets. Grouping the cross-fit folds by date closes both channels. A unit test
   checks that a row is never encoded from its own date. The encoder sits inside the Pipeline, so it is refitted
   on every CV fold's training rows.
2. *Regress the residual over persistence (`PersistenceOffsetRegressor`).* Target = log1p(delay_k) - log1p(lag1);
   log1p(lag1) is added back at predict time. A tree fitted on the level puts lag1 = 300 to 500 min into one
   leaf with one constant output, an error of up to about 100 min. On the residual the same leaf only learns
   the gain or recovery. The encoders then encode "how much delay trains gain at this station". This improved
   every regressor (dev: direct HGB RMSLE 0.703 to 0.684).
3. *Calibrate Stage 1 (isotonic on pooled out-of-fold predictions).* The hurdle mixes in log space, so a
   miscalibrated p is costly. The RF gave trains already 3 h late a median P(late) of only 0.973, which put 3%
   weight on an ~8-min forecast and pulled a 250-min forecast about 25 min too low. Random forests keep
   probabilities away from 0 and 1 (averaging over trees). Calibration on the CV months (never the test set) cut
   the dev MAE for those trains from 47.7 to 29.2 min and the overall MAE from 9.3 to 7.7 min.

**Other choices**
- *Feature handling per family.* Linear models get log terms, a lateness flag, the log(lag1) x log(slack)
  interaction (Phase 2 showed slack acts on the *change* in delay), median imputation with missing indicators
  and scaling. RF gets imputation; HGB handles NaN natively. Station code and train number are target-encoded;
  train type and zone are one-hot.
- *Model selection metric.* Classifiers are tuned on log loss, a proper scoring rule, because the probability
  feeds the hurdle. F1 is reported with a threshold tuned on the CV months. Regressors are tuned on RMSE of
  log1p within their part.
- *Laptop limits (5.8 GB RAM).* `--full` samples whole runs (by a deterministic hash of run_id) up to 1M
  training and 1M test rows. Parquet row groups are streamed and filtered in Arrow, because reading a whole month
  ran out of memory. Each RF tree is fitted on at most 100k bootstrap rows. That is ~20x fewer training rows
  than exist, so the test set is still huge (a 1M-row test set gives tight CIs) but the models could gain a
  little from more data.

**What the results say** (dev sample: 75k train / 18k test rows; full-run numbers below)
- The hurdle (RF + HGB) beats persistence by ~29% RMSLE (CI excludes 0); Stage 1 F1 0.912 vs 0.891.
- **The hurdle does not beat a single-stage HGB on RMSLE** (dev 0.701 vs 0.684, CI excludes 0). The "perfect
  Stage 1" bound (0.506) shows why: the split itself is valuable, but a misclassified row gets the wrong component,
  and classification near 15 min is inherently noisy. The hurdle is kept as the app's model because it gives the
  user P(late) directly, and its Stage 1 is the proposal's classification task. The single-stage model is the
  better pure magnitude forecaster. Both are reported.
- *Known limitation:* for trains already more than 3 h late, persistence is still best in minutes (dev MAE 18 vs
  29 min). A log-scale loss treats 300 vs 330 min as a small error. A minutes-scale or quantile loss for the
  late part would be the next step.
- Feature groups (permutation, shuffled as a block): cascade >> slack > route > station > congestion; time and
  season add almost nothing beyond them. Fog shows up already in the upstream delay.

**Likely questions**
- *Why not tune the threshold on the test set?* Then the test score would be an optimisation result, not an
  estimate. The threshold, calibration and model pair are all chosen on the CV months.
- *Why is R2 so high?* Mostly persistence (R2 0.58 by itself). The honest headline is the skill over
  persistence and its CI.
- *Is the expected log delay the right point forecast for minutes?* No: expm1(E[log]) is closer to a median.
  It is optimal for RMSLE (the proposal's metric), which is why MAE in minutes is reported next to it.

**What the full run found** (`--full`, 42 min: 1.0M training rows / 53.8k runs, 1.0M test rows / 53.0k runs)

| Model (test, Dec 2025 to Feb 2026) | RMSLE | R2 (log1p) | MAE (min) | Skill vs persistence |
|---|---:|---:|---:|---:|
| Persistence | 0.952 | 0.599 | 8.30 | 0 |
| Persistence minus slack | 1.071 | 0.492 | 8.87 | -12.5% |
| Direct HGB (single stage) | **0.615** | **0.832** | 7.06 | **35.3%** |
| Hurdle logreg + HGB | 0.657 | 0.809 | **6.82** | 31.0% |
| **Hurdle RF + HGB (selected)** | 0.636 | 0.821 | 6.83 | 33.2% |
| Hurdle with perfect Stage 1 (bound) | 0.457 | 0.908 | 5.21 | 52.0% |

- Day-block bootstrap over 69 test days: persistence minus selected = +0.316 [+0.313, +0.320]; direct HGB minus
  selected = -0.020 [-0.022, -0.019]. So the single-stage HGB is reliably (slightly) better on RMSLE, while the
  hurdle is better in minutes (MAE 6.83 vs 7.06) and gives P(late).
- Stage 1: calibrated RF F1 **0.913** (precision 0.929, recall 0.898, ROC-AUC 0.962) vs persistence 0.887 and
  logistic regression 0.904. Calibration lowered log loss from 0.248 to 0.241 without hurting F1.
- Stage 2 on true-late rows: HGB R2 0.917 (RMSLE 0.224, MAE 7.4 min) vs persistence 0.178.
- The ranking of models in CV (direct HGB 0.620, hurdle RF+HGB 0.645) matched the test, so the CV protocol
  was a reliable guide.
- Trains already > 3 h late: MAE 24.4 min vs persistence 17.8 (RMSLE 0.324 vs 0.329). This is the limitation above.
- Feature-group importance (RMSLE increase): cascade 1.27, slack 0.13, station 0.10, route 0.06, congestion
  0.03, train 0.02, time 0.01, season 0.00.
- Going from 75k to 1M training rows took RMSLE from 0.701 to 0.636, so the models are data-hungry. More rows
  (beyond the RAM cap) would likely help further.

## Phase 4: Delay and route profiles (`src/trackcast/profiles.py`)

**What:** `python -m trackcast.profiles [--full]` turns the clean table into nine descriptive tables in
`data/processed/profiles[_sample]/` (stations, segments, train_stops, trains, by_month_zone, by_dow, by_hour,
by_date, cascade) and writes `reports/profiles.md` / `.json`. The Streamlit app reads them; the model does not.

**Why it is built this way**
- *Processed by train group, not by month.* A month split would cut a train's year into 12 pieces, so a
  per-(train, stop) median could not be computed exactly. Grouping trains (12 groups, pyarrow `in` filter)
  puts every run of a train in one group. Per-train and per-train-stop tables, medians and p90 included, are
  then exact, and each group fits in RAM.
- *Cross-train tables from partial sums.* Stations, segments, calendar and cascade are summed from per-group
  partials (counts, sums, flags). Station medians / p90 come from exact 1-minute histograms (8k stations x
  1,441 bins), not from averaging per-group medians, which would be wrong. A test checks that one pass and a
  split into groups give identical tables.
- *"Change" uses the previous observed stop*, the same position arithmetic as Phase 2's lag1 (it reuses
  `features.lag_positions`), so a missing stop never reads as recovery. A *segment* is only a pair of
  consecutive scheduled stops (stops_back = 1), so segment times are never blended across a gap.
- *Observed recovery, not just scheduled slack.* Phase 2's slack is timetable-only. Here each segment gets the
  observed recovery capacity: the mean change for trains arriving late at its start and the share that recover.
  This is the "minimum actual run time" idea from the proposal, kept descriptive. Fed into the model it would
  have to be fitted inside the CV folds, or it would leak test-period delays.
- *Profiles use all dates.* They describe the year and are never model inputs, so they cannot leak into a
  forecast. The trade-off: the app's "historic median" includes the test months.

**A data-quality finding: sawtooth stations.** Ranking segments by "delay gained" surfaced mirror pairs:
NDO→MDKU recovers 70 min and MDKU→TSR loses 85 min on the same trains. A train cannot recover 70 min in 4
scheduled minutes. The recorded delay at MDKU is systematically off, most likely because MDKU's timetable time
is wrong. Rule: a station is suspect when its mean change on arrival and on the next segment are both >= 15
min with opposite signs. 67 of 5,385 well-supported stations (mostly SCR halts) are flagged, and 264 ranked
segments touch them. They stay in the tables (flagged) but are excluded from the change rankings, as are
stations whose arrivals mostly come from a suspect station. This is a heuristic: a neighbour reached by
several routes can still inherit part of the error. The lesson is the same as Phase 1's join key: check an
extreme result against physical plausibility before reporting it.

**What the full run found** (36.5M stops, 6,908 trains, 8,142 stations, 29,502 segments, 4 min)
- *Network:* mean arrival delay 36.1 min, 50.6% of arrivals late > 15 min.
- *Delay cascades and rarely recovers* (the cascade table, 16.5M pairs at h = 1):

  | Initial delay | % still late 1 stop later | % still late 20 stops later | Mean change after 20 stops |
  |---|---:|---:|---:|
  | <= 15 min | 14.3 (became late) | 53.4 (became late) | +27 min |
  | 15 to 60 | 83.6 | 68.2 | +16 min |
  | 60 to 180 | 97.8 | 89.4 | +13 min |
  | > 180 | 99.7 | 99.0 | +35 min |

  Late trains do not catch up on average; they keep accumulating delay. Only about 40 to 49% of late trains are
  even slightly lower one stop later. This is why persistence is such a strong baseline and why cascade
  features dominate the Phase 3 importance.
- *Calendar:* December (45.3 min) and January (40.9) are the worst months, against 31 to 32 in August and
  September: the fog season, visible without any weather data. Night arrivals (00:00 to 03:00, 42 to 45 min)
  are far worse than early-morning ones (05:00 to 07:00, 27 to 29 min). That fits delay piling up overnight, but the data alone cannot say why. Day of week barely matters (35.6 to 36.5 min).
- *Train types:* Premium Humsafar trains (PRM) are the worst: they reach the terminus 133 min late on average,
  gaining 100 min en route. A plausible reason is that these specials get lower path priority than regular services; the data cannot show this directly. Vande Bharat (T18) is the
  best at 12 min.
- *Stations:* the worst mean delays cluster in SER (Chakradharpur division: MOU, SWR, GOL, CKP at 130 to 142
  min), and the CKP→TATA segment adds 80 min to late trains.

**Likely questions**
- *Why not compute medians with `groupby().median()` per month and average them?* The median of medians is not
  the median. Partials must be additive (sums, counts, histograms) to merge exactly.
- *Is the "recovery" table causal?* No. It is observed: a segment where late trains recover may have slack,
  or the timetable may pad it. That is why the scheduled minutes are shown next to it, and why suspected
  timetable errors are flagged.
