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
