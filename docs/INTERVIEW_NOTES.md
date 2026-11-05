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
