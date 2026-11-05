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
