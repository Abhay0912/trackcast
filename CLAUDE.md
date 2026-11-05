# TrackCast — Forecasting Cascading Train Delays (DSF course project, NIT Warangal)

Spec: `26CSM1R01_Proposal_Report.pdf` (do not re-read unless asked; summary below).

## Goal
Predict station-level arrival delay using upstream lag, delay velocity, schedule slack, time and weather.
Pipeline: raw telemetry → cleaning → lag & slack features → two-stage hurdle ML → delay & route profiles → Streamlit app.

## Dataset
Kaggle `rxydenxd/indian-railways-delay-dataset` (MIT, ~1 GB, 1 year, ~7k trains). Files in data/raw/ (schemas verified):
- `combined_delay.csv` (1017 MB): `date` (str YYYY-MM-DD), `station_no` (int, stop sequence), `station_name` (station code), `delay` (float min, NaN = missing), `train_no` (int). Delay is given directly (no actual times). Read in chunks / pyarrow; never load whole in dev.
- `combined_schedule.csv` (5.5 MB): `train_no, station_no, station_name, distance_from_origin, arrival_day, arrival_time (HH:MM, NaN at origin), departure_day, departure_time (NaN at terminus)`.
- `train_details.csv`: `train_no, train_name, type_code` (e.g. PASS-TRAINS).
- `station_full_names.csv`: `station_name` (code, join key), `station_full_name, station_zone, station_address`.
Joins: delay ⨝ schedule on (train_no, station_no); ⨝ trains on train_no; ⨝ stations on station_name. Run id = (train_no, date).
No weather/congestion columns: weather = Open-Meteo join (or season flags: fog Dec–Feb north zones, monsoon Jun–Sep); congestion = trains scheduled through station per hour from schedule.

## Key definitions
- Delay_k = actual_arrival_k − scheduled_arrival_k, clipped at 0. Target for regression: log1p(Delay_k).
- Upstream lag L_{k-1} = Delay at previous stop of the same train run.
- Delay velocity ΔL_k = Delay_{k-1} − Delay_{k-2}.
- Schedule slack S_k = scheduled dwell/run time beyond minimum observed.
- Time features: sin/cos of hour, day-of-week, month. Weather flags (fog/monsoon/rain).
- Stage 1: classify Delay > 15 min (LogisticRegression, RandomForest) → F1.
- Stage 2: regress log1p delay (Ridge, HistGradientBoosting) → RMSLE, R².
- Station codes: sklearn TargetEncoder (fit inside CV folds only).

## Rules
- Never use information from stop k or later to build features for stop k (no leakage).
- Split train/test by date or by train run, never random rows.
- Develop on a ~100k-row sample (`data/processed/sample.parquet`), full data only for final runs.
- Do not open or print raw CSVs in data/; inspect with df.info()/head()/describe() only.
- Reusable logic in `src/trackcast/`; notebooks only call it and plot.
- After each phase, append design decisions + trade-offs to `docs/INTERVIEW_NOTES.md`.

## Layout
data/{raw,interim,processed}/ (gitignored) · notebooks/ · src/trackcast/ · app/ · docs/ · tests/

## Commands
- Env: `.venv\Scripts\activate` · Install: `pip install -r requirements.txt` then `pip install -e .`
- Data: `python -m trackcast.download` · Schema peek: `python -m trackcast.inspect_raw`
- App: `streamlit run app/streamlit_app.py` · Tests: `pytest -q`
