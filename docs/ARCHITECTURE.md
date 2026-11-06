# TrackCast architecture

How the system turns four raw CSVs into a forecasting app. This doc covers the data flow, what each module
does, which command builds which artifact, where leakage is prevented, how the model and the route forecast
fit together, and how the app finds and checks what it serves. Results are in [REPORT.md](REPORT.md). The
reasoning behind each design choice is in [INTERVIEW_NOTES.md](INTERVIEW_NOTES.md).

## 1. Data flow

```
 Kaggle rxydenxd/indian-railways-delay-dataset
        |  python -m trackcast.download
        v
 data/raw/  combined_delay.csv (1 GB)   combined_schedule.csv   train_details.csv   station_full_names.csv
        |                                      |  (timetable: also read by features, forecast, the app)
        |  python -m trackcast.data            |
        |  pass 1: stream 2M-row chunks -> data/interim/delay_by_month/*.parquet
        |  pass 2: one month at a time: join timetable / trains / stations, clean, sample whole runs
        v
 data/processed/clean.parquet (36.5M stops)    data/processed/sample.parquet (~100k rows, whole runs)
        |                                                  |
        |   every step below runs on the sample by default; --full runs it on clean.parquet
        |
        +--> features.py ---> features[_sample].parquet  (one row per predictable stop, leak-free)
        |                         |
        |                         v
        |                     train.py  (models.py + evaluation.py)
        |                         +--> models/hurdle[_sample].joblib
        |                         +--> data/processed/test_predictions[_sample].parquet
        |                         |
        +--> profiles.py --> data/processed/profiles[_sample]/  (9 descriptive tables)
        |                         |
        +--> eda.py <-------------+  --> data/processed/eda[_sample]/  + reports/figures/eda_*.png
        |                         |
        +--> results.py <---------+---- test_predictions, profiles, eda
        |                               --> data/processed/results[_sample]/  + reports/figures/results_*.png
        |
        +--> forecast.py <-- hurdle model + test-period runs  --> reports/route_forecast.md
                                                          
 serving.py demo <-- dev-sample model + profiles + results + timetable  --> demo/  (self-contained, ~19 MB)

 app/streamlit_app.py
     serving.py   finds a source (full year | dev sample | demo/), loads and checks the model, caches by fingerprint
     forecast.py  builds the user's run and calls features.build_features: the training feature code
     reads        models/, profiles/, results/, reports/models.json, reports/route_forecast.json, timetable
```

Every step also writes a report: `reports/<step>.md` for people and `reports/<step>.json` for the app and
tests. The proposal's Figure 1 (raw telemetry → cleaning → lag & slack features → two-stage hurdle ML → delay &
route profiles) maps onto `data.py` → `features.py` → `train.py` → `profiles.py`. `eda.py`, `results.py`,
`forecast.py`, `serving.py` and the app sit on top of those four steps.

## 2. Modules

All logic lives in `src/trackcast/`. The notebook ([notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb)) and the
app only call it and draw.

| Module | Role | Reads | Writes |
|---|---|---|---|
| `config.py` | Every path, threshold and constant (15-min hurdle, test cutoff, embargo, caps, season months, zones). No module hard-codes them. | – | – |
| `download.py` | Fetches the Kaggle dataset (needs `%USERPROFILE%\.kaggle\kaggle.json`). | Kaggle API | `data/raw/*.csv` |
| `inspect_raw.py` | Prints a schema summary of each raw table without loading the 1 GB file. | `data/raw/` | stdout |
| `data.py` | Phase 1. Streams the delay CSV in two passes. Joins the timetable on `(train_no, station_name)` and takes the timetable's `station_no` as stop order. Clips early arrivals to 0 (keeps `delay_raw`), sets delays > 24 h to NaN, keeps NaN as missing (never 0), drops runs with < 3 observed delays. Tests whether `date` is the run's start date and stops the build if it is not. Samples whole runs. | `data/raw/*.csv` | `data/interim/delay_by_month/` (deleted unless `--keep-interim`), `clean.parquet`, `sample.parquet`, `reports/data_quality.{md,json}` |
| `features.py` | Phase 2. One row per predictable stop: cascade (`lag1`, `lag2`, `delay_velocity`, staleness), slack (timetable only), route, time (sin/cos of the scheduled time), season (`fog_season`, `fog_night`, `monsoon`), congestion (scheduled trains per station-hour), raw categoricals. Targets `delay`, `y_late`, `y_log`. `--horizon H` for H-stop-ahead features. Exports `build_features` (used by the app) and `schedule_features`. | `sample.parquet` or `clean.parquet` (one month at a time), timetable CSV | `features[_sample].parquet`, `reports/features.{md,json}` |
| `models.py` | Phase 3 building blocks. Per-family preprocessing, `GroupedTargetEncoder` (target encoding with folds grouped by date), `PersistenceOffsetRegressor` (Stage 2 learns the residual over `log1p(lag1)`), isotonic calibration, `combine` (the hurdle formula), and `HurdleModel` (the saved object, with `.predict` and `.meta`). | – | – |
| `evaluation.py` | Temporal split with embargo, forward-chaining month folds, metrics (RMSLE, R², MAE, F1), the persistence baselines, day-block bootstrap, breakdowns, grouped permutation importance. | – | – |
| `train.py` | Phase 3 driver. Loads whole runs up to the row caps, runs CV over the families, picks the classifier × regressor pair, threshold and calibration on CV months, refits, scores the test period once. | `features[_sample].parquet` | `models/hurdle[_sample].joblib`, `data/processed/test_predictions[_sample].parquet`, `reports/models.{md,json}` |
| `profiles.py` | Phase 4. Descriptive tables over all dates: stations, segments, train_stops, trains, by_month_zone, by_dow, by_hour, by_date, cascade. Processes 12 groups of trains, so per-train medians are exact, and merges cross-train tables from additive partials and 1-minute histograms. Flags "sawtooth" stations (suspected timetable errors). | `sample.parquet` or `clean.parquet`, timetable | `data/processed/profiles[_sample]/*.parquet`, `reports/profiles.{md,json}` |
| `forecast.py` | Phase 5. `Timetable` (from the raw CSVs or the demo parquet), `forecast_route` (the recursive multi-stop forecast), `point_forecast` (median / mean / gate), and `python -m trackcast.forecast`, which simulates the app on real test-period runs. | model, `clean.parquet` (or `sample.parquet`), timetable, `profiles/cascade.parquet` | `reports/route_forecast.{md,json}` |
| `eda.py` | Phase 6. The proposal's exploratory analysis: distribution, trajectories, fog / monsoon, junction recovery, congestion, per-class and per-corridor profile metrics. Streams train groups into additive sum tables. Figures are matplotlib `Figure` objects shared by the script, the notebook and the tests. | `sample.parquet` or `clean.parquet`, Phase 4 profiles | `data/processed/eda[_sample]/`, `reports/figures/eda_*.png` (10), `reports/eda.{md,json}` |
| `results.py` | Phase 7. Test-set scores per train class, corridor, station group and train, taken from the saved predictions (no refit), with a shared day-block bootstrap. Through-station delay per station, and bottleneck / absorber rankings. | `test_predictions[_sample].parquet`, profiles, eda, `clean.parquet` | `data/processed/results[_sample]/` (5 tables), `reports/figures/results_*.png` (2), `reports/results.{md,json}` |
| `serving.py` | Phase 8. `Source` (one consistent set: model + profiles + results + timetable), `available_sources`, `fingerprint`, `load_model` / `check_model`, `report_notes`, `build_demo`. CLI: `check`, `demo`. | built artifacts | `demo/` (incl. `manifest.json`) |
| `app/streamlit_app.py` | Layout, caching and charts only, in plain language for a 10-minute class demo. Five tabs: Overview, Predict a train's delay, How delays spread, How the model works, Results & bottlenecks. The full tables stay in `reports/`. | a source from `serving.py`; `reports/data_quality.json`, `reports/profiles.json` (Overview: the full dataset, also in the demo) and `reports/models.json` | – |

## 3. Artifacts and the command that builds each

Run from the repo root with the venv active. A step without `--full` uses the ~100k-row dev sample and writes
the `_sample` variant. With `--full` it uses the whole year. Both variants write the same `reports/` files, so
the last run wins (the committed reports are from `--full` runs).

| Order | Command | Builds | Needs |
|---:|---|---|---|
| 0 | `python -m trackcast.download` | `data/raw/*.csv` | Kaggle token |
| 1 | `python -m trackcast.data` | `data/processed/clean.parquet`, `data/processed/sample.parquet`, `reports/data_quality.*` | raw CSVs |
| 2 | `python -m trackcast.features [--full]` | `data/processed/features[_sample].parquet`, `reports/features.*` | step 1, raw timetable |
| 3 | `python -m trackcast.train [--full]` | `models/hurdle[_sample].joblib`, `data/processed/test_predictions[_sample].parquet`, `reports/models.*` | step 2 |
| 4 | `python -m trackcast.profiles [--full]` | `data/processed/profiles[_sample]/` (9 tables), `reports/profiles.*` | step 1 |
| 5 | `python -m trackcast.forecast` | `reports/route_forecast.*` | step 3. Uses the full model and `clean.parquet` when both exist, else the sample. Step 4 is needed for the cascade-profile baseline. |
| 6 | `python -m trackcast.eda [--full]` | `data/processed/eda[_sample]/`, `reports/figures/eda_*.png`, `reports/eda.*` | step 4, same flag |
| 7 | `python -m trackcast.results [--full]` | `data/processed/results[_sample]/`, `reports/figures/results_*.png`, `reports/results.*` | steps 3, 4, 6, same flag |
| 8 | `python -m trackcast.serving demo` | `demo/` (model, profiles, results, timetable parquet, `manifest.json`) | steps 3, 4 (and 7) without `--full`, raw timetable |
| – | `python -m trackcast.serving check` | nothing (prints status) | – |
| – | `streamlit run app/streamlit_app.py` | nothing (serves) | a full, sample or demo source |
| – | `pytest -q` | nothing | tests that need built artifacts skip without them |

`data/`, `models/` and `demo/` are gitignored. `reports/` (Markdown, JSON, PNG) is committed.

## 4. Leakage rules and where they are enforced

The rule from the proposal: never use information from stop k or later to build features for stop k. The
system applies it at four levels.

| Rule | Where it is enforced | Test |
|---|---|---|
| **Features for stop k use only delays observed at stops ≤ k − H of the same run** (H = 1 by default). A missing stop is skipped, never read as 0. | `features.lag_positions` gathers every delay-derived feature from a row position ≤ k − H inside the run (`np.maximum.accumulate` cut at the run start). `build_features` uses the *scheduled* time at k, never the actual time. | `tests/test_features.py::test_no_leakage_from_stop_k_or_later`: randomly changing or blanking every delay at stops ≥ k must leave every feature of stop k bit-for-bit unchanged, for H = 1 and 2. Also `test_lags_skip_missing_and_respect_runs` and `test_horizon`. |
| **Slack and congestion come from the timetable only.** It is published before the train runs. | `features.schedule_features` reads no delay column. "Minimum observed" run time is the minimum *scheduled* time across trains. | `test_features.py::test_schedule_slack` |
| **Anything fitted on delays is fitted on training rows only.** | Target encoders, imputers and scalers sit inside the sklearn `Pipeline` (`models.preprocessor`), so every CV fold refits them. `GroupedTargetEncoder` groups its internal cross-fitting folds by run date: a row is never encoded from its own run's later stops or from same-day trains. Cross-run aggregates (profiles) are never model inputs. | `test_models.py::test_grouped_target_encoder_never_uses_same_date_targets`, `test_model_inputs_exclude_targets`, `test_predictions_do_not_depend_on_targets` |
| **Split by date, never by row.** Test = runs starting on or after `TEST_START_DATE` (2025-12-01). A 3-day embargo before every cutoff keeps multi-day runs from straddling it. | `evaluation.temporal_split`, `evaluation.month_folds`. Row caps in `train.py` sample whole runs by a hash of `run_id` (`run_uniform`). The dev sample is drawn by whole runs (`data.sample_runs`). | `test_models.py::test_temporal_split_with_embargo`, `test_month_folds_forward_chaining`, `test_run_uniform_keeps_whole_runs`; `test_data.py::test_sample_runs_whole_and_reproducible` |
| **The test set is scored once.** Hyper-parameters, the F1 threshold, the isotonic calibration map and the classifier × regressor pair are all chosen on the CV months. | `train.cross_validate` → `train.fit_final` → `train.evaluate_test` | – (protocol) |
| **The app and the route simulation use the training feature path and hide future delays.** | `forecast.empty_run` + `forecast.with_delays` build a run with only the delays the user entered (NaN elsewhere), and `forecast_route` calls `features.build_features` on it. `simulate_routes` keeps only the start stop's delay. | `test_forecast.py::test_stop_features_match_training_path`, `test_simulate_routes_hides_future_delays` |

Known soft spots, documented rather than hidden:
- Profiles, EDA tables and bottleneck rankings use all dates, test months included. They describe the year and
  never feed the model. The route evaluation's *cascade_profile* baseline uses them, so that baseline has seen
  the test months. `reports/route_forecast.md` says so.
- HistGradientBoosting's early stopping holds out random rows of the *training* data. Stops of one run can sit on
  both sides of that inner split, so early stopping may stop a little late. The test set is not involved.

## 5. The hurdle model and the recursive route forecast

**One stop ahead (what `train.py` fits and scores).**

```
 features of stop k  --->  Stage 1: RandomForest  --->  isotonic calibration  --->  p = P(delay_k > 15 min)
 (lag1, lag2, slack,                                                                    |
  time, season,      --->  Stage 2, late part:  HGB on log1p(delay) - log1p(lag1)  ---> m_late
  congestion,        --->  Stage 2, low part:   HGB on log1p(delay) - log1p(lag1)  ---> m_low
  encoded station)                                                                      |
                                       pred_log = p * m_late + (1 - p) * m_low   (E[log1p delay])
```

- Stage 1 and Stage 2 are separate pipelines in one `HurdleModel`. The classifier family (LogReg | RF) and the
  regressor family (Ridge | HGB) are chosen on CV. The test run selected RF + HGB (`reports/models.md`).
- `m_late` is fitted on rows with delay > 15 min, `m_low` on the rest. Both learn the change over
  persistence (`PersistenceOffsetRegressor`), so "the delay carries over" is built in.
- `pred_log` is the RMSLE-optimal point forecast. `expm1(pred_log)` is reported as `pred_delay_min`.
  `late_pred = p ≥ threshold`, with the threshold tuned for F1 on the CV months.
- `HurdleModel.meta` records the input columns, horizon, late threshold, training source and rows, and, for
  models trained after Phase 8, library versions and `trained_at`.

**Many stops ahead (what the app shows, `forecast.forecast_route`).** The model is a 1-stop model, so later stops
are forecast recursively:

```
 known: {station_no: delay} entered by the user
 for each later stop k on the route:
     run      = timetable rows of the train + known delays (NaN elsewhere)
     features = features.build_features(run, horizon=1)          # training code, row for stop k
     pred     = model.predict(features[k])                       # p, m_late, m_low
     delay_k  = p * expm1(m_late) + (1 - p) * expm1(m_low)       # ROUTE_FORECAST_STRATEGY = "mean"
     known[k] = delay_k                                          # becomes lag1 for stop k + 1
```

Feeding forward the one-stop RMSLE-optimal value (`median`) shrinks a late train towards 0 over 20 steps. The
expected minutes (`mean`) is the strategy chosen, because it had the lowest MAE from 2 stops ahead on the
simulated test runs (`reports/route_forecast.md`). The app shows those accuracy figures next to every
forecast. Uncertainty is not propagated through the recursion.

## 6. Serving: how the app finds, loads and checks artifacts

**Sources** (`serving.available_sources`). A source is one consistent set of model + profiles (+ results) +
timetable. The app offers, in order:
1. *Full year*: `models/hurdle.joblib` + `data/processed/profiles/` + `data/processed/results/`. The timetable
   comes from the raw CSVs.
2. *Dev sample*: `models/hurdle_sample.joblib` + `profiles_sample/` + `results_sample/`.
3. *Demo*: `demo/` (or `$TRACKCAST_DEMO_DIR`), used only when neither local source is complete, or always when
   `TRACKCAST_DEMO=1`. It holds the dev-sample model (lzma-compressed), its profiles and results, and the
   timetable as parquet (`schedule.parquet`, `sched_feats.parquet`), so it needs no raw CSVs.

A source is offered only if its required files exist. `serving check` lists the missing ones and the command
that builds them.

**Loading** (`serving.load_model`). It never raises: it returns a `LoadedModel` with the model or an error, plus
warnings. The sidebar shows an error, the Forecast tab is disabled, and the other tabs keep working. Warnings
(e.g. library versions) are not shown in the app, to keep the demo uncluttered. `serving check` prints them.

**Checks** (`serving.check_model`), run at every load:

| Check | Result if it fails |
|---|---|
| Object is a `HurdleModel` | error |
| Saved `input_columns` and every fitted pipeline's `feature_names_in_` equal `models.MODEL_INPUT_COLUMNS` (what `features.py` builds now) | error, naming added / removed columns |
| `horizon_stops == 1` (the recursion steps one stop at a time) | error |
| `significant_delay_min` equals `config.SIGNIFICANT_DELAY_MIN` | error |
| Stage 1 threshold is a probability | error |
| Library versions saved at training equal the running ones (and sklearn's own pickle-version warning) | warning |

After loading, `single_threaded` sets every `n_jobs` to 1, because one-row scoring is faster without a thread
pool.

**Caching.** Every loader in the app takes `serving.fingerprint(paths)`, the `(path, mtime_ns, size)` of every
file, as part of its cache key. A rebuilt artifact is reloaded and re-checked on the next interaction, with no
restart. Shared tables use `st.cache_resource` (one copy shared by all sessions; derive with `.copy()`, never
edit in place). Route forecasts are cached on their inputs plus the model and timetable fingerprints.

**Report metrics.** The Model tab shows `reports/models.json`, which describes the latest `--full`
training run. The demo forecasts with the dev-sample model but shows those full-year metrics without a caption
(a deliberate choice to keep the demo short). `serving.report_notes` can still build that caption; `serving check`
and the tests use it.

**Hosting.** `app/requirements.txt` pins only the 8 packages the app needs, at the exact versions the model was
pickled with. The app adds `src/` to `sys.path` when `trackcast` is not installed. Deploy steps are in the
[README](../README.md#demo-deploy-streamlit-community-cloud).

## 7. Tests

`pytest -q` runs one test file per module (`tests/test_<module>.py`). They use synthetic runs, except:
- `tests/test_app.py` drives every tab headless with Streamlit's `AppTest`, serves the demo bundle alone, and
  corrupts the bundled model to check that the app reports the error and does not crash;
- `tests/test_serving.py::test_saved_full_model_fits_the_code` checks the real `models/hurdle.joblib`.

Tests that need built artifacts skip when the artifacts are absent.
