# TrackCast: Forecasting Cascading Train Delays and Schedule Recovery

DSF course project, NIT Warangal (Abhay Nagesh Rao, 26CSM1R01).

TrackCast predicts how late an Indian Railways train will be at each of its next stops. Its inputs are the
delays the train has already picked up, the slack in the timetable ahead, and the time, season and station
traffic. It learns from one year of station-level delay records (36.5M scheduled stops, 6,908 trains, 8,142
stations). It uses a two-stage *hurdle* model: first *will the train be more than 15 min late?*, then *by
how much?*. It also profiles where delays build up and where they are recovered, and ranks the junctions that
add the most delay to the network. A Streamlit app forecasts any train's remaining route live.

## Headline results

Test set: every run starting 2025-12-01 to 2026-02-07 (996,417 stops), never seen in training or tuning.
*Persistence* = "the delay at the previous stop carries over", the naive forecast every model must beat.

| | Persistence | TrackCast hurdle (RF + HGB) |
|---|---:|---:|
| RMSLE (log1p delay) | 0.9517 | **0.6355** (33.2% lower, 95% CI [33.0, 33.4]) |
| R² (log1p delay) | 0.599 | **0.821** |
| Mean absolute error | 8.30 min | **6.83 min** |
| Stage 1 F1, late > 15 min | 0.887 | **0.913** (ROC-AUC 0.962) |

- **Every train class and corridor gains.** Skill over persistence is 31.3–38.2% per train class and
  24.6–39.8% per corridor, and every interval is above 0 ([results.md](reports/results.md)).
- **Multi-stop forecasts.** 10 stops ahead, the app's recursive forecast has an MAE of 22.8 min vs 26.6 for
  persistence ([route_forecast.md](reports/route_forecast.md)).
- **Delays cascade.** 20 stops after a train is 1–3 h late, 89% of such trains are still late
  ([eda.md](reports/eda.md)).
- **Bottlenecks.** Kanpur Central (CNB, ~2,945 delay-minutes added per day), Kharagpur (KGP) and Tatanagar
  (TATA) top the bottleneck ranking ([results.md](reports/results.md)).
- **Honest caveats.** A single-stage HGB is slightly better on RMSLE (0.6153). The hurdle is kept for its
  P(late) and its lower MAE. Trains already > 3 h late are still forecast better by persistence in minutes
  ([models.md](reports/models.md)).

The full write-up is [docs/REPORT.md](docs/REPORT.md).

## Quickstart

Windows / PowerShell, Python ≥ 3.11 (developed on 3.13), run from the repo root.

**1. Set up**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

**2. Get the data** (needs a Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`)
```powershell
python -m trackcast.download      # ~1 GB into data/raw/
python -m trackcast.inspect_raw   # schema summary, without loading the big file
```

**3. Run the pipeline.** Each step runs on the ~100k-row dev sample by default; add `--full` for the whole year
(the committed reports are from `--full` runs). Run the steps in this order. Each writes `reports/<step>.md` and
`.json`.
```powershell
python -m trackcast.data              # clean + join -> data/processed/{clean,sample}.parquet
python -m trackcast.features --full   # leak-free features -> data/processed/features.parquet
python -m trackcast.train --full      # tune, fit, test the hurdle model -> models/hurdle.joblib (~40 min)
python -m trackcast.profiles --full   # delay and route profiles -> data/processed/profiles/
python -m trackcast.forecast          # simulate multi-stop forecasts on test runs -> reports/route_forecast.md
python -m trackcast.eda --full        # exploratory analysis -> reports/figures/eda_*.png (after profiles)
python -m trackcast.results --full    # results per class / route, bottlenecks (after train, profiles, eda)
```
For the app's dev-sample source and the demo bundle, also run `features`, `train`, `profiles`, `eda` and
`results` without `--full`.

**4. Run the app**
```powershell
streamlit run app/streamlit_app.py    # 5 tabs: overview, predict a delay, how delays spread, how the model works, results
python -m trackcast.serving check     # what the app will find: missing files, model checks, load time, memory
pytest -q                             # tests (those needing built artifacts skip without them)
```

**5. Build the demo bundle**
```powershell
python -m trackcast.serving demo      # dev-sample model + profiles + results + timetable -> demo/ (~19 MB)
$env:TRACKCAST_DEMO = "1"; streamlit run app/streamlit_app.py   # serve only the bundle, as a host would
```

## Demo deploy (Streamlit Community Cloud)
1. `python -m trackcast.serving demo`, then `git add -f demo` and push (the bundle is gitignored by default).
2. On share.streamlit.io: entrypoint `app/streamlit_app.py`, Python 3.13 under Advanced settings. The host
   installs the app's own `app/requirements.txt` (exact pins, because the model is a pickle), not the dev
   `requirements.txt`.
3. With no `data/` or `models/` in the repo, the app serves the bundle (`TRACKCAST_DEMO=1` forces this locally).
   Measured locally: ~16 s cold start, ~320 MB RSS, ~0.12 s per forecast stop
   (Phase 8 in [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md)).

## Repo layout

| Path | Contents |
|---|---|
| `src/trackcast/` | All pipeline logic: `config` (paths, thresholds), `download`, `inspect_raw`, `data`, `features`, `models`, `evaluation`, `train`, `profiles`, `forecast`, `eda`, `results`, `serving` |
| `app/` | `streamlit_app.py` (layout and charts only) and its slim, pinned `requirements.txt` for hosting |
| `notebooks/01_eda.ipynb` | Exploratory analysis: loads the EDA tables and calls `trackcast.eda` plot functions |
| `tests/` | pytest suite, one file per module, including headless app tests |
| `reports/` | Committed reports (`*.md` to read, `*.json` for the app) and `figures/` |
| `docs/` | [REPORT.md](docs/REPORT.md), [ARCHITECTURE.md](docs/ARCHITECTURE.md), [INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) |
| `data/{raw,interim,processed}/` | Raw CSVs, interim monthly files, processed parquet (gitignored, rebuilt by the pipeline) |
| `models/` | Trained models `hurdle.joblib` / `hurdle_sample.joblib` (gitignored) |
| `demo/` | Self-contained demo bundle (gitignored, built by `serving demo`) |

## Where to find each result

| Question | Report |
|---|---|
| What was cleaned, and how much data is left? | [reports/data_quality.md](reports/data_quality.md) |
| Which features, persistence baseline, does slack absorb delay? | [reports/features.md](reports/features.md) |
| Model scores, CV, Stage 1 / Stage 2, feature importance | [reports/models.md](reports/models.md) |
| Stations, segments, trains, calendar, how far delays cascade | [reports/profiles.md](reports/profiles.md) |
| Accuracy of the app's multi-stop forecast | [reports/route_forecast.md](reports/route_forecast.md) |
| Distribution, trajectories, fog / monsoon, junction recovery, congestion | [reports/eda.md](reports/eda.md) + [reports/figures/](reports/figures/) |
| Results per train class / route / train; bottleneck junctions | [reports/results.md](reports/results.md) |
| The whole story in one document | [docs/REPORT.md](docs/REPORT.md) |
| How the system fits together | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Why each design choice was made | [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md) |

Dataset: [Indian Railways – Delay Dataset](https://www.kaggle.com/datasets/rxydenxd/indian-railways-delay-dataset) (MIT).
