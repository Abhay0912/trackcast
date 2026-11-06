# TrackCast: Forecasting Cascading Train Delays and Schedule Recovery

DSF course project, NIT Warangal (Abhay Nagesh Rao, 26CSM1R01).

Predicts station-level arrival delays on Indian Railways from upstream lag, delay velocity, schedule slack,
time and seasonal context, using a two-stage hurdle model (classify significant delay → regress magnitude).

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m trackcast.download      # needs %USERPROFILE%\.kaggle\kaggle.json
python -m trackcast.inspect_raw   # schema summary
python -m trackcast.data          # clean + join -> data/processed/{clean,sample}.parquet, reports/data_quality.md
python -m trackcast.features      # dev sample -> features_sample.parquet, reports/features.md (--full for all data)
python -m trackcast.train         # tune + fit + test the hurdle model -> models/, reports/models.md (--full)
python -m trackcast.profiles      # station / segment / train / calendar / cascade profiles -> reports/profiles.md (--full)
python -m trackcast.forecast      # simulate multi-stop route forecasts on test runs -> reports/route_forecast.md
python -m trackcast.eda           # exploratory analysis + report figures -> reports/eda.md, reports/figures/ (--full; after profiles)
python -m trackcast.results       # results per train class / route + bottleneck junctions -> reports/results.md (--full; after train, profiles, eda)
streamlit run app/streamlit_app.py   # dashboard: overview, live route forecast, profiles, cascade, model, classes & bottlenecks
python -m trackcast.serving check    # what the app will find: missing files, model checks, load time, memory, forecast time
python -m trackcast.serving demo     # pack the dev-sample build into demo/ (~19 MB), servable without data/ or models/
```

The app offers the full-year artifacts (`--full` runs of train and profiles) and the dev sample when they exist,
else the demo bundle. It checks the saved model against the current feature code at load, shows any problem in the
sidebar instead of crashing, and reloads artifacts that are rebuilt while it runs.

## Demo deploy (Streamlit Community Cloud)
1. `python -m trackcast.serving demo`, then `git add -f demo` and push (the bundle is gitignored by default).
2. On share.streamlit.io: entrypoint `app/streamlit_app.py`, Python 3.13 under Advanced settings. The app's own
   `app/requirements.txt` (exact pins; the model is a pickle) is installed, not the dev `requirements.txt`.
3. With no `data/` or `models/` in the repo the app serves the bundle (`TRACKCAST_DEMO=1` forces this locally).
   Measured locally: ~16 s cold start, ~320 MB RSS, ~0.12 s per forecast stop.

## Layout
| Path | Purpose |
|---|---|
| `src/trackcast/` | Reusable pipeline code (config, data, features, models, evaluation) |
| `notebooks/` | Notebooks that call `src/` and plot (`01_eda.ipynb`: exploratory analysis) |
| `app/` | Streamlit dashboard (+ its slim `requirements.txt` for hosting) |
| `data/{raw,interim,processed}` | Data (gitignored) |
| `models/` | Trained model artifacts (gitignored) |
| `demo/` | Demo bundle: dev-sample model, profiles, results, timetable parquet (gitignored; built) |
| `reports/figures/` | Saved plots for the report |
| `docs/` | Architecture and interview notes |
| `tests/` | pytest suite |

Dataset: [Indian Railways – Delay Dataset](https://www.kaggle.com/datasets/rxydenxd/indian-railways-delay-dataset) (MIT).
