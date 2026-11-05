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
```

## Layout
| Path | Purpose |
|---|---|
| `src/trackcast/` | Reusable pipeline code (config, data, features, models, evaluation) |
| `notebooks/` | EDA and modelling notebooks that call `src/` |
| `app/` | Streamlit dashboard |
| `data/{raw,interim,processed}` | Data (gitignored) |
| `models/` | Trained model artifacts (gitignored) |
| `reports/figures/` | Saved plots for the report |
| `docs/` | Architecture and interview notes |
| `tests/` | pytest suite |

Dataset: [Indian Railways – Delay Dataset](https://www.kaggle.com/datasets/rxydenxd/indian-railways-delay-dataset) (MIT).
