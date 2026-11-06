"""Phase 8: serve saved artifacts. Find them, load them once, and check they fit the code that uses them.

Usage:  python -m trackcast.serving check           artifact status, model checks, load time, memory, forecast time
        python -m trackcast.serving demo [--out D]   pack the dev-sample build into a demo bundle (default demo/)

A *source* is one consistent set of artifacts: model + profiles (+ results) + a timetable. The app offers the
full-year and dev-sample sources when they are built locally. It falls back to the demo bundle when neither is
(e.g. Streamlit Community Cloud, which has no data/ or models/), and uses only the bundle with TRACKCAST_DEMO=1.

What can go wrong when serving a pickled model, and the check for each:
  * a file is missing                    -> the source is not offered; its build command is shown
  * the pickle is from other library versions -> load errors are caught; versions saved at training are compared
  * the model was trained on other features -> saved input_columns and the pipelines' feature_names_in_ vs the code
  * the model answers another question   -> horizon_stops must be 1 (the route forecast recurses one stop at a
                                            time) and significant_delay_min must match the app's "late" threshold
  * a running server caches an old file  -> cache keys include every file's (mtime, size): rebuilt = reloaded
  * a report describes another model     -> reports/models.json and route_forecast.json compared with model meta
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import time
import warnings as pywarnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from trackcast import config as C
from trackcast import forecast as FC
from trackcast import models as M
from trackcast import profiles as P

DEMO_ENV, DEMO_DIR_ENV = "TRACKCAST_DEMO", "TRACKCAST_DEMO_DIR"
MANIFEST = "manifest.json"


# --------------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    name: str
    model: Path
    profiles: Path
    results: Path
    timetable: Path | None   # saved timetable directory (demo bundle); None = build it from the raw CSVs
    full: bool               # full-year data: selects the larger minimum-support thresholds in the app
    build: str               # command(s) that build this source

    def required(self) -> list[Path]:
        """Files without which the source is not offered (results, reports and raw CSVs are optional)."""
        paths = [self.model, *(self.profiles / f"{t}.parquet" for t in P.TABLES)]
        return paths + ([self.timetable / f for f in FC.TIMETABLE_FILES] if self.timetable is not None else [])

    def missing(self) -> list[Path]:
        return [p for p in self.required() if not p.exists()]

    def timetable_files(self) -> list[Path]:
        if self.timetable is not None:
            return [self.timetable / f for f in FC.TIMETABLE_FILES]
        return [C.RAW_FILES[k] for k in ("schedule", "trains", "stations")]

    def load_timetable(self) -> FC.Timetable | None:
        if not all(p.exists() for p in self.timetable_files()):
            return None
        return FC.Timetable.load(self.timetable)

    def results_ready(self) -> bool:
        return all((self.results / f"{t}.parquet").exists() for t in C.RESULTS_TABLES)


def local_sources() -> list[Source]:
    return [Source("Full year", C.MODEL_PATH, C.PROFILES_DIR, C.RESULTS_DIR, None, True,
                   "python -m trackcast.train --full; python -m trackcast.profiles --full"),
            Source("Dev sample", C.MODEL_SAMPLE_PATH, C.PROFILES_SAMPLE_DIR, C.RESULTS_SAMPLE_DIR, None, False,
                   "python -m trackcast.train; python -m trackcast.profiles")]


def demo_dir(env=None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(DEMO_DIR_ENV) or C.DEMO_DIR)


def demo_source(directory=None) -> Source:
    d = Path(directory) if directory is not None else demo_dir()
    return Source("Demo (dev sample)", d / "model" / C.MODEL_SAMPLE_PATH.name, d / "profiles", d / "results",
                  d / "timetable", False, "python -m trackcast.serving demo")


def available_sources(env=None) -> list[Source]:
    """Ready sources, in the order the app offers them."""
    env = os.environ if env is None else env
    demo = demo_source(demo_dir(env))
    if env.get(DEMO_ENV, "") not in ("", "0"):
        return [] if demo.missing() else [demo]
    ready = [s for s in local_sources() if not s.missing()]
    return ready or ([] if demo.missing() else [demo])


def fingerprint(*paths) -> tuple:
    """(path, mtime_ns, size) of every file under `paths`. It changes whenever an artifact is rebuilt, so as part
    of a cache key it makes a running app reload rebuilt artifacts instead of serving the old ones until restart."""
    out = []
    for p in map(Path, paths):
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            try:
                st = f.stat()
                out.append((str(f), st.st_mtime_ns, st.st_size))
            except FileNotFoundError:
                out.append((str(f), None, None))
    return tuple(out)


def load_tables(directory, tables) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(Path(directory) / f"{t}.parquet") for t in tables}


def load_json(path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------
def library_versions() -> dict:
    return M.environment_meta()["versions"]


def _column_diff(what: str, got, expected) -> str:
    got, expected = list(got), list(expected)
    added, removed = [c for c in expected if c not in got], [c for c in got if c not in expected]
    detail = (f"the code now builds {added} and no longer builds {removed}" if added or removed
              else "same columns, different order")
    return (f"{what}: the model was trained on other features than features.py builds ({detail}); rebuild "
            "features and retrain (python -m trackcast.features; python -m trackcast.train)")


def check_model(model) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a loaded model against this code. An error means forecasts would fail or be wrong."""
    if not isinstance(model, M.HurdleModel):
        return [f"expected a trackcast HurdleModel, got {type(model).__name__}"], []
    errors, warns = [], []
    meta, expected = model.meta or {}, list(M.MODEL_INPUT_COLUMNS)
    saved = meta.get("input_columns")
    if saved is None:
        warns.append("the model's metadata has no input_columns; only the fitted pipelines were checked")
    elif list(saved) != expected:
        errors.append(_column_diff("saved metadata", saved, expected))
    for name, est in (("classifier", model.classifier), ("reg_late", getattr(model.reg_late, "estimator_", None)),
                      ("reg_low", getattr(model.reg_low, "estimator_", None))):
        names = getattr(est, "feature_names_in_", None)
        if names is None:
            errors.append(f"{name} is not a fitted pipeline")
        elif list(names) != expected and not errors:
            errors.append(_column_diff(f"{name} pipeline", names, expected))
    h = meta.get("horizon_stops", 1)
    if h != 1:
        errors.append(f"the model forecasts {h} stops ahead, but the route forecast feeds each forecast back one "
                      "stop at a time and needs a 1-stop model (features and train with --horizon 1)")
    sig = meta.get("significant_delay_min", C.SIGNIFICANT_DELAY_MIN)
    if sig != C.SIGNIFICANT_DELAY_MIN:
        errors.append(f"stage 1 was trained for late > {sig} min, but the app reports late > "
                      f"{C.SIGNIFICANT_DELAY_MIN} min (config.SIGNIFICANT_DELAY_MIN); retrain")
    if not 0 < float(getattr(model, "threshold", np.nan)) < 1:
        errors.append(f"stage 1 threshold {getattr(model, 'threshold', None)!r} is not a probability")
    # scikit-learn stamps every pickled estimator with its version and warns on a mismatch at load time
    # (InconsistentVersionWarning, surfaced by load_model), so models saved before Phase 8 without "versions"
    # are still checked for it; the saved versions add numpy and pandas.
    running, saved_v = library_versions(), meta.get("versions")
    if saved_v is not None:
        diff = [f"{k} {v} (running {running.get(k)})" for k, v in saved_v.items() if running.get(k) != v]
        if diff:
            warns.append("trained with " + ", ".join(diff) + ". A pickle is only guaranteed to load and predict "
                         "the same under the versions it was saved with: pin them or retrain")
    return errors, warns


def single_threaded(model: M.HurdleModel) -> M.HurdleModel:
    """n_jobs = 1 everywhere. The app scores one row per call; starting a thread pool for the forest's 200 trees
    costs more than the trees (measured 91 -> 35 ms per row), and on a shared server one user's forecast should
    not take every core. Batch scoring (train.py) keeps n_jobs = -1."""
    for est in (model.classifier, getattr(model.reg_late, "estimator_", None),
                getattr(model.reg_low, "estimator_", None)):
        if est is not None:
            est.set_params(**{k: 1 for k in est.get_params(deep=True) if k == "n_jobs" or k.endswith("__n_jobs")})
    return model


@dataclass
class LoadedModel:
    path: Path
    model: M.HurdleModel | None        # None when the model cannot be served (see error)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    load_s: float = 0.0

    @property
    def meta(self) -> dict:
        return dict(self.model.meta) if self.model is not None else {}


def load_model(path) -> LoadedModel:
    """Load, check and prepare a model for serving. Never raises: problems come back in .error / .warnings, so
    the app can show them and keep the tabs that do not need the model."""
    path, t0 = Path(path), time.perf_counter()
    if not path.exists():
        return LoadedModel(path, None, f"{path.name} not found")
    with pywarnings.catch_warnings(record=True) as caught:
        pywarnings.simplefilter("always")
        try:
            model = joblib.load(path)
        except Exception as e:  # noqa: BLE001  (unpickling can fail in many ways; all mean "cannot serve")
            return LoadedModel(path, None, f"cannot load {path.name} ({type(e).__name__}: {e}). It was probably "
                               f"saved under other library versions; retrain it in this environment "
                               f"(scikit-learn {library_versions()['scikit-learn']})")
    errors, warns = check_model(model)
    warns += sorted({str(w.message).split("\n")[0] for w in caught if issubclass(w.category, UserWarning)})
    if errors:
        return LoadedModel(path, None, "; ".join(errors), warns)
    return LoadedModel(path, single_threaded(model), None, warns, time.perf_counter() - t0)


def report_notes(model_meta: dict, model_name: str, models_report: dict | None,
                 route_report: dict | None) -> dict[str, str]:
    """Captions for metrics that were not computed with the model being served ('models', 'route')."""
    notes = {}
    rm = (models_report or {}).get("meta", {})
    if rm and model_meta:
        if rm.get("source") != model_meta.get("source"):
            notes["models"] = (f"Metrics are from the latest training run, the model trained on `{rm.get('source')}` "
                               f"({_n(rm.get('train_rows'))} rows); forecasts here come from the "
                               f"`{model_meta.get('source')}` model ({_n(model_meta.get('train_rows'))} rows).")
        elif (rm.get("train_rows") != model_meta.get("train_rows")
              or rm.get("trained_at", model_meta.get("trained_at")) != model_meta.get("trained_at")):
            notes["models"] = ("reports/models.json is from another training run than the model served: rerun "
                               "`python -m trackcast.train` to refresh it.")
    rr = (route_report or {}).get("meta", {})
    if rr.get("model") and rr["model"] != model_name:
        notes["route"] = f"Route-forecast accuracy was simulated with `{rr['model']}`, not the model served here."
    elif rr.get("model_trained_at") and model_meta.get("trained_at") and \
            rr["model_trained_at"] != model_meta["trained_at"]:
        notes["route"] = "Route-forecast accuracy was simulated with an earlier version of this model."
    return notes


def _n(v) -> str:
    return f"{v:,}" if isinstance(v, (int, np.integer)) else "?"


# --------------------------------------------------------------------------------------------------
# Demo bundle
# --------------------------------------------------------------------------------------------------
def build_demo(out=None, source: Source | None = None) -> dict:
    """Pack a built source (default: the dev sample) into a self-contained bundle the app can serve alone:
    model (lzma-recompressed), profiles, results, and the timetable as parquet (only the profiled trains; slack
    is computed over the whole timetable first). The raw CSVs are needed once, here, not at serving time."""
    src = source or local_sources()[1]
    out = Path(out) if out is not None else demo_dir()
    if src.missing():
        raise FileNotFoundError(f"{src.name} is not built ({', '.join(p.name for p in src.missing())}): {src.build}")
    if out.exists() and not (out / MANIFEST).exists():
        raise FileExistsError(f"{out} exists and is not a demo bundle; refusing to overwrite it")
    model = joblib.load(src.model)
    errors, warns = check_model(model)
    if errors:
        raise ValueError(f"{src.model.name} cannot be served: {'; '.join(errors)}")
    tt = src.load_timetable()
    if tt is None:
        raise FileNotFoundError("the timetable CSVs in data/raw are needed to build the bundle: "
                                "python -m trackcast.download")

    tmp = out.with_name(out.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    dst = demo_source(tmp)
    dst.model.parent.mkdir(parents=True)
    joblib.dump(model, dst.model, compress=C.DEMO_MODEL_COMPRESS)
    for d_from, d_to, tables in ((src.profiles, dst.profiles, P.TABLES), (src.results, dst.results, C.RESULTS_TABLES)):
        if d_from == src.results and not src.results_ready():
            continue
        d_to.mkdir(parents=True)
        for t in tables:
            shutil.copy2(d_from / f"{t}.parquet", d_to / f"{t}.parquet")
    trains = pd.read_parquet(src.profiles / "trains.parquet", columns=["train_no"])["train_no"]
    tt.save(dst.timetable, train_nos=set(trains.astype("int64")))

    files = {str(f.relative_to(tmp)).replace("\\", "/"): f.stat().st_size for f in sorted(tmp.rglob("*"))
             if f.is_file()}
    manifest = {"built_at": dt.datetime.now().isoformat(timespec="seconds"), "from": src.name,
                "model": {k: model.meta.get(k) for k in ("source", "train_rows", "train_dates", "trained_at",
                                                         "versions", "classifier", "regressor")},
                "model_warnings": warns, "versions": library_versions(),
                "total_mb": round(sum(files.values()) / 2 ** 20, 1), "files": files}
    (tmp / MANIFEST).write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return manifest


# --------------------------------------------------------------------------------------------------
# check: what the app will find, and what loading it costs
# --------------------------------------------------------------------------------------------------
def _rss_mb() -> float:
    import gc
    gc.collect()   # so a delta measures what was loaded, not garbage freed (or not yet freed) meanwhile
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except ImportError:
        return float("nan")


def _mb(p: Path) -> float:
    files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    return sum(f.stat().st_size for f in files) / 2 ** 20


def check() -> list[dict]:
    """Print and return, per source: missing files, model checks, load times, memory and one route forecast."""
    rows = []
    print(f"Python env: {library_versions()}  (RSS at start {_rss_mb():.0f} MB)")
    for s in [*local_sources(), demo_source()]:
        row = {"source": s.name, "missing": [str(p) for p in s.missing()]}
        print(f"\n{s.name}: " + ("ready" if not row["missing"] else
                                f"not ready, {len(row['missing'])} files missing (build: {s.build})"))
        if not s.model.exists():
            rows.append(row)
            continue
        rss = _rss_mb()
        lm = load_model(s.model)
        row.update(model_mb=round(_mb(s.model), 1), model_load_s=round(lm.load_s, 2),
                   model_rss_mb=round(_rss_mb() - rss), error=lm.error, warnings=lm.warnings)
        print(f"  model {row['model_mb']} MB on disk: load {lm.load_s:.1f} s, +{row['model_rss_mb']} MB RSS")
        for w in lm.warnings:
            print(f"  warning: {w}")
        if lm.error:
            print(f"  ERROR: {lm.error}")
        tt = pr = None
        t0, rss = time.perf_counter(), _rss_mb()
        tt = s.load_timetable()
        if tt is not None:
            row.update(timetable_s=round(time.perf_counter() - t0, 2), timetable_rss_mb=round(_rss_mb() - rss))
            print(f"  timetable: load {row['timetable_s']} s, +{row['timetable_rss_mb']} MB RSS")
        if s.profiles.exists() and not row["missing"]:
            t0, rss = time.perf_counter(), _rss_mb()
            pr = load_tables(s.profiles, P.TABLES)
            row.update(profiles_s=round(time.perf_counter() - t0, 2), profiles_rss_mb=round(_rss_mb() - rss))
            print(f"  profiles {_mb(s.profiles):.1f} MB on disk: load {row['profiles_s']} s, "
                  f"+{row['profiles_rss_mb']} MB RSS")
            if lm.model is not None and tt is not None:
                trains = pr["trains"][pr["trains"]["train_no"].isin(tt.schedule["train_no"])]
                train_no = int(trains.sort_values("n_runs").iloc[-1]["train_no"])
                start = int(tt.route(train_no)["station_no"].iloc[min(2, len(tt.route(train_no)) - 2)])
                t0 = time.perf_counter()
                out = FC.forecast_route(lm.model, tt, train_no, "2026-01-10", {start: 30.0})
                n = int((out["kind"] == "forecast").sum())
                row.update(forecast_s=round(time.perf_counter() - t0, 2), forecast_stops=n)
                print(f"  route forecast, train {train_no}: {n} stops in {row['forecast_s']} s "
                      f"({1000 * row['forecast_s'] / max(n, 1):.0f} ms per stop)")
        print(f"  RSS now {_rss_mb():.0f} MB")
        rows.append(row)
        lm = tt = pr = None  # noqa: F841  (release this source before measuring the next)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="artifact status, model checks, load time and memory per source")
    d = sub.add_parser("demo", help="pack the dev-sample build into a demo bundle")
    d.add_argument("--out", type=Path, default=None, help=f"bundle directory (default {C.DEMO_DIR.name}/)")
    args = ap.parse_args()
    if args.cmd == "check":
        check()
    else:
        m = build_demo(args.out)
        print(f"Demo bundle: {demo_dir() if args.out is None else args.out}  ({m['total_mb']} MB, "
              f"{len(m['files'])} files, model trained on {m['model']['source']})")
        for w in m["model_warnings"]:
            print(f"  warning: {w}")


if __name__ == "__main__":
    main()
