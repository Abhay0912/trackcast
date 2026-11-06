"""Tests for trackcast.serving (Phase 8): finding, checking and loading artifacts, and the demo bundle.

Uses the small synthetic model from test_models and the toy timetable from test_forecast, never the real data."""
import dataclasses
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from trackcast import config as C
from trackcast import models as M
from trackcast import profiles as P
from trackcast import serving as S
from test_forecast import tt  # noqa: F401  (fixture)
from test_models import trained  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def model(trained):  # noqa: F811
    return trained[4]


def test_trained_model_passes_checks(model):
    errors, warns = S.check_model(model)
    assert errors == [] and warns == []
    assert model.meta["input_columns"] == M.MODEL_INPUT_COLUMNS


@pytest.mark.parametrize("meta, needle", [
    ({"horizon_stops": 2}, "1-stop model"),
    ({"significant_delay_min": 30}, "late > 30 min"),
    ({"input_columns": M.MODEL_INPUT_COLUMNS[:-2]}, "other features"),
])
def test_check_model_rejects_mismatched_meta(model, meta, needle):
    bad = dataclasses.replace(model, meta={**model.meta, **meta})
    errors, _ = S.check_model(bad)
    assert errors and needle in " ".join(errors)


def test_check_model_catches_feature_drift_in_fitted_pipelines(model, monkeypatch):
    """A feature added to the code after training: the saved list and the fitted pipelines both disagree."""
    monkeypatch.setattr(M, "MODEL_INPUT_COLUMNS", [*M.MODEL_INPUT_COLUMNS, "new_feature"])
    errors, _ = S.check_model(dataclasses.replace(model, meta={k: v for k, v in model.meta.items()
                                                               if k != "input_columns"}))
    assert errors and "new_feature" in errors[0]


def test_check_model_warns_on_version_mismatch_and_wrong_type(model):
    old = dataclasses.replace(model, meta={**model.meta, "versions": {"scikit-learn": "0.1"}})
    errors, warns = S.check_model(old)
    assert errors == [] and "scikit-learn 0.1" in warns[0]
    assert S.check_model({"not": "a model"})[0]


def test_load_model_serves_single_threaded_and_never_raises(model, tmp_path):
    good = tmp_path / "m.joblib"
    joblib.dump(model, good)
    lm = S.load_model(good)
    assert lm.error is None and lm.model is not None and lm.load_s > 0
    assert all(v == 1 for k, v in lm.model.classifier.get_params().items() if k.endswith("n_jobs"))

    (tmp_path / "corrupt.joblib").write_bytes(b"not a pickle")
    joblib.dump({"a": 1}, tmp_path / "dict.joblib")
    for name, needle in (("corrupt.joblib", "cannot load"), ("dict.joblib", "HurdleModel"),
                         ("missing.joblib", "not found")):
        lm = S.load_model(tmp_path / name)
        assert lm.model is None and needle in lm.error


def test_fingerprint_changes_when_an_artifact_is_rebuilt(tmp_path):
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "a.parquet").write_bytes(b"x")
    fp = S.fingerprint(d)
    st = (d / "a.parquet").stat()
    os.utime(d / "a.parquet", ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
    assert S.fingerprint(d) != fp
    (d / "b.parquet").write_bytes(b"y")
    assert len(S.fingerprint(d)) == 2
    assert S.fingerprint(tmp_path / "missing")[0][1] is None


def _fake_source(tmp_path, model, tt):  # noqa: F811
    """A complete built source on disk from the synthetic model and the toy timetable."""
    root = tmp_path / "built"
    src = S.Source("Synthetic", root / "model.joblib", root / "profiles", root / "results", root / "timetable",
                   False, "n/a")
    src.profiles.mkdir(parents=True)
    joblib.dump(model, src.model)
    for t in P.TABLES:
        df = pd.DataFrame({"train_no": [101], "n_runs": [3]}) if t == "trains" else pd.DataFrame({"x": [1]})
        df.to_parquet(src.profiles / f"{t}.parquet")
    tt.save(src.timetable)
    return src


def test_build_demo_bundle_is_self_contained(tmp_path, model, tt):  # noqa: F811
    src = _fake_source(tmp_path, model, tt)
    out = tmp_path / "demo"
    manifest = S.build_demo(out, src)
    demo = S.demo_source(out)
    assert demo.missing() == [] and not demo.results_ready()          # results are optional
    assert manifest["model"]["classifier"] == model.meta["classifier"]
    assert set(manifest["files"]) >= {"timetable/schedule.parquet", "profiles/trains.parquet"}
    assert S.load_model(demo.model).error is None
    bundled = demo.load_timetable()
    assert set(bundled.schedule["train_no"]) == {101}                 # only profiled trains are shipped
    pd.testing.assert_frame_equal(bundled.route_sched_feats(101).reset_index(drop=True),
                                  tt.route_sched_feats(101).reset_index(drop=True))
    S.build_demo(out, src)                                            # rebuilding over a bundle is fine
    with pytest.raises(FileExistsError):                              # ... over anything else is refused
        (tmp_path / "other").mkdir()
        S.build_demo(tmp_path / "other", src)
    with pytest.raises(FileNotFoundError):
        S.build_demo(tmp_path / "x", dataclasses.replace(src, model=tmp_path / "nope.joblib"))


def test_available_sources_falls_back_to_demo(tmp_path, model, tt, monkeypatch):  # noqa: F811
    out = tmp_path / "demo"
    S.build_demo(out, _fake_source(tmp_path, model, tt))
    not_built = S.Source("Nothing", tmp_path / "no.joblib", tmp_path / "np", tmp_path / "nr", None, False, "n/a")
    monkeypatch.setattr(S, "local_sources", lambda: [not_built])
    env = {S.DEMO_DIR_ENV: str(out)}
    assert [s.name for s in S.available_sources(env)] == ["Demo (dev sample)"]
    assert S.available_sources({S.DEMO_DIR_ENV: str(tmp_path / "empty")}) == []
    monkeypatch.setattr(S, "local_sources", lambda: [S.demo_source(out)])   # stands in for a ready local build
    assert len(S.available_sources(env)) == 1
    assert S.available_sources({**env, S.DEMO_ENV: "1"})[0].model.parent.parent == out


def test_report_notes_flag_metrics_from_another_model():
    meta = {"source": "features_sample.parquet", "train_rows": 75_045, "trained_at": "2026-11-06T00:06:00"}
    models_full = {"meta": {"source": "features.parquet", "train_rows": 999_274}}
    route = {"meta": {"model": "hurdle.joblib"}}
    notes = S.report_notes(meta, "hurdle_sample.joblib", models_full, route)
    assert "features.parquet" in notes["models"] and "999,274" in notes["models"] and "hurdle.joblib" in notes["route"]
    stale = {"meta": {**meta, "trained_at": "2026-11-01T00:00:00"}}
    assert "another training run" in S.report_notes(meta, "m", stale, None)["models"]
    same = {"meta": dict(meta)}
    assert S.report_notes(meta, "hurdle_sample.joblib", same, {"meta": {"model": "hurdle_sample.joblib"}}) == {}
    assert S.report_notes({}, "m", None, None) == {}


def test_timetable_save_load_round_trip(tt, tmp_path):  # noqa: F811
    tt.save(tmp_path)
    back = type(tt).load(tmp_path)
    pd.testing.assert_frame_equal(back.schedule, tt.schedule)
    pd.testing.assert_frame_equal(back.sched_feats, tt.sched_feats)


@pytest.mark.skipif(not C.MODEL_PATH.exists(), reason="needs the full-year model")
def test_saved_full_model_fits_the_code():
    """The real 62 MB model against the current feature code (what the app would serve)."""
    lm = S.load_model(C.MODEL_PATH)
    assert lm.error is None, lm.error
    assert json.loads(json.dumps(lm.meta, default=str))["horizon_stops"] == 1
    assert np.isclose(lm.model.threshold, lm.meta["threshold"])
