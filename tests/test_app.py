"""Smoke tests: the Streamlit app runs every tab without exceptions (skipped without built artifacts)."""
import shutil

import pytest

from trackcast import config as C
from trackcast import forecast as FC
from trackcast import serving as S

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = C.ROOT / "app" / "streamlit_app.py"
ARTIFACTS = [C.MODEL_SAMPLE_PATH, C.PROFILES_SAMPLE_DIR / "cascade.parquet", C.RAW_FILES["schedule"]]
TABS = ["Overview", "How delays spread", "How the model works", "Results & bottlenecks", "Predict a train's delay"]
needs_demo = pytest.mark.skipif(S.demo_source(C.DEMO_DIR).missing() != [],
                                reason="needs the demo bundle: python -m trackcast.serving demo")


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="needs train/profiles outputs and raw data")
def test_app_renders_all_tabs():
    at = run_app()
    assert [t.label for t in at.tabs] == TABS
    at.number_input(key="fc_delay").set_value(120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("Next station") for m in at.metric)
    at.radio(key="res_by").set_value("Route (railway zones)").run()
    assert not at.exception, [e.value for e in at.exception]


@needs_demo
def test_demo_bundle_serves_alone(monkeypatch):
    """TRACKCAST_DEMO=1: only the bundle is offered, the forecast works without data/raw, and no provenance or
    version captions clutter the demo."""
    monkeypatch.setenv(S.DEMO_ENV, "1")
    monkeypatch.setenv(S.DEMO_DIR_ENV, str(C.DEMO_DIR))
    monkeypatch.setattr(C, "RAW_FILES", {k: C.ROOT / "nowhere" / v.name for k, v in C.RAW_FILES.items()})
    at = run_app()
    assert [t.label for t in at.tabs] == TABS
    assert len(at.sidebar.radio) == 0                  # one source: no data picker
    assert any(m.label.startswith("Next station") for m in at.metric)
    shown = " ".join(c.value for c in at.caption) + " ".join(w.value for w in at.sidebar.warning)
    assert "latest training run" not in shown and "Simulated on" not in shown and "library versions" not in shown
    overview = at.tabs[0]                              # dataset details, no model scores
    assert {"Station stops", "Train journeys", "Trains", "Stations"} <= {m.label for m in overview.metric}
    assert not any("RMSLE" in m.label or "F1" in m.label or "vs persistence" in m.label for m in overview.metric)


@needs_demo
def test_forecast_is_not_recomputed_by_other_widgets(monkeypatch):
    """Streamlit reruns the whole script on every widget change; the route forecast must come from cache."""
    monkeypatch.setenv(S.DEMO_ENV, "1")
    monkeypatch.setenv(S.DEMO_DIR_ENV, str(C.DEMO_DIR))
    calls = []
    real = FC.forecast_route
    monkeypatch.setattr(FC, "forecast_route", lambda *a, **k: calls.append(1) or real(*a, **k))
    at = run_app()
    n = len(calls)
    at.radio(key="res_by").set_value("Route (railway zones)").run()   # a widget in another tab: no recompute
    assert not at.exception and len(calls) == n
    at.number_input(key="fc_delay").set_value(95).run()          # a forecast input: computed once more
    assert len(calls) == n + 1


@needs_demo
def test_broken_model_is_reported_not_raised(tmp_path, monkeypatch):
    """A corrupt model pickle: an error in the sidebar and the Forecast tab, every other tab still renders."""
    bundle = tmp_path / "demo"
    shutil.copytree(C.DEMO_DIR, bundle)
    S.demo_source(bundle).model.write_bytes(b"truncated download")
    monkeypatch.setenv(S.DEMO_ENV, "1")
    monkeypatch.setenv(S.DEMO_DIR_ENV, str(bundle))
    at = run_app()
    assert [t.label for t in at.tabs] == TABS
    errors = " ".join(e.value for e in at.error)
    assert "Model not served" in errors and "cannot load" in errors
    assert not any(m.label.startswith("Next station") for m in at.metric)
    assert any(m.label == "Trains" for m in at.metric)
