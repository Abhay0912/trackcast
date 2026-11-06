"""Smoke test: the Streamlit app runs every tab without exceptions (skipped without built artifacts)."""
import pytest

from trackcast import config as C

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = C.ROOT / "app" / "streamlit_app.py"
ARTIFACTS = [C.MODEL_SAMPLE_PATH, C.PROFILES_SAMPLE_DIR / "cascade.parquet", C.RAW_FILES["schedule"]]


@pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="needs train/profiles outputs and raw data")
def test_app_renders_all_tabs():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Forecast a train", "Route profile", "Stations & segments",
                                         "Delay cascade", "Model",
                                         "Classes & bottlenecks"]
    at.number_input(key="fc_delay").set_value(120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.label.startswith("Next stop") for m in at.metric)
    for choice in ("Corridor", "Station group", "Train"):
        at.radio(key="res_by").set_value(choice).run()
        assert not at.exception, [e.value for e in at.exception]
    at.radio(key="bn_side").set_value("Recovers delay (absorbers)").run()
    assert not at.exception, [e.value for e in at.exception]
