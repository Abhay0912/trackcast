"""Tests for trackcast.forecast (route forecasts in the app) on a tiny synthetic timetable."""
import numpy as np
import pandas as pd
import pytest

from trackcast import data as D
from trackcast import features as F
from trackcast import forecast as FC
from test_features import SCHEDULE_RAW, ZONES


@pytest.fixture(scope="module")
def tt():
    sched = D.prepare_schedule(SCHEDULE_RAW, D.QualityStats())
    trains = pd.DataFrame({"train_no": pd.array([101, 202], dtype="int32"), "train_name": ["A EXP", "B EXP"],
                           "type_code": ["EXP-TRAINS", "SF-TRAINS"]})
    stations = pd.DataFrame({"station_name": list(ZONES), "station_full_name": [f"{s} JN" for s in ZONES],
                             "station_zone": list(ZONES.values())}).astype("string")
    return FC.Timetable.from_frames(sched, trains, stations)


class GainOneMinute:
    """Stub model: delay at k = upstream delay + 1, P(late) = 0.5."""
    def predict(self, df):
        nxt = df["lag1"].to_numpy() + 1.0      # both parts agree, so every strategy gives lag1 + 1
        return pd.DataFrame({"p_late": 0.5, "pred_delay_min": nxt, "m_late_log": np.log1p(nxt),
                             "m_low_log": np.log1p(nxt)}, index=df.index)


def test_stop_features_match_training_path(tt):
    f = FC.stop_features(tt, 101, "2025-01-06", {1: 5, 2: 10})
    r = f.set_index("station_no")
    assert r.loc[3, "lag1"] == 10 and r.loc[3, "lag2"] == 5 and r.loc[3, "slack_since_lag"] == 20
    assert list(f.columns) == F.OUTPUT_COLUMNS
    assert r.loc[3, "station_zone"] == "SCR" and r.loc[3, "type_code"] == "EXP-TRAINS"


def test_forecast_route_is_recursive(tt):
    out = FC.forecast_route(GainOneMinute(), tt, 101, "2025-01-06", {1: 5, 2: 10}).set_index("station_no")
    assert list(out["kind"]) == ["observed", "observed", "forecast", "forecast", "forecast"]
    assert list(out.loc[[3, 4, 5], "pred_delay_min"]) == pytest.approx([11, 12, 13])  # each feeds the next
    assert out.loc[3, "station_full_name"] == "CCC JN"
    assert np.isnan(out.loc[2, "p_late"])


def test_forecast_route_ignores_unprovided_and_limits(tt):
    out = FC.forecast_route(GainOneMinute(), tt, 101, "2025-01-06", {2: 30}, max_stops=1).set_index("station_no")
    assert out.loc[1, "kind"] == "not used" and out.loc[3, "pred_delay_min"] == pytest.approx(31)
    assert out.loc[4, "kind"] == "not used"
    with pytest.raises(ValueError):
        FC.forecast_route(GainOneMinute(), tt, 101, "2025-01-06", {})
    with pytest.raises(KeyError):
        tt.route(999)


def test_sched_clock():
    assert FC.sched_clock(10 * 60 + 5) == "10:05"
    assert FC.sched_clock(1440 + 30) == "00:30 (+1d)"
    assert FC.sched_clock(np.nan) == ""


def test_point_forecast_strategies():
    pred = pd.DataFrame({"p_late": [0.8, 0.2], "pred_delay_min": [20.0, 5.0],
                         "m_late_log": np.log1p([40.0, 40.0]), "m_low_log": np.log1p([4.0, 4.0])})
    assert list(FC.point_forecast(pred, "median", 0.5)) == [20.0, 5.0]
    assert np.allclose(FC.point_forecast(pred, "mean", 0.5), [0.8 * 40 + 0.2 * 4, 0.2 * 40 + 0.8 * 4])
    assert np.allclose(FC.point_forecast(pred, "gate", 0.5), [40.0, 4.0])
    with pytest.raises(ValueError):
        FC.point_forecast(pred, "mode", 0.5)


def test_simulate_routes_hides_future_delays(tt):
    """Simulation must only know the start delay: actual later delays never reach the features."""
    runs = [FC.run_frame(tt, 101, day, {1: 5, 2: 10, 3: 400, 4: 500, 5: 600})
            for day in ("2025-01-06", "2025-01-07", "2025-01-08")]
    clean = pd.concat(runs, ignore_index=True)
    out = FC.simulate_routes(GainOneMinute(), clean, tt.sched_feats, steps=3, strategy="median", seed=0)
    assert set(out["h"]) == {1, 2, 3} and out["run_id"].nunique() == 3
    # start is stop 2 (the only position with >= 3 stops after it and >= 1 before): 10 -> 11 -> 12 -> 13
    assert list(out.sort_values(["run_id", "h"])["pred"].iloc[:3]) == pytest.approx([11.0, 12.0, 13.0])
    assert (out["initial"] == 10).all() and set(out.loc[out["h"] == 1, "actual"]) == {400}
