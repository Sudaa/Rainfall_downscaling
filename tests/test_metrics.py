"""Tests for evaluation/, on tiny arrays scored by hand first.

pred = [0, 12, 30], obs = [5, 8, 30]
errors pred - obs = [-5, 4, 0]
  MAE  = (5 + 4 + 0) / 3 = 3
  RMSE = sqrt((25 + 16 + 0) / 3) = sqrt(41 / 3)
  bias = (-5 + 4 + 0) / 3 = -1/3
At 10 mm: day 1 neither (correct negative), day 2 predicted only (false
alarm), day 3 both (hit). No misses.
"""

import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainfall_downscaling_baseline.config import Region
from rainfall_downscaling_baseline.evaluation import metrics
from rainfall_downscaling_baseline.evaluation.evaluate import evaluate

PRED = np.array([0.0, 12.0, 30.0])
OBS = np.array([5.0, 8.0, 30.0])


def test_continuous_scores_match_hand_calculation():
    assert metrics.mae(PRED, OBS) == pytest.approx(3.0)
    assert metrics.rmse(PRED, OBS) == pytest.approx(math.sqrt(41 / 3))
    assert metrics.bias(PRED, OBS) == pytest.approx(-1 / 3)


def test_correlation_of_x_with_itself_is_1():
    assert metrics.correlation(OBS, OBS) == pytest.approx(1.0)


def test_contingency_at_10mm():
    table = metrics.contingency(PRED, OBS, 10.0)
    assert table == {"hits": 1, "misses": 0, "false_alarms": 1, "correct_negatives": 1}
    assert metrics.hit_rate(table) == 1.0
    assert metrics.false_alarm_ratio(table) == 0.5
    assert metrics.csi(table) == 0.5


def test_threshold_counts_as_event():
    # rain exactly at the threshold is an event (>=), on both sides
    table = metrics.contingency(np.array([20.0]), np.array([20.0]), 20.0)
    assert table["hits"] == 1


def test_csi_is_1_for_perfect_and_0_when_nothing_is_hit():
    assert metrics.csi(metrics.contingency(OBS, OBS, 10.0)) == 1.0
    assert metrics.csi(metrics.contingency(np.zeros(3), OBS, 10.0)) == 0.0


def test_scores_are_nan_when_no_event_happens():
    table = metrics.contingency(np.zeros(3), np.zeros(3), 50.0)
    assert math.isnan(metrics.hit_rate(table))
    assert math.isnan(metrics.csi(table))


def _field(values: np.ndarray) -> xr.DataArray:
    coords = {
        "time": pd.date_range("2025-05-01", periods=values.shape[0]),
        "lat": [13.6, 13.8, 20.0],
        "lon": [100.5, 100.7],
    }
    return xr.DataArray(values, dims=("time", "lat", "lon"), coords=coords)


def test_evaluate_scores_land_only_and_each_region():
    obs = _field(10.0 + np.arange(12.0).reshape(2, 3, 2))  # varies, so correlation is defined
    obs[:, 2, 1] = np.nan  # a sea cell
    land = obs.notnull().all("time")
    pred = obs.fillna(999.0) + 1.0  # 1 mm too wet on land, nonsense over the sea
    bangkok = Region(name="bangkok", lat_min=13.5, lat_max=14.0, lon_min=100.3, lon_max=100.9)

    table = evaluate({"model": pred}, obs, land, regions={"bangkok": bangkok})

    assert list(table.index) == [("all", "model"), ("bangkok", "model")]
    assert table.loc[("all", "model"), "bias"] == pytest.approx(1.0)  # the sea cell is ignored
    assert table.loc[("bangkok", "model"), "mae"] == pytest.approx(1.0)


def test_evaluate_refuses_predictions_on_another_grid():
    obs = _field(np.zeros((2, 3, 2)))
    shifted = obs.assign_coords(lat=obs.lat + 0.05)
    with pytest.raises(ValueError):
        evaluate({"model": shifted}, obs, obs.notnull().all("time"))
