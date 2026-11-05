"""Tests for the quantile mapping baseline."""

import numpy as np
import pandas as pd
import xarray as xr

from rainfall_downscaling_baseline.models.quantile_mapping import (
    QuantileMappingDownscaler,
    map_quantiles,
)

LEVELS = np.array([0.0, 2.0, 10.0, 40.0])  # model quantiles of one cell


def test_identical_distributions_change_nothing():
    x = np.array([0.0, 1.0, 5.0, 30.0])
    assert np.allclose(map_quantiles(x, LEVELS, LEVELS), x)


def test_model_twice_too_wet_is_halved():
    x = np.array([2.0, 10.0, 25.0])
    assert np.allclose(map_quantiles(x, LEVELS, LEVELS / 2), x / 2)


def test_drizzle_maps_to_zero_where_observations_are_dry():
    # The model's lower quantiles are drizzle, the observed ones are 0
    obs_q = np.array([0.0, 0.0, 8.0, 40.0])
    assert map_quantiles(np.array([1.5]), LEVELS, obs_q)[0] == 0.0


def test_values_above_training_range_keep_top_correction():
    obs_q = np.array([0.0, 3.0, 15.0, 60.0])  # top quantile is corrected by +20
    assert map_quantiles(np.array([100.0]), LEVELS, obs_q)[0] == 120.0  # not capped at 60


def test_zero_stays_zero():
    obs_q = np.array([1.0, 3.0, 15.0, 60.0])
    assert map_quantiles(np.array([0.0]), LEVELS, obs_q)[0] == 0.0


def test_fit_and_predict_learn_a_doubling():
    # Coarse rain on a 2x2 grid; "observed" fine rain is exactly twice the
    # bilinear field, plus one sea cell. QM must learn the doubling per cell.
    rng = np.random.default_rng(0)
    days = pd.date_range("2022-05-01", periods=31)
    lat, lon = [10.0, 10.25], [100.0, 100.25]
    coarse = xr.DataArray(
        rng.gamma(0.8, 10.0, (31, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": days, "lat": lat, "lon": lon},
    )
    fine_lat, fine_lon = np.arange(10.025, 10.25, 0.05), np.arange(100.025, 100.25, 0.05)
    bilinear = coarse.interp(lat=fine_lat, lon=fine_lon)
    fine = 2 * bilinear
    fine[:, 0, 0] = np.nan  # a sea cell

    model = QuantileMappingDownscaler()
    model.fit(coarse.to_dataset(name="precip"), fine)
    pred = model.predict(coarse.to_dataset(name="precip"))

    land = fine.notnull()
    assert np.allclose(pred.where(land), fine, equal_nan=True)
    assert np.allclose(pred[:, 0, 0], bilinear[:, 0, 0])  # sea cell keeps bilinear
