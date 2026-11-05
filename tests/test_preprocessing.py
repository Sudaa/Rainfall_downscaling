"""Tests for preprocessing/ and the bilinear baseline, on small made-up data."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainfall_downscaling_baseline.config import Period
from rainfall_downscaling_baseline.models.bilinear import BilinearDownscaler
from rainfall_downscaling_baseline.preprocessing.pairs import land_mask, match_days
from rainfall_downscaling_baseline.preprocessing.splits import split_by_year
from rainfall_downscaling_baseline.preprocessing.transforms import Normalizer, from_log, to_log

PERIOD = Period(train_years=[2022, 2023], val_years=[2024], test_years=[2025], months=[5])


def daily(days: pd.DatetimeIndex, values=None) -> xr.DataArray:
    """Rain on a 2x2 grid for the given days (1 mm everywhere unless values are given)."""
    data = np.ones((len(days), 2, 2)) if values is None else values
    coords = {"time": days, "lat": [10.0, 10.25], "lon": [100.0, 100.25]}
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords=coords, name="precip")


def test_log_round_trip():
    rain = daily(
        pd.date_range("2022-05-01", periods=3),
        np.array([0.0, 1.0, 250.0]).repeat(4).reshape(3, 2, 2),
    )
    assert np.allclose(from_log(to_log(rain)), rain)


def test_from_log_clips_negative_predictions():
    assert float(from_log(daily(pd.date_range("2022-05-01", periods=1)) * -1).min()) == 0.0


def test_normalizer_fitted_on_train_gives_mean_0_std_1_there():
    rng = np.random.default_rng(0)
    train = daily(pd.date_range("2022-05-01", periods=50), rng.gamma(1.0, 10.0, (50, 2, 2)))
    scaled = Normalizer().fit(train).transform(train)
    assert float(scaled.mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(scaled.std()) == pytest.approx(1.0)


def test_normalizer_inverse_undoes_transform():
    rain = daily(pd.date_range("2022-05-01", periods=4), np.arange(16.0).reshape(4, 2, 2))
    norm = Normalizer().fit(rain)
    assert np.allclose(norm.inverse(norm.transform(rain)), rain)


def test_split_by_year_has_no_shared_days():
    days = pd.DatetimeIndex([f"{y}-05-{d:02d}" for y in (2022, 2023, 2024, 2025) for d in (1, 2)])
    splits = split_by_year(daily(days), PERIOD)
    years = {name: set(s.time.dt.year.values) for name, s in splits.items()}
    assert years == {"train": {2022, 2023}, "val": {2024}, "test": {2025}}
    all_days = [t for s in splits.values() for t in s.time.values]
    assert len(all_days) == len(set(all_days)) == len(days)  # no day lost, none shared


def test_match_days_keeps_only_common_days():
    coarse = daily(pd.date_range("2022-10-29", periods=2))  # ERA5 lost 31 October
    fine = daily(pd.date_range("2022-10-29", periods=3))
    c, f = match_days(coarse, fine)
    assert list(c.time.values) == list(f.time.values) == list(coarse.time.values)


def test_land_mask_is_false_where_chirps_is_nan():
    values = np.ones((3, 2, 2))
    values[:, 0, 0] = np.nan  # a sea cell
    mask = land_mask(daily(pd.date_range("2022-05-01", periods=3), values))
    assert mask.values.tolist() == [[False, True], [True, True]]


def test_bilinear_reproduces_a_linear_field_exactly():
    # rain = lat + lon is linear, so bilinear interpolation must be exact
    lat, lon = np.arange(10.0, 11.01, 0.25), np.arange(100.0, 101.01, 0.25)
    coarse = xr.DataArray(
        (lat[:, None] + lon[None, :])[None],
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2022-05-01", periods=1), "lat": lat, "lon": lon},
    )
    fine_lat, fine_lon = np.arange(10.025, 11.0, 0.05), np.arange(100.025, 101.0, 0.05)
    fine = xr.DataArray(
        np.zeros((1, len(fine_lat), len(fine_lon))),
        dims=("time", "lat", "lon"),
        coords={"time": coarse.time, "lat": fine_lat, "lon": fine_lon},
    )
    model = BilinearDownscaler()
    model.fit(coarse, fine)
    pred = model.predict(coarse)
    assert pred.shape == fine.shape
    assert np.allclose(pred.isel(time=0), fine_lat[:, None] + fine_lon[None, :])
