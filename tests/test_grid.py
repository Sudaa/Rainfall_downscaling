"""Tests for data/grid.py, on small made-up grids shaped like ERA5 and CHIRPS."""

import numpy as np
import pytest
import xarray as xr

from rainfall_downscaling_baseline.config import Region
from rainfall_downscaling_baseline.data.grid import check_alignment, crop


def grid(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """A dataset with one variable of zeros on the given coordinates."""
    data = np.zeros((len(lat), len(lon)))
    return xr.Dataset({"x": (("lat", "lon"), data)}, coords={"lat": lat, "lon": lon})


# Like the real data: ERA5 points on multiples of 0.25 (north to south),
# CHIRPS cell centres at x.025, x.075, ... (south to north) inside them.
COARSE = grid(np.arange(21.0, 4.99, -0.25), np.arange(97.0, 106.01, 0.25))
FINE = grid(np.arange(5.025, 21.0, 0.05), np.arange(97.025, 106.0, 0.05))
REGION = Region(name="test", lat_min=10.0, lat_max=12.0, lon_min=100.0, lon_max=101.0)


@pytest.mark.parametrize("ds", [COARSE, FINE], ids=["north_to_south", "south_to_north"])
def test_crop_keeps_only_points_inside_region(ds):
    cropped = crop(ds, REGION)
    assert cropped.sizes["lat"] > 0 and cropped.sizes["lon"] > 0
    assert float(cropped.lat.min()) >= REGION.lat_min
    assert float(cropped.lat.max()) <= REGION.lat_max
    assert float(cropped.lon.min()) >= REGION.lon_min
    assert float(cropped.lon.max()) <= REGION.lon_max


def test_crop_outside_data_raises():
    far = Region(name="far", lat_min=40.0, lat_max=41.0, lon_min=100.0, lon_max=101.0)
    with pytest.raises(ValueError, match="no grid points"):
        crop(COARSE, far)


def test_alignment_passes_for_era5_and_chirps_like_grids():
    check_alignment(COARSE, FINE, factor=5)


def test_alignment_passes_with_float32_coordinates():
    # CHIRPS stores coordinates as float32, so they are off by about 1e-6
    fine32 = FINE.assign_coords(lat=FINE.lat.astype("float32"), lon=FINE.lon.astype("float32"))
    check_alignment(COARSE, fine32, factor=5)


def test_alignment_fails_for_wrong_factor():
    with pytest.raises(ValueError, match="not 4 x fine step"):
        check_alignment(COARSE, FINE, factor=4)


def test_alignment_fails_when_fine_grid_sticks_out():
    shifted = FINE.assign_coords(lat=FINE.lat - 0.05)  # first centre at 4.975, below 5.0
    with pytest.raises(ValueError, match="outside coarse grid"):
        check_alignment(COARSE, shifted, factor=5)


def test_alignment_fails_for_a_gap():
    with pytest.raises(ValueError, match="not regular"):
        check_alignment(COARSE, FINE.drop_isel(lat=10), factor=5)
