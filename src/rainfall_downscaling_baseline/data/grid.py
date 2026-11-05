"""Grid operations shared by both datasets: cropping and alignment checks.

Both functions expect dimensions named `lat` and `lon` (the loaders rename them).
"""

import numpy as np
import xarray as xr

from ..config import Region


def crop(ds: xr.Dataset, region: Region) -> xr.Dataset:
    """Keep only the points inside the region.

    `sel` with a slice follows the order of the coordinate: ERA5 latitude runs
    north to south, CHIRPS south to north. A slice in the wrong order returns
    an empty dataset without any error, so the order is chosen from the data.
    Both datasets use longitudes from -180 to 180, which covers Thailand directly.
    """
    lat = ds["lat"]
    if lat[0] > lat[-1]:
        lat_slice = slice(region.lat_max, region.lat_min)
    else:
        lat_slice = slice(region.lat_min, region.lat_max)
    cropped = ds.sel(lat=lat_slice, lon=slice(region.lon_min, region.lon_max))
    if cropped.sizes["lat"] == 0 or cropped.sizes["lon"] == 0:
        raise ValueError(f"crop to {region} left no grid points")
    return cropped


def check_alignment(coarse: xr.Dataset, fine: xr.Dataset, factor: int, tol: float = 1e-4) -> None:
    """Raise ValueError unless the fine grid can be interpolated from the coarse one.

    Both datasets give cell centres, but the grids are offset: ERA5 cells are
    centred on multiples of 0.25 deg, so their edges (e.g. 13.875) cut CHIRPS
    cells (edges on multiples of 0.05) in half. Coarse cells are therefore not
    exact blocks of factor x factor fine cells. The models do not need that:
    they interpolate the coarse field to the fine centres and work on the fine
    grid. What they need, and what this checks, is that

    - both grids are regular (no gaps),
    - the coarse spacing is `factor` times the fine spacing,
    - every fine centre lies inside the coarse points, so interpolation never
      extrapolates.

    `tol` (degrees) absorbs rounding: CHIRPS stores coordinates as float32.
    """
    for dim in ("lat", "lon"):
        c = np.sort(coarse[dim].values.astype(float))
        f = np.sort(fine[dim].values.astype(float))
        c_step, f_step = np.diff(c), np.diff(f)
        if np.ptp(c_step) > tol or np.ptp(f_step) > tol:
            raise ValueError(f"{dim}: grid spacing is not regular")
        if abs(c_step[0] - factor * f_step[0]) > tol:
            raise ValueError(
                f"{dim}: coarse step {c_step[0]:g} is not {factor} x fine step {f_step[0]:g}"
            )
        if f[0] < c[0] - tol or f[-1] > c[-1] + tol:
            raise ValueError(
                f"{dim}: fine grid {f[0]:g}..{f[-1]:g} is outside coarse grid {c[0]:g}..{c[-1]:g}"
            )
