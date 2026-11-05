"""Grid operations shared by both datasets: cropping and alignment checks.

Both functions expect dimensions named `lat` and `lon` (the loaders rename them).
"""

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


def check_alignment(coarse: xr.Dataset, fine: xr.Dataset, factor: int) -> None:
    """Raise ValueError unless each coarse cell is covered by exactly factor x factor fine cells.

    TODO:
    - Are the coordinates cell centres or cell edges in each dataset?
      Print the first few latitudes of both and work out where each cell's
      edges are. This decides whether the 5 to 1 ratio really holds.
    """
    raise NotImplementedError
