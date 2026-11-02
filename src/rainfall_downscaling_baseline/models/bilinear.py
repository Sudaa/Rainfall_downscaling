"""Bilinear interpolation: the floor every other model must beat. Nothing is learned."""

import xarray as xr


class BilinearDownscaler:
    name = "bilinear"

    def fit(self, coarse: xr.DataArray, fine: xr.DataArray) -> None:
        # TODO: store fine.lat and fine.lon as the target grid
        raise NotImplementedError

    def predict(self, coarse: xr.DataArray) -> xr.DataArray:
        # TODO: interpolate coarse onto the stored grid (xarray's interp, method="linear").
        # What happens at the edges of the region?
        raise NotImplementedError
