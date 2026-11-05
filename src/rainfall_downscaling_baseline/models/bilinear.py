"""Bilinear interpolation: the floor every other model must beat. Nothing is learned."""

import xarray as xr


class BilinearDownscaler:
    name = "bilinear"

    def fit(self, coarse: xr.Dataset, fine: xr.DataArray) -> None:
        """Remember the fine grid. That is all: bilinear has nothing to learn."""
        self.lat = fine.lat
        self.lon = fine.lon

    def interpolate[T: (xr.Dataset, xr.DataArray)](self, coarse: T) -> T:
        """Interpolate any coarse fields linearly in lat and lon to the fine cell centres.

        Every fine centre lies inside the coarse points (checked by
        check_alignment), so there is no extrapolation and no NaN at the edges.
        The result covers the sea too; scores are computed over land only.
        """
        return coarse.interp(lat=self.lat, lon=self.lon, method="linear")

    def predict(self, coarse: xr.Dataset) -> xr.DataArray:
        return self.interpolate(coarse["precip"])
