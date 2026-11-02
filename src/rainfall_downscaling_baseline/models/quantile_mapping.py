"""Quantile mapping: bias correction on top of bilinear interpolation.

For each fine grid cell and calendar month, map the distribution of the
interpolated training values onto the distribution of the observed (CHIRPS)
training values. It fixes how often and how much it rains, not where.
"""

import xarray as xr

from .bilinear import BilinearDownscaler


class QuantileMappingDownscaler:
    name = "quantile_mapping"

    def __init__(self, n_quantiles: int = 100) -> None:
        self.n_quantiles = n_quantiles
        self.interpolator = BilinearDownscaler()

    def fit(self, coarse: xr.DataArray, fine: xr.DataArray) -> None:
        # TODO: fit the interpolator, interpolate the training inputs, then store
        # matching quantiles of interpolated and observed rain per cell and month.
        # How do you handle the many zero-rain days?
        raise NotImplementedError

    def predict(self, coarse: xr.DataArray) -> xr.DataArray:
        # TODO: interpolate, then map each value through its cell and month's quantiles
        raise NotImplementedError
