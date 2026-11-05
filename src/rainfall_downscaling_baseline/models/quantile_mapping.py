"""Quantile mapping: bias correction on top of bilinear interpolation.

For each fine grid cell and calendar month, map the distribution of the
interpolated training values onto the distribution of the observed (CHIRPS)
training values. It fixes how often and how much it rains, not where.
"""

import numpy as np
import xarray as xr

from .bilinear import BilinearDownscaler


def map_quantiles(x: np.ndarray, model_q: np.ndarray, obs_q: np.ndarray) -> np.ndarray:
    """Map values of one cell through matching quantiles (model_q -> obs_q).

    - Inside the training range: linear interpolation between quantiles. Drizzle
      that sits where CHIRPS quantiles are 0 maps to 0, which fixes ERA5's
      too-frequent light rain.
    - Above the largest training value: keep the correction of the top quantile
      (an added amount), instead of capping every new extreme at the training maximum.
    - 0 stays 0, and nothing comes out negative.
    """
    y = np.interp(x, model_q, obs_q)
    above = x > model_q[-1]
    y[above] = x[above] + (obs_q[-1] - model_q[-1])
    y[x <= 0] = 0.0
    return np.maximum(y, 0.0)


class QuantileMappingDownscaler:
    name = "quantile_mapping"

    def __init__(self, n_quantiles: int = 100) -> None:
        self.levels = np.linspace(0.0, 1.0, n_quantiles)
        self.interpolator = BilinearDownscaler()

    def fit(self, coarse: xr.Dataset, fine: xr.DataArray) -> None:
        """Store matching quantiles of interpolated and observed rain per land cell and month."""
        self.interpolator.fit(coarse, fine)
        interp = self.interpolator.predict(coarse)
        # Sea cells have no CHIRPS values to map onto: they keep the bilinear values.
        self.land = fine.notnull().all("time").values
        months = interp.time.dt.month.values
        self.model_q, self.obs_q = {}, {}
        for month in np.unique(months):
            days = months == month
            # Shape (quantiles, land cells): one column of quantiles per cell
            self.model_q[month] = np.quantile(
                interp.values[days][:, self.land], self.levels, axis=0
            )
            self.obs_q[month] = np.quantile(fine.values[days][:, self.land], self.levels, axis=0)

    def predict(self, coarse: xr.Dataset) -> xr.DataArray:
        """Interpolate, then map each land cell's values through its own month's quantiles."""
        interp = self.interpolator.predict(coarse)
        out = interp.values.copy()
        months = interp.time.dt.month.values
        for month in np.unique(months):
            if month not in self.model_q:
                raise ValueError(f"month {month} was not in the training data")
            days = months == month
            values = out[days][:, self.land]  # (days, land cells)
            model_q, obs_q = self.model_q[month], self.obs_q[month]
            for cell in range(values.shape[1]):
                values[:, cell] = map_quantiles(values[:, cell], model_q[:, cell], obs_q[:, cell])
            block = out[days]
            block[:, self.land] = values
            out[days] = block
        return interp.copy(data=out)
