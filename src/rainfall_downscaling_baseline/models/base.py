"""The interface every downscaling model follows.

Because all models share `fit` and `predict`, the evaluation code can score any
of them without knowing which one it is. Adding a new model later (diffusion,
VLM-conditioned) means adding one file here, not changing the evaluation.
"""

from typing import Protocol

import xarray as xr


class Downscaler(Protocol):
    name: str

    def fit(self, coarse: xr.Dataset, fine: xr.DataArray) -> None:
        """Learn from training pairs: daily ERA5 (all variables, rain in mm/day) and
        CHIRPS rain (mm/day). Each model uses the variables it needs. Also remember
        the fine grid."""
        ...

    def predict(self, coarse: xr.Dataset) -> xr.DataArray:
        """Return rainfall in mm/day on the fine grid seen in fit."""
        ...
