"""Reversible transforms applied before training and undone before scoring.

Rainfall is skewed with many zeros, so models train on log(1 + rain).
Normalisation statistics come from the training split only.
"""

from dataclasses import dataclass

import xarray as xr


def to_log(rain_mm: xr.DataArray) -> xr.DataArray:
    """mm/day -> log(1 + rain)."""
    raise NotImplementedError


def from_log(log_rain: xr.DataArray) -> xr.DataArray:
    """Inverse of to_log. Clip negatives to 0: a model can predict below zero."""
    raise NotImplementedError


@dataclass
class Normalizer:
    """Standardise a variable with a mean and standard deviation fitted on training data."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, train: xr.DataArray) -> "Normalizer":
        # TODO: why must this never see the test years?
        raise NotImplementedError

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError

    def inverse(self, x: xr.DataArray) -> xr.DataArray:
        raise NotImplementedError
