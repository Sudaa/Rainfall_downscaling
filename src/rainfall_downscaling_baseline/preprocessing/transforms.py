"""Reversible transforms applied before training and undone before scoring.

Rainfall is skewed with many zeros, so models train on log(1 + rain).
Normalisation statistics come from the training split only.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr


def to_log(rain_mm: xr.DataArray) -> xr.DataArray:
    """mm/day -> log(1 + rain). 0 stays 0, and 100 mm becomes about 4.6."""
    return np.log1p(rain_mm)


def from_log(log_rain: xr.DataArray) -> xr.DataArray:
    """Inverse of to_log. Clip negatives to 0: a model can predict below zero."""
    return np.expm1(log_rain).clip(min=0)


@dataclass
class Normalizer:
    """Standardise a variable with a mean and standard deviation fitted on training data."""

    mean: float = 0.0
    std: float = 1.0

    def fit(self, train: xr.DataArray) -> "Normalizer":
        # Only the training years: statistics from validation or test years
        # would leak information about the weather the model is scored on.
        self.mean = float(train.mean())
        self.std = float(train.std())
        return self

    def transform(self, x: xr.DataArray) -> xr.DataArray:
        return (x - self.mean) / self.std

    def inverse(self, x: xr.DataArray) -> xr.DataArray:
        return x * self.std + self.mean
