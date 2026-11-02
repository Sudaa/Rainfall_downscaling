"""Figures for the report. Each function saves one figure and returns its path."""

from pathlib import Path

import xarray as xr


def map_panel(
    coarse: xr.DataArray,
    predictions: dict[str, xr.DataArray],
    obs: xr.DataArray,
    day: str,
    out: Path,
) -> Path:
    """One row of maps for a single day: coarse input, each model, observed. Same colour scale."""
    raise NotImplementedError


def scatter(predictions: dict[str, xr.DataArray], obs: xr.DataArray, out: Path) -> Path:
    """Predicted vs observed rainfall, one panel per model."""
    raise NotImplementedError


def histogram(predictions: dict[str, xr.DataArray], obs: xr.DataArray, out: Path) -> Path:
    """Rainfall distributions on a log y-axis. Shows whether models miss the heavy tail."""
    raise NotImplementedError
