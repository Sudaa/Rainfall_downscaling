"""Score every model's predictions into one results table."""

import pandas as pd
import xarray as xr


def evaluate(
    predictions: dict[str, xr.DataArray],
    obs: xr.DataArray,
    thresholds: tuple[float, ...] = (10.0, 25.0),
) -> pd.DataFrame:
    """One row per model, one column per metric (continuous ones, then each threshold).

    TODO: loop over models, call the functions in metrics.py, collect a DataFrame.
    Skip NaN cells (ocean, edges) consistently for every model.
    """
    raise NotImplementedError
