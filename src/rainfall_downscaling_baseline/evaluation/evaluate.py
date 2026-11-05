"""Score every model's predictions into one results table."""

import pandas as pd
import xarray as xr

from ..config import Region
from ..data.grid import crop
from . import metrics

THRESHOLDS = (1.0, 20.0, 50.0)  # mm/day: rain or no rain, heavy, very heavy


def _land_values(pred: xr.DataArray, obs: xr.DataArray, land: xr.DataArray):
    """Flat numpy arrays of the land cells, the same cells for every model.

    join="exact" fails loudly if a prediction is on a different grid or
    different days than the observations, instead of silently aligning them.
    """
    pred, obs, land = xr.align(pred, obs, land, join="exact")
    keep = (land & obs.notnull()).broadcast_like(obs).values
    return pred.values[keep], obs.values[keep]


def score(pred: xr.DataArray, obs: xr.DataArray, land: xr.DataArray, thresholds=THRESHOLDS):
    """All metrics for one prediction, as a dict of metric name -> value."""
    p, o = _land_values(pred, obs, land)
    row = {
        "mae": metrics.mae(p, o),
        "rmse": metrics.rmse(p, o),
        "bias": metrics.bias(p, o),
        "corr": metrics.correlation(p, o),
    }
    for t in thresholds:
        table = metrics.contingency(p, o, t)
        row[f"hit_rate_{t:g}mm"] = metrics.hit_rate(table)
        row[f"far_{t:g}mm"] = metrics.false_alarm_ratio(table)
        row[f"csi_{t:g}mm"] = metrics.csi(table)
    return row


def evaluate(
    predictions: dict[str, xr.DataArray],
    obs: xr.DataArray,
    land: xr.DataArray,
    regions: dict[str, Region] | None = None,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> pd.DataFrame:
    """One row per (region, model), one column per metric.

    "all" is the whole configured region; every entry of `regions` (such as
    Bangkok) is scored again on its own crop. Only land cells are scored.
    """
    areas = {"all": None} | (regions or {})
    rows = []
    for area, region in areas.items():
        for name, pred in predictions.items():
            if region is None:
                p, o, m = pred, obs, land
            else:
                p, o, m = crop(pred, region), crop(obs, region), crop(land, region)
            rows.append({"region": area, "model": name} | score(p, o, m, thresholds))
    return pd.DataFrame(rows).set_index(["region", "model"])
