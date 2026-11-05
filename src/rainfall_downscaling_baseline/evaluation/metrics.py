"""Verification metrics. Inputs are predicted and observed rainfall in mm/day, same shape.

Pure functions with no plotting or file access, so each one can be tested
against a small hand-computed example. Inputs must already be free of NaN
(evaluate.py keeps land cells only).
"""

import numpy as np

# --- Continuous scores: how close are the amounts? ---


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(pred - obs)))


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root mean squared error. Punishes large misses more than MAE does."""
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean of pred - obs. Positive means the model is too wet."""
    return float(np.mean(pred - obs))


def correlation(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation over all points."""
    return float(np.corrcoef(pred.ravel(), obs.ravel())[0, 1])


# --- Categorical scores: does it get heavy rain events right? ---


def contingency(pred: np.ndarray, obs: np.ndarray, threshold: float) -> dict[str, int]:
    """Count hits, misses, false_alarms and correct_negatives for rain >= threshold."""
    p, o = pred >= threshold, obs >= threshold
    return {
        "hits": int(np.sum(p & o)),
        "misses": int(np.sum(~p & o)),
        "false_alarms": int(np.sum(p & ~o)),
        "correct_negatives": int(np.sum(~p & ~o)),
    }


def _ratio(numerator: int, denominator: int) -> float:
    """NaN instead of a division by zero, e.g. when no event was observed at all."""
    return numerator / denominator if denominator else float("nan")


def hit_rate(table: dict[str, int]) -> float:
    """hits / (hits + misses): share of observed events the model caught."""
    return _ratio(table["hits"], table["hits"] + table["misses"])


def false_alarm_ratio(table: dict[str, int]) -> float:
    """false_alarms / (hits + false_alarms): share of predicted events that did not happen.

    Not the same as the false alarm *rate*, false_alarms / (false_alarms +
    correct_negatives). This project reports the ratio.
    """
    return _ratio(table["false_alarms"], table["hits"] + table["false_alarms"])


def csi(table: dict[str, int]) -> float:
    """Critical success index: hits / (hits + misses + false_alarms)."""
    return _ratio(table["hits"], table["hits"] + table["misses"] + table["false_alarms"])
