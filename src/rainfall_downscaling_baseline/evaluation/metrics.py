"""Verification metrics. Inputs are predicted and observed rainfall in mm/day, same shape.

Pure functions with no plotting or file access, so each one can be tested
against a small hand-computed example.
"""

import numpy as np

# --- Continuous scores: how close are the amounts? ---


def mae(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean absolute error."""
    raise NotImplementedError


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root mean squared error. Punishes large misses more than MAE does."""
    raise NotImplementedError


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean of pred - obs. Positive means the model is too wet."""
    raise NotImplementedError


def correlation(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pearson correlation over all points."""
    raise NotImplementedError


# --- Categorical scores: does it get heavy rain events right? ---


def contingency(pred: np.ndarray, obs: np.ndarray, threshold: float) -> dict[str, int]:
    """Count hits, misses, false_alarms and correct_negatives for rain >= threshold."""
    raise NotImplementedError


def hit_rate(table: dict[str, int]) -> float:
    """hits / (hits + misses): share of observed events the model caught."""
    raise NotImplementedError


def false_alarm_ratio(table: dict[str, int]) -> float:
    """false_alarms / (hits + false_alarms): share of predicted events that did not happen.

    Not the same as the false alarm *rate*, false_alarms / (false_alarms +
    correct_negatives). Say which one you report.
    """
    raise NotImplementedError


def csi(table: dict[str, int]) -> float:
    """Critical success index: hits / (hits + misses + false_alarms)."""
    raise NotImplementedError
