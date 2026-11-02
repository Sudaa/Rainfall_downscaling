"""Time-based train/test split.

A random split of days would leak information: neighbouring days are strongly
correlated, so the model would be tested on weather it has nearly seen.
"""

import xarray as xr

from ..config import Period


def split_by_year(ds: xr.Dataset, period: Period) -> tuple[xr.Dataset, xr.Dataset]:
    """Return (train, test) using period.train_years and period.test_years."""
    raise NotImplementedError
