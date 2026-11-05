"""Time-based train/validation/test split.

A random split of days would leak information: neighbouring days are strongly
correlated, so the model would be tested on weather it has nearly seen.
"""

import xarray as xr

from ..config import Period


def split_by_year[T: (xr.Dataset, xr.DataArray)](ds: T, period: Period) -> dict[str, T]:
    """Return {"train", "val", "test"}: the days of ds in each split's years."""
    year = ds.time.dt.year
    years = {"train": period.train_years, "val": period.val_years, "test": period.test_years}
    return {split: ds.sel(time=year.isin(ys)) for split, ys in years.items()}
