"""Matched daily (coarse, fine) pairs: what every model trains on and is scored on.

`prepare` builds them once from the raw files and saves one pair of files per
split to `cfg.paths.processed`; training and evaluation read those back with
`load_split`, so the slow hourly-to-daily step runs only once.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ..config import Config
from ..data import chirps, era5
from ..data.grid import check_alignment


def match_days[T: (xr.Dataset, xr.DataArray)](
    coarse: T, fine: xr.DataArray
) -> tuple[T, xr.DataArray]:
    """Keep only the days present in both, in the same order.

    ERA5 loses days that are incomplete after the hourly-to-daily step (31
    October), CHIRPS has every day: a model must never be scored on a day
    that only one of them has.
    """
    days = np.intersect1d(coarse.time.values, fine.time.values)
    return coarse.sel(time=days), fine.sel(time=days)


def land_mask(fine: xr.DataArray) -> xr.DataArray:
    """True where CHIRPS has a value on every day. CHIRPS covers land only."""
    return fine.notnull().all("time").rename("land")


def build_pairs(cfg: Config, years: list[int]) -> tuple[xr.Dataset, xr.DataArray]:
    """Daily ERA5 inputs (all variables) and CHIRPS rain for these years, checked and matched."""
    coarse = era5.to_daily(era5.load(cfg, years))
    fine = chirps.load(cfg, years)["precip"]
    check_alignment(coarse, fine, cfg.scale_factor)
    return match_days(coarse, fine)


def _files(cfg: Config, split: str) -> tuple[Path, Path]:
    folder = cfg.paths.processed
    return folder / f"{split}_coarse.nc", folder / f"{split}_fine.nc"


def save_split(cfg: Config, split: str, coarse: xr.Dataset, fine: xr.DataArray) -> None:
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    coarse_file, fine_file = _files(cfg, split)
    coarse.to_netcdf(coarse_file)
    fine.to_netcdf(fine_file, encoding={"precip": {"zlib": True, "complevel": 4}})


def load_split(cfg: Config, split: str) -> tuple[xr.Dataset, xr.DataArray]:
    """Read back one split written by `prepare`: (daily ERA5, CHIRPS rain)."""
    coarse_file, fine_file = _files(cfg, split)
    if not coarse_file.exists():
        raise FileNotFoundError(f"{coarse_file} not found; run `rainfall prepare` first")
    return xr.load_dataset(coarse_file), xr.load_dataarray(fine_file)
