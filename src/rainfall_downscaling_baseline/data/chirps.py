"""CHIRPS high-resolution rainfall target: download, crop immediately, load.

CHIRPS v2.0 daily 0.05 deg files are global and large (1.1 to 1.5 GB per year),
so each one is cropped to the region and months right after download and the
global file is deleted. Cropped files are written to `cfg.paths.raw / "chirps"`.
Changing the region in the config therefore means downloading again.
"""

import time
from pathlib import Path

import requests
import xarray as xr

from ..config import Config
from .grid import crop

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf/p05"
RETRYABLE = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


def _fetch(url: str, dest: Path, max_attempts: int = 8) -> None:
    """Stream a large file to disk in chunks, resuming and retrying after errors.

    After a dropped connection the download resumes from the bytes already in
    `dest` (an HTTP Range request) instead of starting over. A "429 Too Many
    Requests" reply means the server wants us to slow down, so we wait longer
    before each new attempt.
    """
    for attempt in range(1, max_attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                if response.status_code == 416:  # range starts at the end: already complete
                    return
                response.raise_for_status()
                if response.status_code != 206:  # server ignored the range: start over
                    have = 0
                total = have + int(response.headers.get("content-length", 0))
                done = have
                with open(dest, "ab" if have else "wb") as f:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
            print()
            if done >= total:
                return
            reason = "connection closed early"
        except requests.HTTPError as err:
            if err.response.status_code != 429 and err.response.status_code < 500:
                raise  # e.g. 404 Not Found: retrying will not help
            reason = f"HTTP {err.response.status_code}"
        except RETRYABLE as err:
            reason = type(err).__name__
        wait = 60 * attempt
        print(f"\n  {reason}; attempt {attempt}/{max_attempts}, retrying in {wait} s")
        time.sleep(wait)
    raise RuntimeError(f"gave up on {url} after {max_attempts} attempts")


def download(cfg: Config, year: int) -> Path:
    """Download one year, keep only cfg.region and cfg.period.months, delete the global file."""
    out_dir = cfg.paths.raw / "chirps"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"chirps_{year}.nc"
    if target.exists():
        print(f"skip {target.name}: already downloaded")
        return target

    global_file = out_dir / f"chirps_{year}_global.part"
    print(f"download CHIRPS {year} (global file, cropped afterwards)")
    _fetch(f"{BASE_URL}/chirps-v2.0.{year}.days_p05.nc", global_file)

    # The with-block closes the global file before it is deleted (Windows locks open files).
    with xr.open_dataset(global_file) as ds:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        ds = crop(ds, cfg.region)
        ds = ds.sel(time=ds.time.dt.month.isin(cfg.period.months))
        small = ds.load()

    partial = target.with_suffix(".part")
    small.to_netcdf(partial, encoding={"precip": {"zlib": True, "complevel": 4}})
    partial.rename(target)
    global_file.unlink()
    print(f"saved {target.name}: {dict(small.sizes)}")
    return target


def load(cfg: Config, years: list[int]) -> xr.Dataset:
    """Open the cropped files for these years as one Dataset with dims (time, lat, lon)."""
    chirps_dir = cfg.paths.raw / "chirps"
    files = [chirps_dir / f"chirps_{year}.nc" for year in years]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in {chirps_dir}; run download first")
    return xr.concat([xr.open_dataset(f) for f in files], dim="time")
