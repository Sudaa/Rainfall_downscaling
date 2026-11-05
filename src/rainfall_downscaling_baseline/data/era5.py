"""ERA5 coarse inputs: download from the Copernicus Climate Data Store, convert to daily.

Files are written to `cfg.paths.raw / "era5"`, one file per variable group and
year. Needs a CDS account and the API key in `~/.cdsapirc` (never commit that file).
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from cdsapi.api import read_config
from ecmwf.datastores import Client

from ..config import Config

SINGLE_LEVELS = "reanalysis-era5-single-levels"
PRESSURE_LEVELS = "reanalysis-era5-pressure-levels"
ACCUMULATED = {"tp"}  # summed over each hour; every other variable is instantaneous
HOUR = np.timedelta64(1, "h")


def _client() -> Client:
    """A CDS client using the URL and key in ~/.cdsapirc."""
    settings = read_config(Path.home() / ".cdsapirc")
    return Client(url=settings["url"], key=settings["key"], progress=False)


def _rejection_reason(remote) -> str:
    """The CDS explains a rejection only in the error raised when asking for results."""
    try:
        remote.get_results()
    except Exception as err:
        return str(err)
    return ""


def _base_request(cfg: Config, year: int) -> dict:
    """Fields shared by every request: product, dates, hours, area and file format."""
    r = cfg.region
    return {
        "product_type": ["reanalysis"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in cfg.period.months],
        "day": [f"{d:02d}" for d in range(1, 32)],  # the CDS skips dates like 31 June
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [r.lat_max, r.lon_min, r.lat_min, r.lon_max],  # north, west, south, east
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _requests(cfg: Config, year: int) -> dict[str, tuple[str, dict]]:
    """Every request for one year, keyed by the file name it is saved to.

    Each single-level variable gets its own request: asked for together, an
    accumulated variable (precipitation) and an instantaneous one (water vapour)
    come back as a zip of separate files instead of one NetCDF.
    """
    base = _base_request(cfg, year)
    requests = {}
    for var in cfg.era5.single_level_vars:
        requests[f"era5_{var}_{year}.nc"] = (SINGLE_LEVELS, base | {"variable": [var]})
    level = cfg.era5.pressure_level
    requests[f"era5_wind{level}_{year}.nc"] = (
        PRESSURE_LEVELS,
        base | {"variable": cfg.era5.pressure_level_vars, "pressure_level": [str(level)]},
    )
    return requests


def download(
    cfg: Config,
    years: list[int],
    max_queued: int = 2,
    poll_seconds: int = 60,
    print_seconds: int = 600,
) -> list[Path]:
    """Download hourly ERA5 for these years, for the configured months and region.

    The CDS rejects a request when a user already has too many queued for the
    same dataset, and the limit differs per dataset. So at most `max_queued`
    requests per dataset are in the queue at once, lowered automatically when
    the CDS rejects one for that reason; the next is submitted as soon as one
    finishes. Single-level and pressure-level requests still wait side by side.
    Each file is downloaded as soon as its request finishes.

    Job IDs are saved in jobs.json: running this again after a crash or lost
    connection picks up the queued jobs instead of submitting them twice.
    Files that already exist are skipped.
    """
    out_dir = cfg.paths.raw / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs_file = out_dir / "jobs.json"
    jobs: dict[str, str] = json.loads(jobs_file.read_text()) if jobs_file.exists() else {}
    client = _client()

    def save_jobs() -> None:
        jobs_file.write_text(json.dumps(jobs, indent=2))

    wanted = {name: req for year in years for name, req in _requests(cfg, year).items()}
    active = {}  # submitted to the CDS, not downloaded yet
    waiting = []  # not submitted yet
    for name in wanted:
        if (out_dir / name).exists():
            print(f"skip {name}: already downloaded")
        elif name in jobs:
            active[name] = client.get_remote(jobs[name])
        else:
            waiting.append(name)

    # Precipitation first: the bilinear and quantile-mapping baselines need only
    # that, so they are not held up by the other inputs, which only the U-Net uses.
    def precipitation_first(name: str) -> bool:
        return "total_precipitation" not in name

    waiting.sort(key=precipitation_first)
    rejections: Counter[str] = Counter()
    caps = {SINGLE_LEVELS: max_queued, PRESSURE_LEVELS: max_queued}

    def queued(dataset: str) -> int:
        return sum(wanted[n][0] == dataset for n in active)

    last_summary, last_print = "", 0.0
    while active or waiting:
        for name in list(waiting):
            dataset, request = wanted[name]
            if queued(dataset) < caps[dataset]:
                active[name] = client.submit(dataset, request)
                jobs[name] = active[name].request_id
                save_jobs()
                waiting.remove(name)
                print(f"submitted {name}")

        statuses = {}
        for name, remote in list(active.items()):
            status = statuses[name] = remote.status
            if status == "successful":
                # Download to a temporary name first: an interrupted download
                # then never looks like a finished file.
                target = out_dir / name
                partial = target.with_suffix(".part")
                remote.download(str(partial))
                partial.rename(target)
                del active[name]
                print(f"downloaded {name}")
            elif status in ("failed", "rejected"):
                reason = _rejection_reason(remote)
                del active[name], jobs[name]
                save_jobs()
                waiting.append(name)
                waiting.sort(key=precipitation_first)  # a resubmitted job keeps its priority
                dataset = wanted[name][0]
                if "Maximum number queued" in reason:
                    caps[dataset] = max(1, queued(dataset))
                    print(
                        f"{name}: CDS queue full, at most {caps[dataset]} at once for this dataset"
                    )
                    continue
                rejections[name] += 1
                if rejections[name] >= 3:
                    raise RuntimeError(f"CDS {status} {name} 3 times: {reason}")
                print(f"{name} was {status} by the CDS ({reason}); will submit it again")

        if active or waiting:
            # Check every poll_seconds, but print only on a change or every
            # print_seconds, so a long wait does not flood the terminal.
            counts = dict(Counter(statuses[n] for n in active))
            summary = f"in queue: {counts}, not yet submitted: {len(waiting)}"
            if summary != last_summary or time.time() - last_print >= print_seconds:
                print(f"{time.strftime('%H:%M')} {summary}")
                last_summary, last_print = summary, time.time()
            time.sleep(poll_seconds)
    return [out_dir / name for name in wanted]


def load(cfg: Config, years: list[int]) -> xr.Dataset:
    """Open the downloaded files for these years as one Dataset with dims (time, lat, lon)."""
    era5_dir = cfg.paths.raw / "era5"
    files = sorted(f for year in years for f in era5_dir.glob(f"era5_*_{year}.nc"))
    if not files:
        raise FileNotFoundError(f"no ERA5 files for {years} in {era5_dir}; run download first")

    # Group files by variable (the name without its year), join each group along time.
    groups: dict[str, list[xr.Dataset]] = {}
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
        if "pressure_level" in ds.dims:
            ds = ds.squeeze("pressure_level", drop=True)
        ds = ds.drop_vars(["number", "expver"], errors="ignore")
        groups.setdefault(f.stem.rsplit("_", 1)[0], []).append(ds)

    per_variable = [xr.concat(parts, dim="time") for parts in groups.values()]
    # join="exact" fails loudly if variables were downloaded on different grids or dates.
    return xr.merge(per_variable, join="exact")


def to_daily(ds: xr.Dataset, day_start_hour: int = 0) -> xr.Dataset:
    """Turn hourly ERA5 into daily values in the units of the data contract.

    The day labelled D starts `day_start_hour` hours after 00:00 UTC on D and
    lasts 24 hours. Negative values start it on the evening before: -7 gives
    Thai local days (00:00 in Bangkok is 17:00 UTC). The default 0 (UTC days)
    matched CHIRPS best in 2022; see docs/data_contract.md.

    - Precipitation (`tp`) is an accumulation: the value at time T is the rain
      in the hour *ending* at T. So the value at 00:00 belongs to the day
      before. Hourly values are summed and converted from m to mm, and the
      variable is renamed `precip` to match CHIRPS.
    - Everything else (water vapour, winds) is instantaneous: the daily mean.

    Days that do not have all 24 hours (at the edges of the downloaded months)
    are dropped rather than returned with too little rain.
    """
    daily = {}
    complete = None
    for name, da in ds.data_vars.items():
        # Move each value to the start of the hour it describes, then back by
        # the day start, so that grouping by calendar date gives the wanted day.
        shift = (HOUR if name in ACCUMULATED else 0 * HOUR) + day_start_hour * HOUR
        shifted = da.assign_coords(time=da.time - shift)
        if name in ACCUMULATED:
            out = shifted.resample(time="1D").sum() * 1000
            out.attrs = {"units": "mm/day", "long_name": "daily precipitation"}
            daily["precip"] = out
        else:
            out = shifted.resample(time="1D").mean()
            out.attrs = da.attrs
            daily[name] = out

        hours = pd.Series(1, index=shifted.time.values).resample("1D").sum()
        days = set(hours.index[hours == 24])
        complete = days if complete is None else complete & days

    result = xr.Dataset(daily)
    return result.sel(time=sorted(complete))
