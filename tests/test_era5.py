"""Tests for ERA5 hourly -> daily. Small made-up data, so every answer can be worked out by hand."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rainfall_downscaling_baseline.data.era5 import to_daily


def hourly(start: str, hours: int, tp: float = 0.0) -> xr.Dataset:
    """A 2x2 grid with hourly tp (metres per hour) and tcwv equal to the hour of day."""
    time = pd.date_range(start, periods=hours, freq="h")
    shape = (hours, 2, 2)
    coords = {"time": time, "lat": [14.0, 13.75], "lon": [100.0, 100.25]}
    tcwv = np.broadcast_to(time.hour.values[:, None, None], shape).astype(float)
    return xr.Dataset(
        {
            "tp": (("time", "lat", "lon"), np.full(shape, tp)),
            "tcwv": (("time", "lat", "lon"), tcwv),
        },
        coords=coords,
    )


def test_constant_rain_sums_to_24_mm():
    # 1 mm every hour for three days -> 24 mm per day
    daily = to_daily(hourly("2022-05-01 00:00", 72, tp=0.001))
    assert np.allclose(daily["precip"], 24.0)
    assert daily["precip"].attrs["units"] == "mm/day"
    assert "tp" not in daily


def test_value_at_midnight_belongs_to_previous_day():
    ds = hourly("2022-05-01 00:00", 72)
    ds["tp"].loc[{"time": "2022-05-02 00:00"}] = 0.010  # rain in 23:00-24:00 on 1 May
    daily = to_daily(ds)
    assert float(daily["precip"].sel(time="2022-05-01").max()) == pytest.approx(10.0)
    assert float(daily["precip"].sel(time="2022-05-02").max()) == 0.0


def test_incomplete_days_are_dropped():
    # Hours 00:00 on 1 May to 23:00 on 3 May. For rain, 30 April gets only the
    # 00:00 value and 3 May misses the value at 00:00 on 4 May: both incomplete.
    daily = to_daily(hourly("2022-05-01 00:00", 72, tp=0.001))
    assert [str(t)[:10] for t in daily.time.values] == ["2022-05-01", "2022-05-02"]


def test_instantaneous_variables_are_daily_means():
    daily = to_daily(hourly("2022-05-01 00:00", 72))
    assert np.allclose(daily["tcwv"], np.mean(range(24)))  # hours 0..23 -> 11.5


def test_day_start_moves_the_window():
    # Rain only in the hour 17:00-18:00 UTC on 1 May. That is 00:00-01:00 on
    # 2 May in Bangkok, so Thai days (-7) put it on 2 May, UTC days on 1 May.
    ds = hourly("2022-05-01 00:00", 96)
    ds["tp"].loc[{"time": "2022-05-01 18:00"}] = 0.005
    thai = to_daily(ds, day_start_hour=-7)
    assert float(thai["precip"].sel(time="2022-05-02").max()) == pytest.approx(5.0)
    # Thai 1 May starts at 17:00 UTC on 30 April, before the data: incomplete, dropped
    assert str(thai.time.values[0])[:10] == "2022-05-02"
    utc = to_daily(ds)
    assert float(utc["precip"].sel(time="2022-05-01").max()) == pytest.approx(5.0)


def test_gap_between_years_is_not_filled():
    # Two separate seasons joined in one dataset, as load() returns for two years
    ds = xr.concat([hourly("2022-10-30", 48), hourly("2023-05-01", 48)], dim="time")
    daily = to_daily(ds)
    assert len(daily.time) == 2  # 30 Oct 2022 and 1 May 2023, nothing in between
