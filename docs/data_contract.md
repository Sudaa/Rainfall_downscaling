# Data contract

What the pipeline promises about its data after loading. Fill in each "?" from
the dataset documentation and from what you see in the files, not from memory.
Tests in `tests/test_grid.py` check these values.

| Property | ERA5 (input) | CHIRPS (target) |
|---|---|---|
| Product and version | ERA5 hourly reanalysis, single levels + pressure levels (Copernicus CDS) | CHIRPS v2.0 daily, p05 |
| Variables (file name) | tp, tcwv, u and v at 850 hPa | precip |
| Units on disk | tp: m per hour (accumulated); tcwv: kg m-2; u, v: m s-1 | mm/day |
| Units after loading | precip: mm/day (after `to_daily`); tcwv: kg m-2; u, v: m s-1 | mm/day |
| Grid spacing | 0.25 deg | 0.05 deg |
| Coordinates are cell centres or edges? | ? | ? |
| Latitude order | north to south (21.0 to 5.0) | south to north (5.025 to 20.975) |
| Longitude range (0..360 or -180..180) | -180..180 | -180..180 |
| Time step on disk | hourly | daily |
| Definition of a "day" (time zone, start hour) | tp at T = rain in the hour ending at T (UTC). `to_daily` uses 00-24 UTC (07:00-07:00 Bangkok) | "time denotes the first day of the given day"; start hour not stated in the file |
| Missing values (how marked, where) | none | NaN over the sea (36% of the box); land complete |

Region: 5N to 21N, 97E to 106E. Period: see `configs/default.toml`.

## Decisions and why

- **ERA5 day = 00-24 UTC.** Compared with CHIRPS 2022, a day starting at 00 UTC gave cell
  correlation 0.462, a day starting at Thai midnight (17 UTC) about 0.40. 03 UTC was equal
  (0.463), so the standard UTC day was kept.
- **31 October is dropped every year.** Its last hour of rain is stamped 00:00 on 1 November,
  which is not downloaded. A 23-hour day would underestimate rain, so `to_daily` keeps only
  complete days: 183 days per season instead of 184. CHIRPS days are matched to the ERA5 days.
- **ERA5 `tp` is renamed `precip`** after conversion, so both datasets use the same name and unit.
