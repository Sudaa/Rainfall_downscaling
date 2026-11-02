# Data contract

What the pipeline promises about its data after loading. Fill in each "?" from
the dataset documentation and from what you see in the files, not from memory.
Tests in `tests/test_grid.py` check these values.

| Property | ERA5 (input) | CHIRPS (target) |
|---|---|---|
| Product and version | ? | ? |
| Variables (file name) | ? | ? |
| Units on disk | ? | ? |
| Units after loading | ? | mm/day |
| Grid spacing | 0.25 deg | 0.05 deg |
| Coordinates are cell centres or edges? | ? | ? |
| Latitude order | ? | ? |
| Longitude range (0..360 or -180..180) | ? | ? |
| Time step on disk | hourly | daily |
| Definition of a "day" (time zone, start hour) | ? | ? |
| Missing values (how marked, where) | ? | ? |

Region: 5N to 21N, 97E to 106E. Period: see `configs/default.toml`.

## Decisions and why

- ?
