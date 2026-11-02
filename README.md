# Rainfall downscaling baseline

Statistical downscaling of daily rainfall over Thailand, from ERA5 (0.25 deg)
to CHIRPS (0.05 deg), with three baselines: bilinear interpolation, quantile
mapping and a small U-Net.

## Problem statement

TODO: write this in your own words (2-3 sentences).

## Project layout

```
configs/default.toml        experiment settings (region, years, variables, paths)
src/rainfall_downscaling_baseline/
    config.py               loads and validates the settings
    cli.py                  `uv run rainfall <command>`
    data/                   era5.py, chirps.py (download, load), grid.py (crop, align)
    preprocessing/          transforms.py, splits.py, dataset.py
    models/                 base.py (interface), bilinear.py, quantile_mapping.py, unet.py
    evaluation/             metrics.py, evaluate.py, plots.py
tests/                      one test file per module group
docs/                       data contract, paper notes, learning log, report
data/                       raw, interim, processed (not in git)
```

## Quickstart

```
uv sync
uv run rainfall config      # show the settings a run will use
uv run rainfall download
uv run rainfall prepare
uv run rainfall train
uv run rainfall evaluate
uv run pytest
```
