"""Sanity check: do ERA5 and CHIRPS 2022 look like the same rain?

Run as a script (`uv run python notebooks/01_sanity_check.py`) or cell by cell
in VS Code (each `# %%` line starts a cell). Not part of the package: this is
exploration, so it only reads data and saves a figure.
"""

# %% 1. Load both datasets through the package's own loaders
import matplotlib.pyplot as plt
import xarray as xr

from rainfall_downscaling_baseline.config import load_config
from rainfall_downscaling_baseline.data import chirps, era5

cfg = load_config()
coarse = era5.load(cfg, [2022])
fine = chirps.load(cfg, [2022])
print(coarse)
print(fine)

# %% 2. Quick daily ERA5 rain: sum the 24 hourly values of each UTC day, metres -> mm
# Rough on purpose. to_daily() will do this properly (which hours belong to
# which day, and in which time zone); this plot shows whether that matters.
era5_daily = coarse["tp"].resample(time="1D").sum() * 1000
chirps_daily = fine["precip"]
print("ERA5 daily mm:  ", float(era5_daily.min()), "to", float(era5_daily.max()))
print("CHIRPS daily mm:", float(chirps_daily.min()), "to", float(chirps_daily.max()))
print("CHIRPS NaN fraction (ocean):", float(chirps_daily.isel(time=0).isnull().mean()))

# %% 3. Pick the wettest day in CHIRPS (Thailand-mean rain)
wettest = chirps_daily.mean(["lat", "lon"]).idxmax("time").values
print("wettest day:", str(wettest)[:10])

# %% 4. Maps of that day side by side, on the same colour scale
day_era5 = era5_daily.sel(time=wettest)
day_chirps = chirps_daily.sel(time=wettest)
vmax = float(max(day_era5.quantile(0.99), day_chirps.quantile(0.99)))

fig = plt.figure(figsize=(12, 9))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 1, 2)

day_era5.plot(ax=ax1, vmin=0, vmax=vmax, cmap="Blues", cbar_kwargs={"label": "mm/day"})
ax1.set_title(f"ERA5 0.25 deg, {str(wettest)[:10]}")
day_chirps.plot(ax=ax2, vmin=0, vmax=vmax, cmap="Blues", cbar_kwargs={"label": "mm/day"})
ax2.set_title(f"CHIRPS 0.05 deg, {str(wettest)[:10]}")

# %% 5. Thailand-mean rain every day, over land only so both cover the same area
# CHIRPS has no ocean values; ERA5 does. Put ERA5 on the CHIRPS grid (nearest
# cell) and hide the ocean, otherwise ERA5's rain over the sea inflates its mean.
era5_on_land = era5_daily.reindex_like(chirps_daily, method="nearest").where(chirps_daily.notnull())
era5_on_land.mean(["lat", "lon"]).plot(ax=ax3, label="ERA5 (land only)")
chirps_daily.mean(["lat", "lon"]).plot(ax=ax3, label="CHIRPS")
ax3.axvline(wettest, color="grey", linestyle="--", linewidth=1)
ax3.set_ylabel("mean rain over land (mm/day)")
ax3.set_title("Daily Thailand-mean rain, May to Oct 2022")
ax3.legend()

fig.tight_layout()
out = cfg.paths.figures / "sanity_2022.png"
fig.savefig(out, dpi=120)
print("saved", out)

# %% 6. One number to go with the picture: do the two series move together?
corr = float(xr.corr(era5_on_land.mean(["lat", "lon"]), chirps_daily.mean(["lat", "lon"])))
print(f"correlation of daily Thailand-mean rain: {corr:.2f}")
plt.show()
