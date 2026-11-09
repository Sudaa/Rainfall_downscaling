"""Figures for the report. Each function saves one figure and returns its path.

Colours: rain amounts use one blue ramp (light = dry, dark = heavy) with the
same scale in every panel; models use fixed colours in a fixed order, and
the observations (CHIRPS) are drawn in neutral ink so the truth stands apart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

RAIN = LinearSegmentedColormap.from_list(
    "rain", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
RAIN.set_bad("#e8e7e3")  # sea: CHIRPS has no data there
MODEL_COLOURS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]  # fixed order, never cycled
INK, MUTED = "#1a1a19", "#6b6a64"
LABELS = {
    "bilinear": "Bilinear",
    "quantile_mapping": "Quantile mapping",
    "unet": "U-Net",
    "unet+qm": "U-Net + QM",
    "unet+count": "U-Net + count loss",
    "unet+clip": "U-Net + CLIP (text)",
    "unet+clip-image": "U-Net + CLIP (image)",
    "unet+clip-input": "U-Net + CLIP input",
    "unet+clip-input+qm": "U-Net + CLIP input + QM",
}
# Left out of figures (still in the score table): the image-only CLIP ablation is
# almost the same line as CLIP (text), and a 9th model would need a 9th colour.
FIGURE_SKIP = ("unet+clip-image",)
# Each model keeps its own colour in every figure, whichever models are shown.
# The image-only ablation is never drawn, so its former pink went to CLIP input + QM.
COLOUR_OF = {
    "bilinear": MODEL_COLOURS[0],
    "quantile_mapping": MODEL_COLOURS[1],
    "unet": MODEL_COLOURS[2],
    "unet+qm": MODEL_COLOURS[3],
    "unet+clip-input+qm": MODEL_COLOURS[4],
    "unet+clip-input": MODEL_COLOURS[5],
    "unet+clip": MODEL_COLOURS[6],
    "unet+count": MODEL_COLOURS[7],
}
FLOOR = 1e-5  # bottom of the log axis in exceedance()


def wettest_day(obs: xr.DataArray, threshold: float = 50.0) -> str:
    """The day with the largest share of land above `threshold` mm: a heavy-rain test case."""
    share = (obs >= threshold).sum(["lat", "lon"]) / obs.notnull().sum(["lat", "lon"])
    return str(share.idxmax("time").values)[:10]


def map_panel(
    coarse: xr.DataArray,
    predictions: dict[str, xr.DataArray],
    obs: xr.DataArray,
    land: xr.DataArray,
    day: str,
    out: Path,
) -> Path:
    """One row of maps for a single day: coarse input, each model, observed. Same colour scale."""
    panels = {"ERA5 input (0.25°)": coarse.sel(time=day)}
    panels |= {
        LABELS.get(name, name): pred.sel(time=day).where(land) for name, pred in predictions.items()
    }
    panels["CHIRPS (observed)"] = obs.sel(time=day)
    vmax = float(np.ceil(obs.sel(time=day).quantile(0.99) / 10) * 10)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 4.6), sharey=True)
    for ax, (title, field) in zip(axes, panels.items(), strict=True):
        image = field.plot(ax=ax, cmap=RAIN, vmin=0, vmax=vmax, add_colorbar=False)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8, colors=MUTED)
    axes[0].set_ylabel("latitude (°N)", color=MUTED)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, extend="max", pad=0.01)
    bar.set_label("rain (mm/day)", color=MUTED)
    fig.suptitle(f"Daily rain on {day}, land only", fontsize=13, color=INK, x=0.06, ha="left")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def exceedance(
    predictions: dict[str, xr.DataArray], obs: xr.DataArray, land: xr.DataArray, out: Path
) -> Path:
    """Share of land cell-days with rain at or above each amount, on a log y-axis.

    Shows at a glance whether a model rains too often (curve above CHIRPS at the
    left) or misses the heavy tail (curve below CHIRPS at the right).
    """
    amounts = np.arange(0, 151, 1.0)
    keep = (land & obs.notnull()).broadcast_like(obs).values

    def curve(field: xr.DataArray) -> np.ndarray:
        values = np.sort(field.values[keep])
        return 1 - np.searchsorted(values, amounts, side="left") / len(values)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    lines = {
        LABELS.get(name, name): (curve(pred), colour)
        for name, pred in predictions.items()
        for colour in [COLOUR_OF.get(name, MUTED)]
    }
    lines["CHIRPS (observed)"] = (curve(obs), INK)
    ends = []  # (log10 of y, x, name) where each curve leaves the plot
    for name, (share, colour) in lines.items():
        dashes = (4, 2) if colour == INK else ()
        ax.plot(amounts, share, color=colour, linewidth=2, label=name, dashes=dashes)
        last = np.flatnonzero(share >= FLOOR)  # the last point still visible on the axis
        if len(last):
            ends.append((np.log10(share[last[-1]]), amounts[last[-1]], name))

    # Direct labels at the curve ends, pushed apart so lines ending close together
    # do not print their names on top of each other (gap in decades of the log axis).
    gap, previous = 0.17, np.inf
    for log_y, x, name in sorted(ends, reverse=True):
        log_y = min(log_y, previous - gap)
        ax.annotate(
            name,
            (x, 10**log_y),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            va="center",
        )
        previous = log_y
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, 1)
    ax.set_xlim(0, 170)
    ax.set_xlabel("rain amount (mm/day)", color=MUTED)
    ax.set_ylabel("share of land cell-days at or above", color=MUTED)
    ax.grid(True, which="major", color="#e8e7e3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("How often each amount of rain occurs", loc="left", fontsize=13, color=INK)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
