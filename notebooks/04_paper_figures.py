"""Figures for the paper (paper/figures/), sized for the ACM two-column page.

  maps_paper.png      one row of maps on the wettest validation day: ERA5
                      input, bilinear, U-Net, U-Net + QM, CLIP-input U-Net +
                      QM and CHIRPS, same colour scale, Bangkok box outlined
  workflow_paper.png  how the CLIP-input U-Net works, from ERA5 to rain

Uses the seed-0 checkpoints in checkpoints/. Predicting with the CLIP-input
U-Net computes CLIP features for the validation days, which takes a few
minutes. Run:
    uv run python notebooks/04_paper_figures.py
"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from rainfall_downscaling_baseline.config import DEFAULT_CONFIG, load_config
from rainfall_downscaling_baseline.evaluation.plots import INK, MUTED, RAIN, wettest_day

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures"
BANGKOK = (100.3, 13.5, 0.6, 0.5)  # lon, lat, width, height
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})


CACHE = Path(__file__).resolve().parents[1] / "figures" / "maps_paper_cache.pkl"


def maps() -> Path:
    """Predictions are cached in figures/, so restyling the figure takes seconds."""
    import pickle

    if CACHE.exists():
        with CACHE.open("rb") as f:
            panels, land, day = pickle.load(f)
    else:
        panels, land, day = _predict_maps()
        with CACHE.open("wb") as f:
            pickle.dump((panels, land, day), f)
    return _draw_maps(panels, land, day)


def _predict_maps():
    from rainfall_downscaling_baseline.models.bilinear import BilinearDownscaler
    from rainfall_downscaling_baseline.models.quantile_mapping import QuantileMappingDownscaler
    from rainfall_downscaling_baseline.models.unet import UNetDownscaler
    from rainfall_downscaling_baseline.preprocessing.pairs import land_mask, load_split

    cfg = load_config(DEFAULT_CONFIG)
    train_coarse, train_fine = load_split(cfg, "train")
    coarse, obs = load_split(cfg, "val")
    land = land_mask(obs)
    day = wettest_day(obs)
    one = coarse.sel(time=[day])

    panels = {"ERA5 input\n(0.25°)": coarse["precip"].sel(time=day)}
    bilinear = BilinearDownscaler()
    bilinear.fit(train_coarse, train_fine)
    panels["Bilinear\n(0.05°)"] = bilinear.predict(one).isel(time=0)
    for file, label, label_qm in [
        ("unet.pt", "U-Net\n", "U-Net\n+ QM"),
        ("unet_clip-input.pt", None, "U-Net + CLIP\ninput + QM"),
    ]:
        unet = UNetDownscaler.load(cfg.paths.checkpoints / file)
        if label:
            panels[label] = unet.predict(one).isel(time=0)
        qm = QuantileMappingDownscaler(base=unet)
        qm.fit(train_coarse, train_fine)  # QM is fitted on the U-Net's training-year output
        panels[label_qm] = qm.predict(one).isel(time=0)
        print(f"done {file}", flush=True)
    panels["CHIRPS\n(observed)"] = obs.sel(time=day)
    return {k: v.load() for k, v in panels.items()}, land.load(), day


def _draw_maps(panels, land, day) -> Path:
    vmax = 100.0
    fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 2.5), sharey=True)
    for i, (ax, (title, field)) in enumerate(zip(axes, panels.items(), strict=True)):
        if i > 0:  # the ERA5 input is on its own coarse grid
            field = field.where(land)
        image = field.plot(ax=ax, cmap=RAIN, vmin=0, vmax=vmax, add_colorbar=False)
        ax.add_patch(Rectangle(BANGKOK[:2], *BANGKOK[2:], fill=False, lw=0.8, ec=INK))
        ax.set_title(title, fontsize=6.5, color=INK)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")
        ax.set_xticks([98, 102, 106])
        ax.tick_params(labelsize=5.5, colors=MUTED, length=2)
    axes[0].set_ylabel("latitude (°N)", color=MUTED, fontsize=6)
    bar = fig.colorbar(image, ax=axes, shrink=0.8, extend="max", pad=0.025, aspect=25)
    bar.set_label("rain (mm/day)", color=MUTED, fontsize=6)
    bar.ax.tick_params(labelsize=5.5, colors=MUTED)
    out = OUT / "maps_paper.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"day shown: {day}")
    return out


PLAIN, FEATURE, ACCENT, NET = "#f2f1ec", "#e3effc", "#fdeee6", "#ffffff"


def workflow(detail: bool = False) -> Path:
    """Overview for the paper: input -> U-Net with image-feature path -> QM -> output.

    Uses real maps from the map-figure cache, so run the full script once first.
    """
    import pickle

    import numpy as np

    with CACHE.open("rb") as f:
        panels, land, _ = pickle.load(f)
    fields = list(panels.values())  # ERA5, bilinear, U-Net, U-Net+QM, CLIP+QM, CHIRPS
    era5, bil, final, chirps = fields[0], fields[1].where(land), fields[4].where(land), fields[5]

    fh, y0 = (2.95, -12) if detail else (2.55, 0)
    fig, ax = plt.subplots(figsize=(7.1, fh))
    ax.set_xlim(0, 200)
    ax.set_ylim(y0, 76)
    ax.axis("off")
    blue = "#2a78d6"

    def thumb(x, y, w, field, label):
        h = w * (7.1 / 200) / (fh / (76 - y0)) * 16 / 9  # keep the 9° x 16° map shape
        inset = ax.inset_axes([x, y, w, h], transform=ax.transData)
        values = (
            np.flipud(field.values) if field.lat.values[0] < field.lat.values[-1] else field.values
        )
        inset.imshow(
            np.ma.masked_invalid(values),
            cmap=RAIN,
            vmin=0,
            vmax=100,
            interpolation="nearest",
            aspect="auto",
        )
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
        ax.text(
            x + w / 2,
            y - 2.2,
            label,
            ha="center",
            va="top",
            fontsize=5.5,
            color=INK,
            linespacing=1.25,
        )
        return x, y, w, h

    def arrow(a, b, colour=INK, dashed=False, both=False):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="<|-|>" if both else "-|>",
                mutation_scale=6,
                lw=0.7,
                color=colour,
                linestyle=(0, (3, 2)) if dashed else "-",
                shrinkA=0,
                shrinkB=0,
            )
        )

    def box(x, y, w, h, text, fill=PLAIN, colour=INK, size=5.5):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=1",
                fc=fill,
                ec=colour,
                lw=0.6,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.25)

    def stage(x0, x1, text):
        ax.text(
            (x0 + x1) / 2,
            73,
            text,
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold",
            color=INK,
        )
        ax.plot([x0, x1], [70.5, 70.5], color=MUTED, lw=0.5)

    stage(2, 44, "Input")
    stage(50, 124, "Model")
    stage(130, 198, "Post-processing and output")

    # Input: coarse ERA5 and its bilinear interpolation
    e = thumb(
        4,
        30,
        15,
        era5,
        "ERA5, 0.25°\nrain, TCWV,\nu, v 850 hPa" if detail else "ERA5, 0.25°\n4 variables",
    )
    b = thumb(
        27, 30, 15, bil, "Bilinear\nto 0.05°;\nlog(1 + r)" if detail else "Bilinear\nto 0.05°"
    )
    arrow((e[0] + e[2] + 0.8, 46), (b[0] - 0.8, 46))

    # Model: residual U-Net drawn as a U of feature maps
    # Each level: (centre y, height ~ grid size, width ~ channels); bottleneck last
    levels = [(52, 22, 1.6), (44, 16, 2.4), (36, 11, 3.2)]
    enc_x, dec_x = [55, 65, 73.5], [105, 95, 86.5]
    bxc, byc, bh = 80, 26, 7
    for xs in (enc_x, dec_x):
        for x, (cy, h, w) in zip(xs, levels, strict=True):
            ax.add_patch(Rectangle((x - w / 2, cy - h / 2), w, h, fc=NET, ec=INK, lw=0.6))
    ax.add_patch(Rectangle((bxc - 2.4, byc - bh / 2), 4.8, bh, fc=ACCENT, ec=INK, lw=0.6))
    # Skip connections near the top of each level, above the deeper levels
    for x1, x2, (cy, h, w) in zip(enc_x, dec_x, levels, strict=True):
        y = cy + h / 2 - 2
        arrow((x1 + w / 2 + 0.4, y), (x2 - w / 2 - 0.4, y), colour=MUTED, dashed=True)
    # Down the encoder and up the decoder with right-angled arrows: out of the
    # bottom of a level, then straight across into the next one (mirrored going up)
    down = [(x, cy, h, w) for x, (cy, h, w) in zip(enc_x, levels, strict=True)]
    down.append((bxc, byc, bh, 4.8))
    turns = [37, 31, 24]  # the height at which each arrow turns, inside the next level
    for ((x1, c1, h1, _), (x2, _, _, w2)), ty in zip(pairwise(down), turns, strict=True):
        ax.plot([x1, x1], [c1 - h1 / 2, ty], color=INK, lw=0.7)
        arrow((x1 - 0.35, ty), (x2 - w2 / 2 - 0.3, ty))
    up = [(bxc, byc, bh, 4.8)] + [
        (x, cy, h, w) for x, (cy, h, w) in list(zip(dec_x, levels, strict=True))[::-1]
    ]
    for ((x1, _, _, w1), (x2, c2, h2, _)), ty in zip(pairwise(up), turns[::-1], strict=True):
        ax.plot([x1 + w1 / 2, x2 + 0.35], [ty, ty], color=INK, lw=0.7)
        arrow((x2, ty), (x2, c2 - h2 / 2 - 0.3))
    ax.text(80, 66.5, "Residual U-Net", ha="center", fontsize=6, color=INK)
    ax.text(62, 63, "encoder", ha="center", fontsize=5, color=MUTED)
    ax.text(98, 63, "decoder", ha="center", fontsize=5, color=MUTED)
    arrow((b[0] + b[2] + 0.8, 52), (enc_x[0] - 1.2, 52))
    if detail:  # channels and grid size of each level
        # Left of the vertical part of each arrow, below the arrow into this level
        for x, text, ty in zip(
            enc_x, ["32 ch\n128²", "64 ch\n64²", "128 ch\n32²"], [39.6, 33.2, 26.6], strict=True
        ):
            ax.text(
                x - 1.3,
                ty,
                text,
                ha="right",
                va="center",
                fontsize=4.4,
                color=MUTED,
                linespacing=1.15,
            )
        ax.text(
            bxc - 1.5,
            byc - bh / 2 - 2.2,
            "256 ch, 16²",
            ha="right",
            va="center",
            fontsize=4.4,
            color=MUTED,
        )
        ax.text(80, 63, "- - skip", ha="center", fontsize=5, color=MUTED)

    # Image-feature path into the bottleneck
    fy, fbh = (5.5, 11) if detail else (7.5, 8)
    fxs = [55, 81, 107]
    if detail:
        ftexts = [
            "Rain map drawn\nas image: 128²\ntiles, stride 32",
            "ViT-B/32, frozen\n(CLIP or random)\n→ 7 × 7 × 768",
            "PCA → 32 dims;\ntiles averaged\n→ 32 × 40 × 23",
        ]
    else:
        ftexts = [
            "Rain map drawn\nas image tiles",
            "ViT-B/32, frozen\n(CLIP or random)",
            "PCA to 32\nfeature maps",
        ]
    for x, t in zip(fxs, ftexts, strict=True):
        box(x, fy, 23, fbh, t, fill=FEATURE, colour=blue, size=5.0 if detail else 5.2)
    for a, c in pairwise(fxs):
        arrow((a + 12, fy), (c - 12, fy), colour=blue)
    arrow((b[0] + b[2] + 0.8, 34), (fxs[0] - 9, fy + fbh / 2 + 0.4), colour=blue)
    ax.plot([fxs[2], fxs[2], bxc], [fy + fbh / 2 + 0.4, 16, 16], color=blue, lw=0.7)
    arrow((bxc, 16), (bxc, byc - bh / 2 - 0.4), colour=blue)
    ax.text(
        bxc + 2, 18.3, "joined at the bottleneck", fontsize=5, color=blue, ha="left", va="center"
    )
    ax.text(
        30,
        fy + 1.5,
        "Image-feature\npath",
        ha="center",
        va="center",
        fontsize=5.5,
        color=blue,
        style="italic",
        linespacing=1.2,
    )

    # Post-processing and output
    px = 114
    ax.add_patch(plt.Circle((px, 52), 2.4, fc=NET, ec=INK, lw=0.6))
    ax.text(px, 52, "+", ha="center", va="center", fontsize=7, color=INK)
    arrow((dec_x[0] + 1, 52), (px - 2.5, 52))
    if not detail:
        ax.text(
            px + 3.5,
            47.5,
            "add to bilinear\n(log space)",
            ha="center",
            va="top",
            fontsize=5,
            color=MUTED,
            linespacing=1.2,
        )
    if detail:
        qx, qm_half = 141, 11.6
        box(qx, 52, 22.5, 10, "Quantile mapping\nper cell & month\n(100 quantiles)", size=4.8)
        ax.text(
            px + 6,
            46.5,
            r"$\hat{r} = e^{\log(1 + r_{\mathrm{bil}}) + f_\theta} - 1$",
            ha="center",
            va="top",
            fontsize=5,
            color=INK,
        )
        ox, cx = 155, 182
    else:
        qx, qm_half = 139, 7.2
        box(qx, 52, 13, 9, "Quantile\nmapping")
        ox, cx = 152, 181
    arrow((px + 2.5, 52), (qx - qm_half, 52))
    o = thumb(ox, 30, 15, final, "Downscaled\n0.05°")
    arrow((qx + qm_half, 52), (o[0] - 0.8, 52))
    c = thumb(cx, 30, 15, chirps, "CHIRPS\n(observed)")
    arrow((o[0] + o[2] + 0.8, 46), (c[0] - 0.8, 46), dashed=True, both=True, colour=MUTED)
    ax.text(
        (o[0] + o[2] + c[0]) / 2,
        49.5,
        "loss /\nscores",
        ha="center",
        va="bottom",
        fontsize=5,
        color=MUTED,
        linespacing=1.2,
    )

    if detail:
        ax.text(
            174,
            14,
            "Training: MSE in log space\nEvaluation: MAE, bias, r,\nCSI at 20 and 50 mm/day",
            ha="center",
            va="top",
            fontsize=5,
            color=MUTED,
            linespacing=1.3,
        )
        ax.text(
            1,
            -8,
            "Trained on May–Oct 2022–23 (random 128² land patches, Adam), early "
            "stopping on 2024; the 2025 test year is held out. The plain U-Net baseline "
            "omits the blue path.",
            ha="left",
            va="center",
            fontsize=5,
            color=MUTED,
        )
    out = OUT / ("workflow_medium.png" if detail else "workflow_paper.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def workflow_detailed() -> Path:
    """The full pipeline with every setting, kept for supplementary material."""
    fig, ax = plt.subplots(figsize=(7.1, 3.75))
    ax.set_xlim(0, 200)
    ax.set_ylim(4, 100)
    ax.axis("off")

    def box(x, y, w, h, text, fill=PLAIN, size=5.0, bold=False, lw=0.6):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.25,rounding_size=0.9",
                fc=fill,
                ec=INK,
                lw=lw,
            )
        )
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color=INK,
            weight="bold" if bold else "normal",
            linespacing=1.3,
        )

    def arrow(a, b, dashed=False, colour=INK):
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=6,
                lw=0.7,
                color=colour,
                linestyle=(0, (3, 2)) if dashed else "-",
                shrinkA=0,
                shrinkB=0,
            )
        )

    def line(points, colour=INK):
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=colour, lw=0.7, solid_capstyle="butt")

    def title(x, y, text):
        ax.text(x, y, text, fontsize=6.0, weight="bold", color=INK, ha="left", va="center")

    def note(x, y, text, ha="center", size=4.8):
        ax.text(x, y, text, fontsize=size, color=MUTED, ha=ha, va="center", linespacing=1.3)

    # (a) Data and preprocessing ------------------------------------------------
    title(1, 97, "(a) Data and training")
    ax_ = 21
    box(
        ax_,
        84,
        40,
        13,
        "ERA5 reanalysis, daily, 0.25°:\nprecipitation, water vapour,\nu and v wind at 850 hPa",
    )
    box(
        ax_,
        64,
        40,
        15,
        "Bilinear interpolation to the\nCHIRPS grid (0.05°, 320×180);\n"
        "rain as log(1 + r); z-scores\nfrom 2022–23 statistics",
    )
    box(
        ax_,
        42,
        40,
        15,
        "CHIRPS v2.0, daily, 0.05°,\nland only (target y);\n"
        "residual target:\nlog(1 + y) − log(1 + r$_{\\mathrm{bil}}$)",
    )
    box(
        ax_,
        17,
        40,
        21,
        "Training (May–Oct 2022–23):\nMSE on the log residual over\n"
        "land, random 128² patches,\nAdam (lr 10$^{-3}$), batch 8,\n"
        "early stopping on 2024\n(patience 5, ≤ 40 epochs)",
        fill=NET,
    )
    arrow((ax_, 77.2), (ax_, 71.8))
    arrow((ax_, 34.2), (ax_, 27.8))

    # (c) Residual U-Net ---------------------------------------------------------
    title(49, 97, "(c) Residual U-Net (2.00 M parameters)")
    note(
        49,
        92.5,
        "Each block: two 3×3 conv–BatchNorm–ReLU.  ↘ max-pool 2   "
        "↗ transposed conv 2   - - - skip connection (concatenation)",
        ha="left",
    )
    bw, bh = 13, 8
    enc = [(56, 82, "32 ch\n128²"), (68, 71, "64 ch\n64²"), (80, 60, "128 ch\n32²")]
    dec = [(140, 82, "32 ch\n128²"), (128, 71, "64 ch\n64²"), (116, 60, "128 ch\n32²")]
    for x, y, t in enc + dec:
        box(x, y, bw, bh, t, fill=NET, size=5)
    bx, by = 98, 49
    box(bx, by, 27, 9, "Bottleneck, 16²\n128 + 32 → 256 ch", fill=ACCENT, size=5)
    hx = 155
    box(hx, 82, 10, bh, "1×1\nconv", fill=NET, size=5)
    for (x1, y1, _), (x2, y2, _) in pairwise(enc):
        arrow((x1 + 2, y1 - bh / 2 - 0.3), (x2 - bw / 2 - 0.3, y2 + 1))
    arrow((enc[2][0] + 2, enc[2][1] - bh / 2 - 0.3), (bx - 13.8, by + 1.5))
    arrow((bx + 13.8, by + 1.5), (dec[2][0] - 2, dec[2][1] - bh / 2 - 0.3))
    for (x1, y1, _), (x2, y2, _) in pairwise(dec[::-1]):
        arrow((x1 + bw / 2 + 0.3, y1 + 1), (x2 - 2, y2 - bh / 2 - 0.3))
    for (x1, y1, _), (x2, y2, _) in zip(enc, dec, strict=True):
        arrow((x1 + bw / 2 + 0.3, y1), (x2 - bw / 2 - 0.3, y2), dashed=True, colour=MUTED)
    arrow((dec[0][0] + bw / 2 + 0.3, 82), (hx - 5.3, 82))
    # ERA5 inputs into the first block
    arrow((ax_ + 20.3, 66), (enc[0][0] - bw / 2 - 0.3, 82))
    note(51.5, 73.5, "4 × 128²")
    note(hx, 76, "residual f$_\\theta$")

    # (b) Feature path -----------------------------------------------------------
    title(49, 35, "(b) Feature path (frozen image encoder)")
    fy, fw, fh = 18.5, 20, 21
    fx = [57, 80.5, 104, 127.5, 151]
    ftexts = [
        "Bilinear rain\ndrawn as RGB\nimage (7-step\ncolour map,\n0–100 mm)",
        "Tiles of 128²,\nstride 32\n(21 per day),\nresized\nto 224²",
        "Frozen ViT-B/32\nOpenAI CLIP or\nrandom weights\n(control)\n→ 7×7×768",
        "PCA to 32\ncomponents (fit\non every 6th\ntraining day),\nstandardised",
        "Each tile → 16²;\noverlapping\ntiles averaged\n→ 32 maps\nof 40 × 23",
    ]
    for x, t in zip(fx, ftexts, strict=True):
        box(x, fy, fw, fh, t, fill=FEATURE, size=4.8)
    for a, b in pairwise(fx):
        arrow((a + fw / 2 + 0.3, fy), (b - fw / 2 - 0.3, fy))
    arrow((ax_ + 20.3, 61), (fx[0] - fw / 2 - 0.3, fy + 4))
    line([(fx[-1], fy + fh / 2 + 0.3), (fx[-1], 40.5), (bx + 4, 40.5)])
    arrow((bx + 4, 40.5), (bx + 4, by - 4.8))
    note(133, 44, "16² window for each 128² patch,\njoined at the bottleneck input")

    # (d) Output and post-processing ---------------------------------------------
    title(162, 97, "(d) Output, evaluation")
    dx, dw = 181, 36
    box(dx, 82, dw, 9, "$\\hat r = \\exp(\\log(1 + r_{\\mathrm{bil}}) + f_\\theta) - 1$", size=5)
    box(
        dx,
        64,
        dw,
        16,
        "Quantile mapping per\nland cell and month\n"
        "(100 quantiles), fitted on\nthe U-Net's 2022–23 output",
    )
    box(dx, 46, dw, 10, "Downscaled daily rain\n0.05° (320 × 180)", fill=ACCENT, bold=True)
    box(
        dx,
        21,
        dw,
        25,
        "Evaluation over land:\nMAE, bias, Pearson r,\n"
        "CSI at 20 and 50 mm/day;\nThailand and Bangkok;\nvalidation 2024,\n"
        "test 2025 (held out);\n3 seeds (mean, range)",
        fill=NET,
    )
    arrow((hx + 5.3, 82), (dx - dw / 2 - 0.3, 82))
    arrow((dx, 77.2), (dx, 72.3))
    arrow((dx, 55.7), (dx, 51.3))
    arrow((dx, 40.7), (dx, 33.8))
    out = OUT / "workflow_detailed.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def workflow_simple() -> Path:
    """The first, compact version: two rows of boxes (main path and feature path)."""
    fig, ax = plt.subplots(figsize=(7.1, 2.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    w, h = 11.0, 8.4

    def box(x, y, text, fill=PLAIN):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                fc=fill,
                ec=INK,
                lw=0.6,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=5.3, color=INK, linespacing=1.25)

    def arrow(a, b):
        ax.add_patch(
            FancyArrowPatch(
                a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK, shrinkA=0, shrinkB=0
            )
        )

    top, bottom = 22.5, 7.0
    xs = [6.5, 19.0, 31.5, 44.0, 56.5, 69.0, 81.5, 94.0]
    texts = [
        "ERA5 (0.25°)\nrain, vapour,\nwind 850 hPa",
        "Bilinear\nto 0.05°",
        "U-Net\nencoder\n(3 levels)",
        "Bottleneck\n+ 32 feature\nmaps",
        "U-Net\ndecoder\n(log residual)",
        "+ log(1 +\nbilinear rain)",
        "Quantile\nmapping\n(cell, month)",
        "Rain at 0.05°\n(320 × 180)",
    ]
    for i, (x, t) in enumerate(zip(xs, texts, strict=True)):
        box(x, top, t, fill=ACCENT if i == 3 else PLAIN)
    for a, b in pairwise(xs):
        arrow((a + w / 2 + 0.3, top), (b - w / 2 - 0.3, top))
    fx = [19.0, 31.5, 44.0]
    ftexts = [
        "Rain map as\nimage; 128²\ntiles, stride 32",
        "Frozen ViT-B/32\n(CLIP or\nrandom weights)",
        "Patch tokens\n→ PCA 32 →\n40×23 maps",
    ]
    for x, t in zip(fx, ftexts, strict=True):
        box(x, bottom, t, fill=FEATURE)
    arrow((19.0, top - h / 2 - 0.3), (19.0, bottom + h / 2 + 0.3))
    for a, b in pairwise(fx):
        arrow((a + w / 2 + 0.3, bottom), (b - w / 2 - 0.3, bottom))
    arrow((44.0, bottom + h / 2 + 0.3), (44.0, top - h / 2 - 0.3))
    ax.text(
        57.0,
        bottom + 2.6,
        "CLIP-input path (not used by the plain U-Net)",
        fontsize=5.5,
        color=MUTED,
        ha="left",
        va="center",
    )
    ax.text(
        57.0,
        bottom - 1.0,
        "Training: MSE in log space against CHIRPS on 128² land patches;\n"
        "loss variants add a heavy-rain count loss or a CLIP loss (MRA).",
        fontsize=5.5,
        color=MUTED,
        ha="left",
        va="center",
        linespacing=1.3,
    )
    out = OUT / "workflow_simple.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    if "workflow" not in sys.argv[1:]:  # `... workflow` redraws only the diagrams
        print(f"saved {maps()}", flush=True)
    print(f"saved {workflow()}", flush=True)
    print(f"saved {workflow(detail=True)}", flush=True)
    print(f"saved {workflow_detailed()}", flush=True)
    print(f"saved {workflow_simple()}", flush=True)
