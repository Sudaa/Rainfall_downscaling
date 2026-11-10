"""Command-line entry point: `uv run rainfall <command>`.

Each command only loads the config and calls package functions. Keeping the
logic out of this file means the same functions can be called from tests and
notebooks. Imports sit inside each command so a command only needs the
libraries it actually uses.
"""

import argparse

from .config import DEFAULT_CONFIG, Config, load_config


def cmd_config(cfg: Config) -> None:
    """Print the validated settings. Useful to check what a run will use."""
    print(cfg.model_dump_json(indent=2))
    print(f"scale factor: {cfg.scale_factor}")


def cmd_download_era5(cfg: Config) -> None:
    from .data import era5

    era5.download(cfg, cfg.period.all_years)


def cmd_download_chirps(cfg: Config) -> None:
    from .data import chirps

    for year in cfg.period.all_years:
        chirps.download(cfg, year)


def cmd_download(cfg: Config) -> None:
    cmd_download_era5(cfg)
    cmd_download_chirps(cfg)


def cmd_prepare(cfg: Config) -> None:
    # Transforms (log, normalisation) are fitted later by the models that need
    # them, on the training split only; prepare saves plain mm/day.
    from .preprocessing.pairs import build_pairs, save_split
    from .preprocessing.splits import split_by_year

    coarse, fine = build_pairs(cfg, cfg.period.all_years)
    coarse_splits = split_by_year(coarse, cfg.period)
    fine_splits = split_by_year(fine, cfg.period)
    for split in coarse_splits:
        save_split(cfg, split, coarse_splits[split], fine_splits[split])
        print(f"{split}: {coarse_splits[split].sizes['time']} days")
    print(f"saved to {cfg.paths.processed}")


def cmd_train(
    cfg: Config,
    aux: str = "none",
    aux_weight: float = 0.0,
    clip_input: bool = False,
    seed: int = 0,
    patience: int = 5,
    clip_random: bool = False,
) -> None:
    """Train the U-Net on the training years, stopping early on the validation year.

    The baselines train in seconds, so evaluate fits them itself; only the
    U-Net is trained here and saved for evaluate to load. With `aux` it adds a
    second loss term (count, clip or clip-image) and is saved under its own name.
    Seeds other than 0 are repeat runs, saved with `_seed<n>` added to the name.
    """
    from .models.unet import UNetDownscaler
    from .preprocessing.pairs import load_split

    model = UNetDownscaler(
        val=load_split(cfg, "val"),
        aux=aux,
        aux_weight=aux_weight,
        clip_input=clip_input,
        clip_random=clip_random,
        seed=seed,
        patience=patience,
    )
    model.fit(*load_split(cfg, "train"))
    out = cfg.paths.checkpoints / f"{model.name.replace('+', '_')}{seed_suffix(seed, '_')}.pt"
    model.save(out)
    print(f"saved {out}")


def seed_suffix(seed: int, style: str = " ") -> str:
    """'' for seed 0 (the original runs), else ' (seed n)' in tables or '_seed<n>' in files."""
    if seed == 0:
        return ""
    return f" (seed {seed})" if style == " " else f"_seed{seed}"


def base_model(name: str) -> str:
    """Model name without its seed: 'unet+qm (seed 2)' -> 'unet+qm'."""
    return name.split(" (seed ")[0]


# U-Nets that also get quantile mapping on top at evaluation
QM_ON_TOP = ("unet", "unet+clip-input", "unet+clip-input-random")


def cmd_evaluate(cfg: Config, split: str = "val") -> None:
    """Fit every model on train, predict `split`, score over land, save the table.

    Use the validation split while developing; score the test split once, at the end.
    """
    from .evaluation.evaluate import evaluate
    from .models.bilinear import BilinearDownscaler
    from .models.quantile_mapping import QuantileMappingDownscaler
    from .preprocessing.pairs import land_mask, load_split

    train_coarse, train_fine = load_split(cfg, "train")
    coarse, obs = load_split(cfg, split)
    predictions = {}
    for model in [BilinearDownscaler(), QuantileMappingDownscaler()]:
        model.fit(train_coarse, train_fine)
        predictions[model.name] = model.predict(coarse)
    checkpoints = sorted(cfg.paths.checkpoints.glob("unet*.pt"))
    if not checkpoints:
        print(f"no U-Net in {cfg.paths.checkpoints}: run `rainfall train` to include it")
    for checkpoint in checkpoints:
        from .models.unet import UNetDownscaler

        unet = UNetDownscaler.load(checkpoint)
        suffix = seed_suffix(unet.seed)
        print(f"predicting with {checkpoint.name} ...", flush=True)
        predictions[unet.name + suffix] = unet.predict(coarse)
        if unet.name in QM_ON_TOP:
            # Quantile mapping on top of the U-Net, fitted on its training-year output
            unet_qm = QuantileMappingDownscaler(base=unet)
            unet_qm.fit(train_coarse, train_fine)
            predictions[unet_qm.name + suffix] = unet_qm.predict(coarse)

    table = evaluate(predictions, obs, land_mask(obs), cfg.eval_regions)
    print(f"Scores on {split} ({obs.sizes['time']} days, land only)")
    print(table.round(3).to_string())
    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures / f"scores_{split}.csv"
    table.to_csv(out)
    print(f"saved {out}")
    summarise_seeds(table, cfg.paths.figures / f"scores_{split}_seeds.csv")

    from .evaluation.plots import FIGURE_SKIP, exceedance, map_panel, wettest_day

    land = land_mask(obs)
    day = wettest_day(obs)
    # Figures hold at most 8 models (one fixed colour each) and only the seed-0
    # runs; the score table has all of them
    shown = {k: v for k, v in predictions.items() if k not in FIGURE_SKIP and base_model(k) == k}
    for path in (
        map_panel(coarse["precip"], shown, obs, land, day, cfg.paths.figures / f"maps_{split}.png"),
        exceedance(shown, obs, land, cfg.paths.figures / f"exceedance_{split}.png"),
    ):
        print(f"saved {path}")


def summarise_seeds(table, out) -> None:
    """Mean, and spread (min to max), of each score over the seeds of each model.

    Only printed when some model has repeat runs: one run can win or lose by
    luck, the mean over seeds is what a comparison should rest on.
    """
    table = table.reset_index()
    table["base"] = table["model"].map(base_model)
    runs = table.groupby(["region", "base"])["model"].count()
    if runs.max() <= 1:
        return
    scores = ["mae", "bias", "corr", "csi_20mm", "csi_50mm"]
    grouped = table.groupby(["region", "base"])[scores]
    summary = grouped.mean().round(3)
    spread = (grouped.max() - grouped.min()).round(3).add_suffix("_range")
    summary = summary.join(spread).join(runs.rename("runs"))
    print("\nMean over seeds (and range = max - min):")
    print(summary.to_string())
    summary.to_csv(out)
    print(f"saved {out}")


COMMANDS = {
    "config": cmd_config,
    "download": cmd_download,
    "download-era5": cmd_download_era5,
    "download-chirps": cmd_download_chirps,
    "prepare": cmd_prepare,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rainfall", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to a TOML config")
    parser.add_argument(
        "--split", choices=["val", "test"], default="val", help="split to score (evaluate only)"
    )
    parser.add_argument(
        "--aux",
        choices=["none", "count", "clip", "clip-image"],
        default="none",
        help="extra loss term for the U-Net (train only)",
    )
    parser.add_argument(
        "--aux-weight", type=float, default=0.0, help="lambda of the extra loss (train only)"
    )
    parser.add_argument(
        "--clip-input",
        action="store_true",
        help="feed CLIP patch features of the ERA5 rain map into the U-Net (train only)",
    )
    parser.add_argument(
        "--clip-random",
        action="store_true",
        help="control: like --clip-input, but the image network has random weights (train only)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed; not 0 = a repeat run (train only)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="epochs without validation improvement before stopping (train only)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "evaluate":
        cmd_evaluate(cfg, args.split)
    elif args.command == "train":
        cmd_train(
            cfg,
            args.aux,
            args.aux_weight,
            args.clip_input,
            args.seed,
            args.patience,
            args.clip_random,
        )
    else:
        COMMANDS[args.command](cfg)
