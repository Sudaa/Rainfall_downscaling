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


def cmd_train(cfg: Config) -> None:
    """Train the U-Net on the training years, stopping early on the validation year.

    The baselines train in seconds, so evaluate fits them itself; only the
    U-Net is trained here and saved for evaluate to load.
    """
    from .models.unet import UNetDownscaler
    from .preprocessing.pairs import load_split

    model = UNetDownscaler(val=load_split(cfg, "val"))
    model.fit(*load_split(cfg, "train"))
    out = cfg.paths.checkpoints / "unet.pt"
    model.save(out)
    print(f"saved {out}")


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
    checkpoint = cfg.paths.checkpoints / "unet.pt"
    if checkpoint.exists():
        from .models.unet import UNetDownscaler

        predictions["unet"] = UNetDownscaler.load(checkpoint).predict(coarse)
    else:
        print(f"no {checkpoint}: run `rainfall train` to include the U-Net")

    table = evaluate(predictions, obs, land_mask(obs), cfg.eval_regions)
    print(f"Scores on {split} ({obs.sizes['time']} days, land only)")
    print(table.round(3).to_string())
    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    out = cfg.paths.figures / f"scores_{split}.csv"
    table.to_csv(out)
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
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "evaluate":
        cmd_evaluate(cfg, args.split)
    else:
        COMMANDS[args.command](cfg)
