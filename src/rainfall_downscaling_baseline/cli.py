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
    # TODO: load both datasets, convert ERA5 to daily, crop, check alignment,
    # split by year, fit transforms on train, save arrays to cfg.paths.processed
    raise NotImplementedError


def cmd_train(cfg: Config) -> None:
    # TODO: fit each model in models/ on the training split, save to cfg.paths.checkpoints
    raise NotImplementedError


def cmd_evaluate(cfg: Config) -> None:
    # TODO: predict the test split with every model, score with evaluation.evaluate,
    # print the table and save figures to cfg.paths.figures
    raise NotImplementedError


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
    args = parser.parse_args(argv)
    COMMANDS[args.command](load_config(args.config))
