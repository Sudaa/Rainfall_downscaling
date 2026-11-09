"""Repeat runs with other seeds: names, saved settings and the summary over seeds."""

import pandas as pd

from rainfall_downscaling_baseline.cli import base_model, seed_suffix, summarise_seeds


def test_seed_names():
    assert seed_suffix(0) == "" and seed_suffix(0, "_") == ""
    assert seed_suffix(2) == " (seed 2)" and seed_suffix(2, "_") == "_seed2"
    assert base_model("unet+qm (seed 2)") == "unet+qm"
    assert base_model("unet+qm") == "unet+qm"


def _table(models, csi):
    rows = [
        {
            "region": "all",
            "model": m,
            "mae": 1.0,
            "bias": 0.0,
            "corr": 0.5,
            "csi_20mm": c,
            "csi_50mm": c / 2,
        }
        for m, c in zip(models, csi, strict=True)
    ]
    return pd.DataFrame(rows).set_index(["region", "model"])


def test_summary_averages_over_seeds(tmp_path):
    out = tmp_path / "seeds.csv"
    table = _table(["unet", "unet (seed 1)", "unet (seed 2)", "bilinear"], [0.1, 0.2, 0.3, 0.4])
    summarise_seeds(table, out)
    summary = pd.read_csv(out).set_index("base")
    assert summary.loc["unet", "csi_20mm"] == 0.2
    assert summary.loc["unet", "csi_20mm_range"] == 0.2
    assert summary.loc["unet", "runs"] == 3
    assert summary.loc["bilinear", "runs"] == 1


def test_no_summary_without_repeat_runs(tmp_path):
    out = tmp_path / "seeds.csv"
    summarise_seeds(_table(["unet", "bilinear"], [0.1, 0.4]), out)
    assert not out.exists()


def test_seed_and_patience_are_saved(tmp_path):
    from test_unet import _toy_pair

    from rainfall_downscaling_baseline.models.unet import UNetDownscaler

    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(
        val=val, epochs=1, batch_size=4, patch_size=16, base=4, seed=3, patience=9
    )
    model.fit(*train)
    model.save(tmp_path / "m.pt")
    loaded = UNetDownscaler.load(tmp_path / "m.pt")
    assert (loaded.seed, loaded.patience) == (3, 9)
