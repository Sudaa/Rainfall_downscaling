"""Tests for the U-Net: shapes, the land-only loss, and a tiny training run end to end."""

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from rainfall_downscaling_baseline.models.unet import (
    UNet,
    UNetDownscaler,
    _count_loss,
    _masked_mse,
    _soft_share,
)


def test_unet_output_matches_input_size_even_when_not_a_multiple_of_8():
    net = UNet(in_channels=4, base=4)
    out = net(torch.zeros(2, 4, 37, 45))
    assert out.shape == (2, 37, 45)


def test_masked_mse_ignores_the_sea():
    pred, target = torch.zeros(1, 2, 2), torch.tensor([[[1.0, 1.0], [1.0, 99.0]]])
    land = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # last cell is sea
    assert _masked_mse(pred, target, land).item() == pytest.approx(1.0)


def _toy_pair(days: int, seed: int) -> tuple[xr.Dataset, xr.DataArray]:
    """Coarse 5x5 ERA5-like fields over 1 degree, and fine 'observed' rain on 20x20 cells."""
    rng = np.random.default_rng(seed)
    time = pd.date_range("2022-05-01", periods=days)
    lat, lon = np.arange(10.0, 11.01, 0.25), np.arange(100.0, 101.01, 0.25)
    shape = (days, len(lat), len(lon))
    coarse = xr.Dataset(
        {var: (("time", "lat", "lon"), rng.gamma(1.0, 5.0, shape)) for var in ("precip", "tcwv")}
        | {var: (("time", "lat", "lon"), rng.normal(0, 3, shape)) for var in ("u", "v")},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    fine_lat, fine_lon = np.arange(10.025, 11.0, 0.05), np.arange(100.025, 101.0, 0.05)
    fine = 1.5 * coarse["precip"].interp(lat=fine_lat, lon=fine_lon)
    fine[:, :3, :3] = np.nan  # a corner of sea
    return coarse, fine.rename("precip")


def test_fit_predict_save_load_end_to_end(tmp_path):
    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(val=val, epochs=2, batch_size=4, patch_size=16, base=4)
    model.fit(*train)
    pred = model.predict(val[0])

    assert pred.shape == val[1].shape
    assert float(pred.min()) >= 0.0  # rain is never negative
    assert len(model.history) == 2

    model.save(tmp_path / "unet.pt")
    loaded = UNetDownscaler.load(tmp_path / "unet.pt")
    assert np.allclose(loaded.predict(val[0]), pred, atol=1e-5)


def test_soft_share_counts_pixels_above_threshold():
    rain = torch.tensor([[[0.0, 5.0], [60.0, 100.0]]])  # 2 of 4 pixels are far above 20 mm
    land = torch.ones(2, 2)
    assert _soft_share(rain, land, 20.0, 2.0).item() == pytest.approx(0.5, abs=1e-3)


def test_count_loss_is_zero_for_identical_fields_and_positive_for_too_dry():
    true = torch.full((2, 16, 16), 40.0)
    land = torch.ones(16, 16)
    assert _count_loss(true, true, land, (20.0,)).item() == pytest.approx(0.0)
    dry = torch.full((2, 16, 16), 5.0, requires_grad=True)
    loss = _count_loss(dry, true, land, (20.0,))
    loss.backward()
    assert loss.item() > 0.9
    assert dry.grad.abs().sum() > 0  # the soft threshold gives a gradient to learn from


def test_count_loss_model_trains_saves_and_loads(tmp_path):
    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(
        val=val, epochs=1, batch_size=4, patch_size=16, base=4, aux="count", aux_weight=5.0
    )
    model.fit(*train)
    assert model.name == "unet+count"
    assert "val_aux" in model.history[0]
    model.save(tmp_path / "m.pt")
    loaded = UNetDownscaler.load(tmp_path / "m.pt")
    assert loaded.name == "unet+count" and loaded.aux_weight == 5.0
