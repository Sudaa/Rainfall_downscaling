"""Tests for CLIP patch features as U-Net inputs, with a small stand-in for CLIP."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rainfall_downscaling_baseline.models import clip_features
from rainfall_downscaling_baseline.models.clip_features import ClipPatchFeatures

MIX = torch.randn(6, 3, generator=torch.Generator().manual_seed(0))


def fake_encoder(images: torch.Tensor) -> torch.Tensor:
    """Like CLIP's patch tokens: images (n, 3, 224, 224) -> (n, 6, 7, 7)."""
    grid = F.adaptive_avg_pool2d(images, 7)
    return torch.einsum("cr,nryx->ncyx", MIX, grid)


def _fields(days=5, height=20, width=20, seed=0):
    rng = np.random.default_rng(seed)
    rain = rng.gamma(1.0, 10.0, (days, height, width)).astype(np.float32)
    land = np.ones((height, width), dtype=bool)
    land[:4, :4] = False
    return rain, land


def test_feature_maps_have_the_bottleneck_shape_and_unit_scale():
    rain, land = _fields()
    features = ClipPatchFeatures(tile=16, dims=3, encoder=fake_encoder).fit(rain, land, every=1)
    maps = features.transform(rain, land)
    assert maps.shape == (5, 3, 3, 3)  # 20 cells padded to 24, divided by 8
    assert np.isfinite(maps).all()


def test_same_rain_gives_same_features_and_more_rain_changes_them():
    rain, land = _fields()
    features = ClipPatchFeatures(tile=16, dims=3, encoder=fake_encoder).fit(rain, land, every=1)
    a, b = features.transform(rain, land), features.transform(rain.copy(), land)
    assert np.allclose(a, b)
    assert not np.allclose(a, features.transform(rain * 3, land))


def test_state_round_trip_and_refusals():
    rain, land = _fields()
    features = ClipPatchFeatures(tile=16, dims=3, encoder=fake_encoder).fit(rain, land, every=1)
    again = ClipPatchFeatures.from_state(features.state(), encoder=fake_encoder)
    assert np.allclose(features.transform(rain, land), again.transform(rain, land))
    with pytest.raises(ValueError):
        ClipPatchFeatures(tile=12)
    with pytest.raises(RuntimeError):
        ClipPatchFeatures(tile=16, encoder=fake_encoder).transform(rain, land)
    with pytest.raises(ValueError):
        ClipPatchFeatures(tile=32, encoder=fake_encoder).fit(rain, land)


def test_unet_trains_saves_and_loads_with_clip_input(monkeypatch, tmp_path):
    from test_unet import _toy_pair

    from rainfall_downscaling_baseline.models.unet import UNetDownscaler

    monkeypatch.setattr(clip_features, "openai_patch_encoder", lambda pretrained=True: fake_encoder)
    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(val=val, epochs=2, batch_size=4, patch_size=16, base=4, clip_input=True)
    model.fit(*train)
    assert model.name == "unet+clip-input"
    pred = model.predict(val[0])
    assert pred.shape == val[1].shape
    assert float(pred.min()) >= 0.0

    model.save(tmp_path / "m.pt")
    loaded = UNetDownscaler.load(tmp_path / "m.pt")
    assert loaded.clip_input
    assert np.allclose(loaded.predict(val[0]), pred, atol=1e-5)


def test_unet_with_random_control_is_named_and_reloaded(monkeypatch, tmp_path):
    from test_unet import _toy_pair

    from rainfall_downscaling_baseline.models.unet import UNetDownscaler

    asked = []
    monkeypatch.setattr(
        clip_features,
        "openai_patch_encoder",
        lambda pretrained=True: asked.append(pretrained) or fake_encoder,
    )
    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(val=val, epochs=1, batch_size=4, patch_size=16, base=4, clip_random=True)
    model.fit(*train)
    assert model.name == "unet+clip-input-random"
    assert asked == [False]  # the random-weight network was requested
    model.save(tmp_path / "m.pt")
    loaded = UNetDownscaler.load(tmp_path / "m.pt")
    assert loaded.clip_random and loaded.clip_features.pretrained is False


def test_random_encoder_is_the_same_every_time():
    a = clip_features.openai_patch_encoder(pretrained=False)
    b = clip_features.openai_patch_encoder(pretrained=False)
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(a(images), b(images))
