"""Tests for the CLIP alignment loss, with a tiny stand-in for CLIP so they run offline and fast."""

import pytest
import torch
from torch import nn

from rainfall_downscaling_baseline.models import clip_loss
from rainfall_downscaling_baseline.models.clip_loss import (
    CLIP_MEAN,
    CLIP_STD,
    SEA,
    ClipAlignmentLoss,
    describe,
    rain_to_image,
)


class FakeClip(nn.Module):
    """Same interface as CLIP: encode_image and encode_text give feature vectors."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.image = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(48, 8))
        self.words = nn.Embedding(4, 8)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image(images)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.words(tokens)


def fake_tokenizer(prompts: list[str]) -> torch.Tensor:
    return torch.arange(len(prompts))


def fake_loss(mode: str = "text") -> ClipAlignmentLoss:
    return ClipAlignmentLoss(mode=mode, model=FakeClip(), tokenizer=fake_tokenizer)


def test_rain_to_image_colours_land_and_sea():
    rain = torch.zeros(1, 4, 4)
    land = torch.ones(4, 4)
    land[0, 0] = 0  # one sea pixel
    image = rain_to_image(rain, land, size=4) * CLIP_STD + CLIP_MEAN  # undo CLIP normalisation
    assert image.shape == (1, 3, 4, 4)
    assert torch.allclose(image[0, :, 3, 3], torch.ones(3), atol=1e-5)  # 0 mm on land: white
    assert torch.allclose(image[0, :, 0, 0], SEA, atol=1e-5)  # sea: grey


def test_rain_to_image_passes_gradients():
    rain = torch.full((1, 8, 8), 30.0, requires_grad=True)
    rain_to_image(rain, torch.ones(8, 8), size=8).sum().backward()
    assert rain.grad.abs().sum() > 0


def test_describe_picks_words_from_share_of_heavy_rain():
    land = torch.ones(10, 10)
    fields = torch.zeros(4, 10, 10)
    fields[1, 0, :3] = 30  # 3% of pixels above 20 mm -> "little"
    fields[2, :1] = 30  # 10% -> "some"
    fields[3, :5] = 30  # 50% -> "a lot of"
    assert describe(fields, land, 20.0).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", ["text", "image"])
def test_loss_is_zero_for_identical_fields(mode):
    rain = torch.rand(2, 32, 32) * 60
    assert fake_loss(mode)(rain, rain, torch.ones(32, 32)).item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("mode", ["text", "image"])
def test_loss_is_positive_and_trains_the_prediction(mode):
    truth = torch.zeros(2, 32, 32)
    truth[:, 8:24, 8:24] = 60.0  # a storm in the middle
    pred = torch.full((2, 32, 32), 5.0, requires_grad=True)  # smeared drizzle
    loss = fake_loss(mode)(pred, truth, torch.ones(32, 32))
    loss.backward()
    assert loss.item() > 0
    assert pred.grad.abs().sum() > 0  # the U-Net would get a learning signal


def test_clip_stays_frozen():
    loss = fake_loss()
    assert not any(p.requires_grad for p in loss.model.parameters())


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        ClipAlignmentLoss(mode="video", model=FakeClip(), tokenizer=fake_tokenizer)


def test_unet_trains_with_clip_loss(monkeypatch, tmp_path):
    from test_unet import _toy_pair

    from rainfall_downscaling_baseline.models.unet import UNetDownscaler

    # Use the stand-in instead of downloading the real CLIP
    monkeypatch.setattr(clip_loss, "ClipAlignmentLoss", lambda mode: fake_loss(mode))
    train, val = _toy_pair(16, seed=1), _toy_pair(8, seed=2)
    model = UNetDownscaler(
        val=val, epochs=1, batch_size=4, patch_size=16, base=4, aux="clip", aux_weight=1.0
    )
    model.fit(*train)
    assert model.name == "unet+clip"
    model.save(tmp_path / "m.pt")
    assert UNetDownscaler.load(tmp_path / "m.pt").aux == "clip"  # loading needs no CLIP
