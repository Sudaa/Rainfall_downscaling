"""CLIP regional alignment loss (MRA), adapted from Clip4Rain for downscaling.

For centred square crops of three sizes, the predicted and the observed rain
are drawn as images with a fixed colour map and passed through a frozen CLIP
image encoder. Each crop is described in words by how much of it has heavy
rain in CHIRPS ("There is little heavy rain in the image."). The loss makes
the prediction match that sentence as closely as the observed crop does, as in
Clip4Rain's multi-level regional alignment, where the prompt is filled in from
the real frame.

mode="image" is the ablation without text: it compares the CLIP image features
of prediction and observation directly.

CLIP is frozen and only used inside the training loss; predictions never need it.
"""

import torch
import torch.nn.functional as F

# The blue ramp of the report figures (evaluation/plots.py), from 0 to VMAX mm/day
COLOURS = (
    torch.tensor(
        [
            [255, 255, 255],
            [205, 226, 251],
            [134, 182, 239],
            [57, 135, 229],
            [37, 106, 191],
            [24, 79, 149],
            [13, 54, 107],
        ],
        dtype=torch.float32,
    )
    / 255
)
SEA = torch.tensor([232, 231, 227], dtype=torch.float32) / 255
VMAX = 100.0
# CLIP was trained on images normalised with these values
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

WORDS = ("no", "little", "some", "a lot of")
PROMPT = "There is {} heavy rain in the image."
# Upper bounds of the share of land pixels above the threshold for each word:
# 0 -> "no", up to 5% -> "little", up to 20% -> "some", more -> "a lot of"
SHARE_BOUNDS = torch.tensor([0.0, 0.05, 0.20])


def rain_to_image(rain: torch.Tensor, land: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Rain in mm, shape (batch, lat, lon), as CLIP input images, shape (batch, 3, size, size).

    Colours are interpolated linearly between the ramp's steps, so the image
    changes smoothly with the rain and gradients can flow back to the U-Net.
    """
    position = (rain / VMAX).clamp(0, 1) * (len(COLOURS) - 1)
    low = position.floor().long().clamp(max=len(COLOURS) - 2)
    frac = (position - low).unsqueeze(-1)
    rgb = COLOURS[low] * (1 - frac) + COLOURS[low + 1] * frac  # (batch, lat, lon, 3)
    rgb = torch.where(land.unsqueeze(-1) > 0, rgb, SEA)
    image = F.interpolate(rgb.permute(0, 3, 1, 2), size=(size, size), mode="bilinear")
    return (image - CLIP_MEAN) / CLIP_STD


def describe(true_rain: torch.Tensor, land: torch.Tensor, threshold: float) -> torch.Tensor:
    """Index into WORDS for each observed crop, from its share of land pixels above threshold."""
    share = ((true_rain >= threshold) * land).sum((-2, -1)) / land.sum((-2, -1)).clamp(min=1)
    return torch.bucketize(share, SHARE_BOUNDS)


class ClipAlignmentLoss:
    def __init__(
        self,
        mode: str = "text",
        threshold: float = 20.0,
        levels: tuple[int, ...] = (1, 2, 4),
        model=None,
        tokenizer=None,
    ) -> None:
        """`model`/`tokenizer` default to OpenAI's CLIP ViT-B/32 (downloaded once, ~580 MB).
        Tests pass a small stand-in instead."""
        if mode not in ("text", "image"):
            raise ValueError(f"mode must be 'text' or 'image', got {mode!r}")
        if model is None:
            import open_clip

            # OpenAI's weights were trained with the QuickGELU activation; the
            # plain "ViT-B-32" config would run them with a different one.
            name = "ViT-B-32-quickgelu"
            model, _, _ = open_clip.create_model_and_transforms(name, pretrained="openai")
            tokenizer = open_clip.get_tokenizer(name)
        self.model = model.eval().requires_grad_(False)  # frozen: CLIP itself never changes
        self.mode, self.threshold, self.levels = mode, threshold, levels
        with torch.no_grad():
            text = self.model.encode_text(tokenizer([PROMPT.format(w) for w in WORDS]))
        self.text = F.normalize(text, dim=-1)  # (words, features)

    def _embed(self, rain: torch.Tensor, land: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model.encode_image(rain_to_image(rain, land)), dim=-1)

    def __call__(
        self, pred_rain: torch.Tensor, true_rain: torch.Tensor, land: torch.Tensor
    ) -> torch.Tensor:
        """Mean alignment gap over crop sizes; inputs in mm, shape (batch, lat, lon)."""
        land = land.expand_as(true_rain)
        height, width = true_rain.shape[-2:]
        gaps = []
        for level in self.levels:
            side = min(height, width) // level
            top, left = (height - side) // 2, (width - side) // 2
            crop = (..., slice(top, top + side), slice(left, left + side))
            pred_emb = self._embed(pred_rain[crop], land[crop])
            with torch.no_grad():  # the observation is fixed: no gradient needed
                true_emb = self._embed(true_rain[crop], land[crop])
            if self.mode == "image":
                gaps.append((1 - (pred_emb * true_emb).sum(-1)).mean())
            else:
                text = self.text[describe(true_rain[crop], land[crop], self.threshold)]
                pred_sim, true_sim = (pred_emb * text).sum(-1), (true_emb * text).sum(-1)
                gaps.append((pred_sim - true_sim).abs().mean())
        return torch.stack(gaps).mean()
