"""Does off-the-shelf CLIP understand CHIRPS rain maps?

1. Zero-shot: pick the best of the 4 prompts for real crops; compare with the true word.
2. Intensity: scale a crop's rain by 0.25..4; does similarity to "a lot of" rise?
3. Linear information: nearest-centroid classifier on CLIP image features,
   fitted on 2022-23 crops, tested on 2024 crops.

Run alone (it loads CLIP and needs a few GB of memory):
    uv run python notebooks/02_clip_diagnostic.py
"""

import numpy as np
import torch
import torch.nn.functional as F

from rainfall_downscaling_baseline.config import load_config
from rainfall_downscaling_baseline.models.clip_loss import (
    WORDS,
    ClipAlignmentLoss,
    describe,
)
from rainfall_downscaling_baseline.preprocessing.pairs import land_mask, load_split

rng = np.random.default_rng(0)
cfg = load_config()
clip = ClipAlignmentLoss(mode="text")
SIZE, N = 128, 300


def sample_crops(fine):
    """N random 128x128 crops with at least half land, as (rain, land) tensors."""
    land = land_mask(fine).values.astype(np.float32)
    rain = np.nan_to_num(fine.values.astype(np.float32))
    crops, masks = [], []
    while len(crops) < N:
        d = rng.integers(len(rain))
        y, x = rng.integers(0, land.shape[0] - SIZE), rng.integers(0, land.shape[1] - SIZE)
        m = land[y : y + SIZE, x : x + SIZE]
        if m.mean() >= 0.5:
            crops.append(rain[d, y : y + SIZE, x : x + SIZE])
            masks.append(m)
    return torch.tensor(np.stack(crops)), torch.tensor(np.stack(masks))


@torch.no_grad()
def embed(rain, land):
    out = []
    for i in range(0, len(rain), 32):
        out.append(clip._embed(rain[i : i + 32], land[i : i + 32]))
    return torch.cat(out)


train_rain, train_land = sample_crops(load_split(cfg, "train")[1])
val_rain, val_land = sample_crops(load_split(cfg, "val")[1])
train_word = describe(train_rain, train_land, 20.0)
val_word = describe(val_rain, val_land, 20.0)
train_emb, val_emb = embed(train_rain, train_land), embed(val_rain, val_land)

counts = torch.bincount(val_word, minlength=4).tolist()
print("2024 crops per true word:", dict(zip(WORDS, counts, strict=True)))
print(f"always guessing the most common word: {max(counts) / N:.0%}")

# 1. Zero-shot
guess = (val_emb @ clip.text.T).argmax(1)
print(f"\n1. zero-shot accuracy: {(guess == val_word).float().mean():.0%} (chance 25%)")
confusion = torch.zeros(4, 4, dtype=torch.int64)
for t, g in zip(val_word, guess, strict=True):
    confusion[t, g] += 1
print("   rows = true word, columns = CLIP's choice", list(WORDS))
for w, row in zip(WORDS, confusion.tolist(), strict=True):
    print(f"   {w:9s} {row}")

# 2. Intensity: same crops, rain scaled
wet = val_word >= 2
rain, land = val_rain[wet][:40], val_land[wet][:40]
print("\n2. similarity to 'a lot of' when the same wet crops are scaled:")
for factor in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
    sim = (embed(rain * factor, land) @ clip.text[3]).mean()
    share = describe(rain * factor, land, 20.0).float().mean()
    print(f"   rain x {factor:<4}: sim {sim:.4f}   (mean true word index {share:.2f})")

# 3. Nearest centroid on image features (train 2022-23 -> test 2024)
centroids = torch.stack(
    [
        F.normalize(train_emb[train_word == k].mean(0), dim=-1)
        if (train_word == k).any()
        else torch.zeros(train_emb.shape[1])
        for k in range(4)
    ]
)
probe = (val_emb @ centroids.T).argmax(1)
print(f"\n3. image-feature classifier accuracy on 2024: {(probe == val_word).float().mean():.0%}")
