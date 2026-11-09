"""Do CLIP features of ERA5 maps carry information about CHIRPS heavy rain?

A cheap check before building a CLIP-conditioned U-Net. For random 128x128
crops we predict one number, the share of land cells with at least 20 mm in
CHIRPS, from three feature sets computed from ERA5 only (CHIRPS is never an
input, as it would not exist when downscaling):

  A. ERA5 summaries: mean, std, 90th percentile and max of each of the four
     U-Net inputs over land, plus the share of land >= 10 / 20 mm in bilinear rain
  B. A + CLIP image features of the ERA5 rain map
  C. A + CLIP image features of all four ERA5 maps

Each CLIP embedding (512 numbers) is reduced to 16 with PCA. A ridge
regression is fitted on 2022-23 crops (penalty chosen by 5-fold
cross-validation inside the training crops) and scored once on 2024 crops.
If B or C is not clearly better than A, CLIP adds little that ERA5 does not
already give the U-Net.

Run alone (it loads CLIP and takes several minutes on a CPU):
    uv run python notebooks/03_clip_feature_test.py
"""

import numpy as np
import torch

from rainfall_downscaling_baseline.config import load_config
from rainfall_downscaling_baseline.models.clip_loss import (
    SHARE_BOUNDS,
    VMAX,
    WORDS,
    ClipAlignmentLoss,
)
from rainfall_downscaling_baseline.models.unet import INPUTS
from rainfall_downscaling_baseline.preprocessing.pairs import land_mask, load_split

SIZE, N_TRAIN, N_VAL, PCA_DIMS, THRESHOLD = 128, 1000, 500, 16, 20.0
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
rng = np.random.default_rng(0)
cfg = load_config()
clip = ClipAlignmentLoss(mode="text")


def sample_windows(land, days, n):
    """n random (day, y, x) crop corners with at least half land."""
    windows = []
    while len(windows) < n:
        d = rng.integers(days)
        y, x = rng.integers(0, land.shape[0] - SIZE), rng.integers(0, land.shape[1] - SIZE)
        if land[y : y + SIZE, x : x + SIZE].mean() >= 0.5:
            windows.append((d, y, x))
    return windows


def crop(field, windows):
    return np.stack([field[d, y : y + SIZE, x : x + SIZE] for d, y, x in windows])


def load_crops(split, n):
    """ERA5 crops of each input on the fine grid, CHIRPS crops and land crops."""
    coarse, fine = load_split(cfg, split)
    land = land_mask(fine).values.astype(np.float32)
    windows = sample_windows(land, len(fine.time), n)
    era5 = {}
    for var in INPUTS:  # one variable at a time, to keep memory low
        field = coarse[var].interp(lat=fine.lat, lon=fine.lon, method="linear")
        era5[var] = crop(field.values.astype(np.float32), windows)
        del field
    if "precip" in era5:
        era5["precip"] = era5["precip"].clip(min=0)
    chirps = crop(np.nan_to_num(fine.values.astype(np.float32)), windows)
    return era5, chirps, np.stack([land[y : y + SIZE, x : x + SIZE] for _, y, x in windows])


def land_share(rain, land, threshold):
    return ((rain >= threshold) * land).sum((1, 2)) / land.sum((1, 2))


def summaries(era5, land):
    feats = []
    for var in INPUTS:
        values = [crop_[m > 0] for crop_, m in zip(era5[var], land, strict=True)]
        feats += [
            [v.mean() for v in values],
            [v.std() for v in values],
            [np.percentile(v, 90) for v in values],
            [v.max() for v in values],
        ]
    feats += [land_share(era5["precip"], land, 10.0), land_share(era5["precip"], land, 20.0)]
    return np.array(feats, dtype=np.float64).T


@torch.no_grad()
def clip_features(field, land):
    """512-number CLIP embedding of each crop drawn with the rain colour map."""
    field = torch.tensor(field, dtype=torch.float32)
    land = torch.tensor(land, dtype=torch.float32)
    return torch.cat(
        [clip._embed(field[i : i + 32], land[i : i + 32]) for i in range(0, len(field), 32)]
    ).numpy()


def as_rain_scale(field, low, high):
    """Map a non-rain variable onto 0..VMAX so it uses the same colour map."""
    return ((field - low) / (high - low)).clip(0, 1) * VMAX


class PCA:
    def fit(self, x, k):
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:k].T
        return self

    def transform(self, x):
        return (x - self.mean) @ self.components


def ridge_fit(x, y, alpha):
    x1 = np.c_[np.ones(len(x)), x]
    penalty = alpha * np.eye(x1.shape[1])
    penalty[0, 0] = 0  # do not shrink the intercept
    return np.linalg.solve(x1.T @ x1 + penalty, x1.T @ y)


def ridge_predict(w, x):
    return np.c_[np.ones(len(x)), x] @ w


def fit_and_predict(x_train, y_train, x_val):
    """Standardise, pick the ridge penalty by 5-fold CV on train, refit, predict 2024."""
    mean, std = x_train.mean(0), x_train.std(0) + 1e-9
    x_train, x_val = (x_train - mean) / std, (x_val - mean) / std
    folds = np.arange(len(x_train)) % 5
    rng.shuffle(folds)
    cv = {}
    for alpha in ALPHAS:
        errors = []
        for f in range(5):
            w = ridge_fit(x_train[folds != f], y_train[folds != f], alpha)
            errors.append(
                ((ridge_predict(w, x_train[folds == f]) - y_train[folds == f]) ** 2).mean()
            )
        cv[alpha] = np.mean(errors)
    alpha = min(cv, key=cv.get)
    pred = ridge_predict(ridge_fit(x_train, y_train, alpha), x_val).clip(0, 1)
    return pred, alpha


def to_word(share):
    return np.searchsorted(SHARE_BOUNDS.numpy(), share, side="left")


# --- data -----------------------------------------------------------------
print("loading crops ...")
train_era5, train_chirps, train_land = load_crops("train", N_TRAIN)
val_era5, val_chirps, val_land = load_crops("val", N_VAL)
y_train = land_share(train_chirps, train_land, THRESHOLD)
y_val = land_share(val_chirps, val_land, THRESHOLD)
word_val = to_word(y_val)
counts = np.bincount(word_val, minlength=4)
print(f"{N_TRAIN} train crops (2022-23), {N_VAL} validation crops (2024)")
print("2024 crops per true word:", dict(zip(WORDS, counts.tolist(), strict=True)))
print(
    f"mean CHIRPS share >= {THRESHOLD:.0f} mm: train {y_train.mean():.3f}, 2024 {y_val.mean():.3f}"
)

# --- features ---------------------------------------------------------------
summ_train, summ_val = summaries(train_era5, train_land), summaries(val_era5, val_land)
clip_train, clip_val = {}, {}
for var in INPUTS:
    print(f"CLIP features of ERA5 {var} ...")
    if var == "precip":
        tr, va = train_era5[var], val_era5[var]
    else:  # stretch between the 1st and 99th percentile of the training crops
        low, high = np.percentile(train_era5[var][train_land > 0], [1, 99])
        tr, va = as_rain_scale(train_era5[var], low, high), as_rain_scale(val_era5[var], low, high)
    emb_train, emb_val = clip_features(tr, train_land), clip_features(va, val_land)
    pca = PCA().fit(emb_train, PCA_DIMS)
    clip_train[var], clip_val[var] = pca.transform(emb_train), pca.transform(emb_val)

sets = {
    "A  ERA5 summaries": (summ_train, summ_val),
    "B  A + CLIP(rain)": (
        np.c_[summ_train, clip_train["precip"]],
        np.c_[summ_val, clip_val["precip"]],
    ),
    "C  A + CLIP(all 4)": (
        np.c_[summ_train, *[clip_train[v] for v in INPUTS]],
        np.c_[summ_val, *[clip_val[v] for v in INPUTS]],
    ),
    "D  CLIP(rain) only": (clip_train["precip"], clip_val["precip"]),
}


# --- scores on 2024 -----------------------------------------------------------
def scores(pred):
    rmse = np.sqrt(((pred - y_val) ** 2).mean())
    r2 = 1 - ((pred - y_val) ** 2).sum() / ((y_val - y_val.mean()) ** 2).sum()
    return rmse, r2, np.corrcoef(pred, y_val)[0, 1], (to_word(pred) == word_val).mean()


preds = {"0  training mean": np.full(N_VAL, y_train.mean())}
alphas = {}
for name, (x_tr, x_va) in sets.items():
    preds[name], alphas[name] = fit_and_predict(x_tr, y_train, x_va)

print(f"\nPredicting the CHIRPS share of land >= {THRESHOLD:.0f} mm on 2024 crops")
print(f"{'features':22s} {'n':>4s} {'RMSE':>7s} {'R2':>6s} {'corr':>6s} {'word acc':>9s}")
for name, pred in preds.items():
    n = 0 if name.startswith("0") else sets[name][0].shape[1]
    rmse, r2, corr, acc = scores(pred)
    print(f"{name:22s} {n:4d} {rmse:7.4f} {r2:6.3f} {corr:6.3f} {acc:9.0%}")
print(f"(always guessing the most common word: {counts.max() / N_VAL:.0%})")
print("ridge penalty chosen by CV:", {k[:2].strip(): v for k, v in alphas.items()})

# Is the gain over A real? Paired bootstrap over the 2024 crops.
err_a = (preds["A  ERA5 summaries"] - y_val) ** 2
print("\nRMSE gain over A, with 95% bootstrap interval (positive = better than A):")
for name in ("B  A + CLIP(rain)", "C  A + CLIP(all 4)"):
    err = (preds[name] - y_val) ** 2
    gains = []
    for _ in range(2000):
        i = rng.integers(0, N_VAL, N_VAL)
        gains.append(np.sqrt(err_a[i].mean()) - np.sqrt(err[i].mean()))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    point = np.sqrt(err_a.mean()) - np.sqrt(err.mean())
    print(f"   {name:20s} {point:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
