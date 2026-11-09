"""Small U-Net that predicts a correction (residual) on top of bilinear interpolation.

This mirrors CorrDiff's first step, a regression network that predicts the
mean. Bilinear already gives a reasonable field, so the network only has to
learn what bilinear gets wrong, in log(1 + rain) space:

    prediction = from_log(to_log(bilinear) + unet(inputs))

Inputs, all interpolated to the fine grid and normalised with training
statistics: rain (as log(1 + rain)), water vapour, and u and v wind at 850 hPa.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader

from ..preprocessing.dataset import PairedPatches
from ..preprocessing.transforms import Normalizer, from_log, to_log
from .bilinear import BilinearDownscaler

INPUTS = ("precip", "tcwv", "u", "v")


def _block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Three levels down and up, with skip connections between matching levels.

    Going down halves the grid three times, so inputs are padded to a multiple
    of 8 and the output is cut back to the input size.

    `side_channels` > 0 adds maps at the bottleneck's resolution (1/8 of the
    grid, e.g. CLIP patch features), joined to the bottleneck's input.
    """

    def __init__(self, in_channels: int, base: int = 32, side_channels: int = 0) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList(
            [_block(in_channels, c[0]), _block(c[0], c[1]), _block(c[1], c[2])]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottom = _block(c[2] + side_channels, c[3])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)]
        )
        self.merge = nn.ModuleList([_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor, side: torch.Tensor | None = None) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = nn.functional.pad(x, (0, -width % 8, 0, -height % 8), mode="reflect")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        if side is not None:
            x = torch.cat([x, side[..., : x.shape[-2], : x.shape[-1]]], dim=1)
        x = self.bottom(x)
        for up, merge, skip in zip(self.up, self.merge, reversed(skips), strict=True):
            x = merge(torch.cat([up(x), skip], dim=1))
        return self.head(x)[:, 0, :height, :width]


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, land: torch.Tensor) -> torch.Tensor:
    """Mean squared error over land cells only: the sea has no CHIRPS values."""
    return ((pred - target) ** 2 * land).sum() / land.sum().clamp(min=1)


def _soft_share(rain: torch.Tensor, land: torch.Tensor, threshold: float, width: float):
    """Share of land pixels above `threshold` mm, per field, with a smooth threshold.

    A hard `rain > threshold` has no gradient, so the network could not learn
    from it. The sigmoid ramps from 0 to 1 around the threshold instead
    (`width` mm sets how fast): 18 mm counts about 0.27 at 20 mm, 25 mm about 0.92.
    """
    wet = torch.sigmoid((rain - threshold) / width)
    return (wet * land).sum((-2, -1)) / land.sum((-2, -1)).clamp(min=1)


def _count_loss(
    pred_rain: torch.Tensor,
    true_rain: torch.Tensor,
    land: torch.Tensor,
    thresholds: tuple[float, ...],
    width: float = 2.0,
    levels: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """How far apart the shares of heavy-rain pixels are, in centred crops of several sizes.

    Crops are the whole field, its centre half and its centre quarter (like the
    multi-level crops of Clip4Rain's MRA loss, but with no CLIP): the share is
    judged per region, not per pixel, so rain slightly out of place is not
    punished twice, while rain that is too light everywhere is.
    """
    height, width_px = pred_rain.shape[-2:]
    gaps = []
    for level in levels:
        h, w = height // level, width_px // level
        top, left = (height - h) // 2, (width_px - w) // 2
        crop = (..., slice(top, top + h), slice(left, left + w))
        for t in thresholds:
            pred_share = _soft_share(pred_rain[crop], land[crop], t, width)
            true_share = _soft_share(true_rain[crop], land[crop], t, width)
            gaps.append((pred_share - true_share).abs().mean())
    return torch.stack(gaps).mean()


LOG_MAX = float(np.log1p(1000.0))  # cap on log(1 + rain) before exp, so it cannot overflow


AUX_LOSSES = ("none", "count", "clip", "clip-image")
_SIDE_CACHE: dict = {}  # CLIP feature maps already computed in this run


class UNetDownscaler:
    def __init__(
        self,
        val: tuple[xr.Dataset, xr.DataArray] | None = None,
        epochs: int = 40,
        patience: int = 5,
        batch_size: int = 8,
        patch_size: int = 128,
        learning_rate: float = 1e-3,
        base: int = 32,
        seed: int = 0,
        aux: str = "none",
        aux_weight: float = 0.0,
        thresholds: tuple[float, ...] = (20.0, 50.0),
        clip_input: bool = False,
    ) -> None:
        """`aux` adds a second loss term judged per region, weighted by `aux_weight` (lambda):

        - "count": share of pixels above `thresholds` in crops (no CLIP)
        - "clip": CLIP alignment of crops with a text prompt (Clip4Rain's MRA)
        - "clip-image": CLIP image features only, no text (the ablation)

        `clip_input` also feeds CLIP patch features of the ERA5 rain map into
        the bottleneck (see clip_features.py). It can be combined with `aux`.
        """
        if aux not in AUX_LOSSES:
            raise ValueError(f"aux must be one of {AUX_LOSSES}, got {aux!r}")
        self.val = val  # (coarse, fine) of the validation split, for early stopping
        self.epochs, self.patience = epochs, patience
        self.batch_size, self.patch_size = batch_size, patch_size
        self.learning_rate, self.base, self.seed = learning_rate, base, seed
        self.aux, self.aux_weight, self.thresholds = aux, aux_weight, thresholds
        self.clip_input = clip_input
        self.name = "unet" if aux == "none" else f"unet+{aux}"
        if clip_input:
            self.name += "+clip-input"
        self.interpolator = BilinearDownscaler()
        self.clip_features = None

    # --- data ---

    def _inputs(self, coarse: xr.Dataset) -> np.ndarray:
        """Normalised input channels on the fine grid, shape (days, channels, lat, lon)."""
        channels = []
        for var in INPUTS:
            field = self.interpolator.interpolate(coarse[var])
            if var == "precip":
                field = to_log(field.clip(min=0))
            channels.append(self.norms[var].transform(field).values.astype(np.float32))
        return np.stack(channels, axis=1)

    def _side(self, coarse: xr.Dataset) -> np.ndarray | None:
        """CLIP patch-feature maps of the ERA5 rain, or None without `clip_input`."""
        if not self.clip_input:
            return None
        # The same days are often asked for twice (predict, then QM on top): reuse them
        key = (str(coarse.time.values[0]), str(coarse.time.values[-1]), coarse.sizes["time"])
        # PCA does not depend on the seed, so other seeds' models share these maps
        key += (float(self.clip_features.components.sum()), float(self.clip_features.mean.sum()))
        if key not in _SIDE_CACHE:
            rain = self.interpolator.predict(coarse).clip(min=0).values.astype(np.float32)
            _SIDE_CACHE[key] = self.clip_features.transform(rain, self.land)
        return _SIDE_CACHE[key]

    def _residual(self, coarse: xr.Dataset, fine: xr.DataArray) -> np.ndarray:
        """Training target: log(1 + CHIRPS) - log(1 + bilinear), 0 over the sea."""
        bilinear = self.interpolator.predict(coarse).clip(min=0)
        residual = to_log(fine) - to_log(bilinear)
        return residual.fillna(0).values.astype(np.float32)

    def _loss(self, inputs, residual, target, land) -> tuple[torch.Tensor, float, float]:
        """Total loss, and its MSE and auxiliary parts for printing.

        The auxiliary part needs rain in mm: bilinear log rain is channel 0 of
        the inputs (undo its normalisation), plus the predicted or true residual.
        """
        mse = _masked_mse(residual, target, land)
        if self.aux == "none":
            return mse, mse.item(), 0.0
        norm = self.norms["precip"]
        log_bilinear = inputs[:, 0] * norm.std + norm.mean
        pred_rain = torch.expm1((log_bilinear + residual).clamp(max=LOG_MAX)).clamp(min=0)
        true_rain = torch.expm1((log_bilinear + target).clamp(max=LOG_MAX)).clamp(min=0)
        if self.aux == "count":
            extra = _count_loss(pred_rain, true_rain, land, self.thresholds)
        else:
            extra = self.clip_loss(pred_rain, true_rain, land)
        return mse + self.aux_weight * extra, mse.item(), extra.item()

    def _predict_residual(self, inputs: np.ndarray, side: np.ndarray | None = None) -> np.ndarray:
        """Run the network on whole fields, a few days at a time."""
        self.net.eval()
        with torch.no_grad():
            parts = [
                self.net(
                    torch.from_numpy(inputs[i : i + self.batch_size]),
                    None if side is None else torch.from_numpy(side[i : i + self.batch_size]),
                ).numpy()
                for i in range(0, len(inputs), self.batch_size)
            ]
        return np.concatenate(parts)

    # --- Downscaler interface ---

    def fit(self, coarse: xr.Dataset, fine: xr.DataArray) -> None:
        """Train on random patches; keep the weights with the lowest validation loss."""
        torch.manual_seed(self.seed)
        self.interpolator.fit(coarse, fine)
        # Normalisation statistics come from the training years only
        self.norms = {
            var: Normalizer().fit(to_log(coarse[var]) if var == "precip" else coarse[var])
            for var in INPUTS
        }
        self.land = fine.notnull().all("time").values
        if self.aux.startswith("clip"):
            from .clip_loss import ClipAlignmentLoss  # loads CLIP only when it is needed

            self.clip_loss = ClipAlignmentLoss(mode="image" if self.aux == "clip-image" else "text")
        if self.clip_input:
            from .clip_features import ClipPatchFeatures  # loads CLIP only when it is needed

            rain = self.interpolator.predict(coarse).clip(min=0).values.astype(np.float32)
            print("CLIP patch features: fitting PCA ...", flush=True)
            self.clip_features = ClipPatchFeatures(tile=self.patch_size).fit(rain, self.land)
            print("CLIP patch features: training days ...", flush=True)
        side = self._side(coarse)
        patches = PairedPatches(
            self._inputs(coarse),
            self._residual(coarse, fine),
            self.land,
            self.patch_size,
            self.seed,
            side,
        )
        loader = DataLoader(patches, batch_size=self.batch_size, shuffle=True)
        val_side = None
        if self.val is not None:
            if self.clip_input:
                print("CLIP patch features: validation days ...", flush=True)
            val_inputs, val_side = self._inputs(self.val[0]), self._side(self.val[0])
            val_target = torch.from_numpy(self._residual(*self.val))
        land = torch.from_numpy(self.land.astype(np.float32))

        side_channels = self.clip_features.dims if self.clip_input else 0
        self.net = UNet(len(INPUTS), self.base, side_channels)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        best_loss, best_weights, waited = float("inf"), None, 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            self.net.train()
            losses = []
            for x, y, mask, *side_batch in loader:
                optimiser.zero_grad()
                loss, _, _ = self._loss(x, self.net(x, *side_batch), y, mask)
                loss.backward()
                optimiser.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            # Early stopping uses the same total loss on whole validation fields
            if self.val is not None:
                val_loss, mse, extra = self._validation_loss(val_inputs, val_target, land, val_side)
            else:
                val_loss, mse, extra = train_loss, float("nan"), float("nan")
            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_loss,
                    "val_mse": mse,
                    "val_aux": extra,
                }
            )
            seconds = time.time() - start
            parts = f" (mse {mse:.4f}, {self.aux} {extra:.4f})" if self.aux != "none" else ""
            losses_text = f"train {train_loss:.4f}  val {val_loss:.4f}{parts}"
            print(f"epoch {epoch:2d}  {losses_text}  ({seconds:.0f} s)")

            # Early stopping: keep the best weights, stop when validation stops improving
            if val_loss < best_loss:
                best_loss, best_weights, waited = val_loss, copy.deepcopy(self.net.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    best_epoch = epoch - waited
                    print(f"no improvement for {self.patience} epochs: keep epoch {best_epoch}")
                    break
        self.net.load_state_dict(best_weights)

    def _validation_loss(self, inputs, target, land, side=None):
        """Mean loss over the validation days, a few days at a time (CLIP is memory hungry)."""
        residual = torch.from_numpy(self._predict_residual(inputs, side))
        totals, mses, extras = [], [], []
        with torch.no_grad():
            for i in range(0, len(inputs), self.batch_size):
                part = slice(i, i + self.batch_size)
                total, mse, extra = self._loss(
                    torch.from_numpy(inputs[part]),
                    residual[part],
                    target[part],
                    land.expand_as(residual[part]),
                )
                totals.append(total.item())
                mses.append(mse)
                extras.append(extra)
        return float(np.mean(totals)), float(np.mean(mses)), float(np.mean(extras))

    def predict(self, coarse: xr.Dataset) -> xr.DataArray:
        bilinear = self.interpolator.predict(coarse).clip(min=0)
        residual = self._predict_residual(self._inputs(coarse), self._side(coarse))
        return from_log(to_log(bilinear) + residual).rename("precip")

    # --- saving, so evaluate does not have to train again ---

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weights": self.net.state_dict(),
                "base": self.base,
                "aux": self.aux,
                "aux_weight": self.aux_weight,
                "thresholds": self.thresholds,
                "norms": {var: (n.mean, n.std) for var, n in self.norms.items()},
                "lat": torch.tensor(self.interpolator.lat.values),
                "lon": torch.tensor(self.interpolator.lon.values),
                "land": torch.tensor(self.land),
                "history": self.history,
                "seed": self.seed,
                "patience": self.patience,
                "clip_input": self.clip_input,
                "clip_features": self.clip_features.state() if self.clip_input else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "UNetDownscaler":
        saved = torch.load(path)
        if "aux" in saved:
            aux, aux_weight = saved["aux"], saved["aux_weight"]
        else:  # saved before `aux` existed: only the count loss was possible
            aux_weight = saved.get("count_weight", 0.0)
            aux = "count" if aux_weight else "none"
        model = cls(
            base=saved["base"],
            aux=aux,
            aux_weight=aux_weight,
            thresholds=saved.get("thresholds", (20.0, 50.0)),
            clip_input=saved.get("clip_input", False),
            seed=saved.get("seed", 0),
            patience=saved.get("patience", 5),
        )
        model.norms = {var: Normalizer(*stats) for var, stats in saved["norms"].items()}
        lat, lon = saved["lat"].numpy(), saved["lon"].numpy()
        model.interpolator.lat = xr.DataArray(lat, coords={"lat": lat})
        model.interpolator.lon = xr.DataArray(lon, coords={"lon": lon})
        model.land, model.history = saved["land"].numpy(), saved["history"]
        side_channels = 0
        if model.clip_input:
            from .clip_features import ClipPatchFeatures

            model.clip_features = ClipPatchFeatures.from_state(saved["clip_features"])
            side_channels = model.clip_features.dims
        model.net = UNet(len(INPUTS), model.base, side_channels)
        model.net.load_state_dict(saved["weights"])
        return model
