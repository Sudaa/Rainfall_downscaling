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
    """

    def __init__(self, in_channels: int, base: int = 32) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.down = nn.ModuleList(
            [_block(in_channels, c[0]), _block(c[0], c[1]), _block(c[1], c[2])]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottom = _block(c[2], c[3])
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in (2, 1, 0)]
        )
        self.merge = nn.ModuleList([_block(c[i] * 2, c[i]) for i in (2, 1, 0)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x = nn.functional.pad(x, (0, -width % 8, 0, -height % 8), mode="reflect")
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
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
        count_weight: float = 0.0,
        thresholds: tuple[float, ...] = (20.0, 50.0),
    ) -> None:
        """`count_weight` (lambda) > 0 adds the heavy-rain count loss: MSE + lambda * count."""
        self.val = val  # (coarse, fine) of the validation split, for early stopping
        self.epochs, self.patience = epochs, patience
        self.batch_size, self.patch_size = batch_size, patch_size
        self.learning_rate, self.base, self.seed = learning_rate, base, seed
        self.count_weight, self.thresholds = count_weight, thresholds
        self.name = "unet" if count_weight == 0 else "unet+count"
        self.interpolator = BilinearDownscaler()

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

    def _residual(self, coarse: xr.Dataset, fine: xr.DataArray) -> np.ndarray:
        """Training target: log(1 + CHIRPS) - log(1 + bilinear), 0 over the sea."""
        bilinear = self.interpolator.predict(coarse).clip(min=0)
        residual = to_log(fine) - to_log(bilinear)
        return residual.fillna(0).values.astype(np.float32)

    def _loss(self, inputs, residual, target, land) -> tuple[torch.Tensor, float, float]:
        """Total loss, and its MSE and count parts for printing.

        The count part needs rain in mm: bilinear log rain is channel 0 of the
        inputs (undo its normalisation), plus the predicted or true residual.
        """
        mse = _masked_mse(residual, target, land)
        if self.count_weight == 0:
            return mse, mse.item(), 0.0
        norm = self.norms["precip"]
        log_bilinear = inputs[:, 0] * norm.std + norm.mean
        pred_rain = torch.expm1((log_bilinear + residual).clamp(max=LOG_MAX)).clamp(min=0)
        true_rain = torch.expm1((log_bilinear + target).clamp(max=LOG_MAX)).clamp(min=0)
        count = _count_loss(pred_rain, true_rain, land, self.thresholds)
        return mse + self.count_weight * count, mse.item(), count.item()

    def _predict_residual(self, inputs: np.ndarray) -> np.ndarray:
        """Run the network on whole fields, a few days at a time."""
        self.net.eval()
        with torch.no_grad():
            parts = [
                self.net(torch.from_numpy(inputs[i : i + self.batch_size])).numpy()
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
        patches = PairedPatches(
            self._inputs(coarse),
            self._residual(coarse, fine),
            self.land,
            self.patch_size,
            self.seed,
        )
        loader = DataLoader(patches, batch_size=self.batch_size, shuffle=True)
        if self.val is not None:
            val_inputs = self._inputs(self.val[0])
            val_target = torch.from_numpy(self._residual(*self.val))
        land = torch.from_numpy(self.land.astype(np.float32))

        self.net = UNet(len(INPUTS), self.base)
        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        best_loss, best_weights, waited = float("inf"), None, 0
        self.history = []
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            self.net.train()
            losses = []
            for x, y, mask in loader:
                optimiser.zero_grad()
                loss, _, _ = self._loss(x, self.net(x), y, mask)
                loss.backward()
                optimiser.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            # Early stopping uses the same total loss on whole validation fields
            if self.val is not None:
                pred = torch.from_numpy(self._predict_residual(val_inputs))
                total, mse, count = self._loss(
                    torch.from_numpy(val_inputs), pred, val_target, land.expand_as(pred)
                )
                val_loss = total.item()
            else:
                val_loss, mse, count = train_loss, float("nan"), float("nan")
            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_loss,
                    "val": val_loss,
                    "val_mse": mse,
                    "val_count": count,
                }
            )
            seconds = time.time() - start
            parts = f" (mse {mse:.4f}, count {count:.4f})" if self.count_weight else ""
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

    def predict(self, coarse: xr.Dataset) -> xr.DataArray:
        bilinear = self.interpolator.predict(coarse).clip(min=0)
        residual = self._predict_residual(self._inputs(coarse))
        return from_log(to_log(bilinear) + residual).rename("precip")

    # --- saving, so evaluate does not have to train again ---

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "weights": self.net.state_dict(),
                "base": self.base,
                "count_weight": self.count_weight,
                "thresholds": self.thresholds,
                "norms": {var: (n.mean, n.std) for var, n in self.norms.items()},
                "lat": torch.tensor(self.interpolator.lat.values),
                "lon": torch.tensor(self.interpolator.lon.values),
                "land": torch.tensor(self.land),
                "history": self.history,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "UNetDownscaler":
        saved = torch.load(path)
        model = cls(
            base=saved["base"],
            count_weight=saved.get("count_weight", 0.0),
            thresholds=saved.get("thresholds", (20.0, 50.0)),
        )
        model.norms = {var: Normalizer(*stats) for var, stats in saved["norms"].items()}
        lat, lon = saved["lat"].numpy(), saved["lon"].numpy()
        model.interpolator.lat = xr.DataArray(lat, coords={"lat": lat})
        model.interpolator.lon = xr.DataArray(lon, coords={"lon": lon})
        model.land, model.history = saved["land"].numpy(), saved["history"]
        model.net = UNet(len(INPUTS), model.base)
        model.net.load_state_dict(saved["weights"])
        return model
