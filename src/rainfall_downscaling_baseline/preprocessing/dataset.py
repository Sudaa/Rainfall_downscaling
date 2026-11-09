"""PyTorch Dataset of random patches cut from daily fields, for training the U-Net."""

import numpy as np
import torch
from torch.utils.data import Dataset


class PairedPatches(Dataset):
    """One random square patch per day: (inputs, target, land mask), all on the fine grid.

    A new location is drawn every time a day is requested, so each epoch sees
    different parts of the region. That is cheap augmentation, and small
    patches train much faster on a CPU than whole fields.
    """

    def __init__(
        self,
        inputs: np.ndarray,  # (days, channels, lat, lon), already normalised
        target: np.ndarray,  # (days, lat, lon), 0 over the sea
        land: np.ndarray,  # (lat, lon), True on land
        patch_size: int = 128,
        seed: int = 0,
        side: np.ndarray | None = None,  # (days, channels, lat / 8, lon / 8), e.g. CLIP features
    ) -> None:
        """With `side`, patch corners fall on multiples of 8, so the matching part of
        the coarser side map can be cut out exactly; it is returned as a fourth item."""
        self.inputs, self.target, self.patch_size = inputs, target, patch_size
        self.side = side
        self.land = land.astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, day: int) -> tuple[torch.Tensor, ...]:
        size = self.patch_size
        height, width = self.land.shape
        if self.side is None:
            y = self.rng.integers(0, height - size + 1)
            x = self.rng.integers(0, width - size + 1)
        else:
            y = 8 * self.rng.integers(0, (height - size) // 8 + 1)
            x = 8 * self.rng.integers(0, (width - size) // 8 + 1)
        window = (slice(y, y + size), slice(x, x + size))
        items = (
            torch.from_numpy(self.inputs[day][:, window[0], window[1]]),
            torch.from_numpy(self.target[day][window]),
            torch.from_numpy(self.land[window]),
        )
        if self.side is None:
            return items
        cells = size // 8
        side = self.side[day][:, y // 8 : y // 8 + cells, x // 8 : x // 8 + cells]
        return (*items, torch.from_numpy(side))
