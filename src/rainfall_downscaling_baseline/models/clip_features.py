"""CLIP patch features of the ERA5 rain map, used as extra U-Net inputs.

The CLIP loss only uses CLIP while training. Here CLIP describes the *input*
instead: the ERA5 rain map (bilinear on the fine grid, the same colour image as
the CLIP loss) is cut into overlapping square tiles the size of a training
patch. Frozen CLIP turns each tile into a 7 x 7 grid of patch features, which
keep where things are in the tile, unlike the single summary vector. PCA
(fitted on training days) shrinks each feature from 768 numbers to a few, the
grids are stitched back into one map per day at the U-Net bottleneck's
resolution (1/8 of the fine grid), and overlapping tiles are averaged.

Because every feature always comes from a tile of the same size, a random
training patch and a whole field at prediction time see features at the same
scale. Only ERA5 is used, so this works for any day, including the test year.

`pretrained=False` is the control experiment: the same ViT-B/32 with random
weights (always the same ones, from a fixed seed), so everything is identical
except what CLIP learned from 400 million image-caption pairs.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .clip_loss import rain_to_image

DOWN = 8  # the U-Net halves the grid three times
RANDOM_SEED = 1234  # fixes the random weights of the control encoder


def openai_patch_encoder(pretrained: bool = True):
    """OpenAI CLIP ViT-B/32 as a function: images (n, 3, 224, 224) -> (n, 768, 7, 7).

    With `pretrained=False` the same network has random weights instead.
    """
    import open_clip

    if pretrained:
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai"
        )
    else:
        with torch.random.fork_rng():  # do not disturb the training seed
            torch.manual_seed(RANDOM_SEED)
            model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu")
    visual = model.visual.eval().requires_grad_(False)
    visual.output_tokens = True

    def encode(images: torch.Tensor) -> torch.Tensor:
        _, tokens = visual(images)  # (n, 49, 768), the class token already removed
        n, count, width = tokens.shape
        side = int(round(count**0.5))
        return tokens.transpose(1, 2).reshape(n, width, side, side)

    return encode


class ClipPatchFeatures:
    def __init__(
        self,
        tile: int = 128,
        dims: int = 32,
        encoder=None,
        batch: int = 32,
        pretrained: bool = True,
    ) -> None:
        """`encoder` defaults to OpenAI CLIP (loaded when first needed); tests pass a stand-in."""
        self.pretrained = pretrained
        if tile % DOWN:
            raise ValueError(f"tile must be a multiple of {DOWN}, got {tile}")
        self.tile, self.dims, self.batch = tile, dims, batch
        self.stride = max(DOWN, tile // 4)
        self._encoder = encoder
        self.mean = self.components = self.std = None  # set by fit

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = openai_patch_encoder(self.pretrained)
        return self._encoder

    # --- tiling ---

    def _padded(self, rain: np.ndarray, land: np.ndarray):
        """Pad to a multiple of 8, like the U-Net does; padding is drawn as sea."""
        height, width = land.shape
        pad_h, pad_w = -height % DOWN, -width % DOWN
        rain = np.pad(rain, ((0, 0), (0, pad_h), (0, pad_w)))
        land = np.pad(land, ((0, pad_h), (0, pad_w)))
        if min(land.shape) < self.tile:
            raise ValueError(f"field {land.shape} is smaller than one tile ({self.tile})")
        return rain, land

    def _corners(self, size: int) -> list[int]:
        starts = list(range(0, size - self.tile + 1, self.stride))
        if starts[-1] != size - self.tile:
            starts.append(size - self.tile)  # the last tile ends at the edge
        return starts

    @torch.no_grad()
    def _tile_tokens(self, tiles: np.ndarray, land_tiles: np.ndarray) -> torch.Tensor:
        """Raw CLIP patch features of each tile: (tiles, width, grid, grid)."""
        rain, land = torch.from_numpy(tiles), torch.from_numpy(land_tiles)
        return torch.cat(
            [
                self.encoder(rain_to_image(rain[i : i + self.batch], land[i : i + self.batch]))
                for i in range(0, len(rain), self.batch)
            ]
        )

    def _day_tiles(self, rain_day: np.ndarray, land: np.ndarray):
        ys, xs = self._corners(land.shape[0]), self._corners(land.shape[1])
        windows = [(y, x) for y in ys for x in xs]
        tiles = np.stack([rain_day[y : y + self.tile, x : x + self.tile] for y, x in windows])
        lands = np.stack([land[y : y + self.tile, x : x + self.tile] for y, x in windows])
        return windows, tiles.astype(np.float32), lands.astype(np.float32)

    # --- fit and transform ---

    def fit(self, rain: np.ndarray, land: np.ndarray, every: int = 6) -> "ClipPatchFeatures":
        """PCA of patch features from every `every`-th training day.

        Rain in mm, shape (days, lat, lon); land, shape (lat, lon).
        """
        rain, land = self._padded(rain, land)
        total, outer, count = None, None, 0
        for day in range(0, len(rain), every):
            _, tiles, lands = self._day_tiles(rain[day], land)
            tokens = self._tile_tokens(tiles, lands).double()
            flat = tokens.permute(0, 2, 3, 1).reshape(-1, tokens.shape[1])
            total = flat.sum(0) if total is None else total + flat.sum(0)
            outer = flat.T @ flat if outer is None else outer + flat.T @ flat
            count += len(flat)
        mean = total / count
        covariance = outer / count - torch.outer(mean, mean)
        values, vectors = torch.linalg.eigh(covariance)  # ascending order
        self.dims = min(self.dims, len(values))  # cannot keep more than there are
        order = values.argsort(descending=True)[: self.dims]
        self.mean = mean.float()
        self.components = vectors[:, order].float()  # (width, dims)
        self.std = values[order].clamp(min=1e-12).sqrt().float()  # unit variance per component
        return self

    def transform(self, rain: np.ndarray, land: np.ndarray) -> np.ndarray:
        """Feature maps, shape (days, dims, padded lat / 8, padded lon / 8)."""
        if self.mean is None:
            raise RuntimeError("call fit first")
        rain, land = self._padded(rain, land)
        cells = self.tile // DOWN
        out = np.zeros((len(rain), self.dims, land.shape[0] // DOWN, land.shape[1] // DOWN))
        counts = np.zeros(out.shape[-2:])
        for day in range(len(rain)):
            windows, tiles, lands = self._day_tiles(rain[day], land)
            tokens = self._tile_tokens(tiles, lands)
            reduced = torch.einsum(
                "nwyx,wd->ndyx", tokens - self.mean.view(1, -1, 1, 1), self.components
            ) / self.std.view(1, -1, 1, 1)
            grids = F.interpolate(reduced, size=(cells, cells), mode="bilinear").numpy()
            for (y, x), grid in zip(windows, grids, strict=True):
                out[day, :, y // DOWN : y // DOWN + cells, x // DOWN : x // DOWN + cells] += grid
                if day == 0:
                    counts[y // DOWN : y // DOWN + cells, x // DOWN : x // DOWN + cells] += 1
        return (out / counts).astype(np.float32)

    # --- saving: only PCA is stored, CLIP itself is reloaded when needed ---

    def state(self) -> dict:
        return {
            "tile": self.tile,
            "dims": self.dims,
            "mean": self.mean,
            "components": self.components,
            "std": self.std,
            "pretrained": self.pretrained,
        }

    @classmethod
    def from_state(cls, state: dict, encoder=None) -> "ClipPatchFeatures":
        features = cls(
            tile=state["tile"],
            dims=state["dims"],
            encoder=encoder,
            pretrained=state.get("pretrained", True),
        )
        features.mean, features.components, features.std = (
            state["mean"],
            state["components"],
            state["std"],
        )
        return features
