"""Load and validate experiment settings from a TOML file.

Every other module receives a `Config` object instead of reading files or
hard-coding numbers, so changing the region or period means editing one file.
pydantic checks the values when they are loaded, so a typo fails loudly here
rather than silently producing wrong data later.
"""

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


class Region(BaseModel):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if not -90 <= self.lat_min < self.lat_max <= 90:
            raise ValueError("need -90 <= lat_min < lat_max <= 90")
        if not self.lon_min < self.lon_max:
            raise ValueError("need lon_min < lon_max")
        return self


class Period(BaseModel):
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]
    months: list[int]

    @model_validator(mode="after")
    def _check_split(self) -> Self:
        # A time-based split: models are tuned and scored only on years after
        # all training years, so they never see that weather during training.
        train_end, val_start, val_end = (
            max(self.train_years),
            min(self.val_years),
            max(self.val_years),
        )
        if not train_end < val_start <= val_end < min(self.test_years):
            raise ValueError("years must be ordered train < validation < test, with no overlap")
        if not all(1 <= m <= 12 for m in self.months):
            raise ValueError("months must be between 1 and 12")
        return self

    @property
    def all_years(self) -> list[int]:
        return self.train_years + self.val_years + self.test_years


class Era5(BaseModel):
    single_level_vars: list[str]
    pressure_level_vars: list[str]
    pressure_level: int
    resolution_deg: float


class Chirps(BaseModel):
    resolution_deg: float


class Paths(BaseModel):
    raw: Path
    interim: Path
    processed: Path
    figures: Path
    checkpoints: Path


class Config(BaseModel):
    region: Region
    eval_regions: dict[str, Region] = {}
    period: Period
    era5: Era5
    chirps: Chirps
    paths: Paths

    @model_validator(mode="after")
    def _check_resolution_ratio(self) -> Self:
        ratio = self.era5.resolution_deg / self.chirps.resolution_deg
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"coarse/fine resolution ratio must be a whole number, got {ratio}")
        return self

    @property
    def scale_factor(self) -> int:
        """How many fine cells fit along one coarse cell (5 for 0.25 / 0.05)."""
        return round(self.era5.resolution_deg / self.chirps.resolution_deg)


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        return Config.model_validate(tomllib.load(f))
