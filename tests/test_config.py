import pytest
from pydantic import ValidationError

from rainfall_downscaling_baseline.config import Config


def test_default_config_loads(cfg):
    assert cfg.region.lat_min < cfg.region.lat_max


def test_scale_factor_is_five(cfg):
    assert cfg.scale_factor == 5


def test_test_years_must_follow_train_years(cfg):
    data = cfg.model_dump()
    data["period"]["test_years"] = [min(cfg.period.train_years)]
    with pytest.raises(ValidationError):
        Config.model_validate(data)
