"""Shared test fixtures. pytest makes these available to every test file."""

import pytest

from rainfall_downscaling_baseline.config import Config, load_config


@pytest.fixture
def cfg() -> Config:
    return load_config()
