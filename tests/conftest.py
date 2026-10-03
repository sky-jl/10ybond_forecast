import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def config():
    with open(ROOT / "config" / "quarterly_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def monthly():
    from forecasting.synthetic import make_synthetic_monthly
    return make_synthetic_monthly()
