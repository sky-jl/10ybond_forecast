import copy

import numpy as np
import pytest

from forecasting import run_forecast
from forecasting.attribution import attribute
from forecasting.term_premium import fit_tp_regression


def test_attribution_splits_add_up(monthly):
    a = attribute(monthly)
    total = a["us_10y_change_bps"]
    for name in ("Real vs inflation", "Front end vs curve"):
        assert sum(a["splits"][name].values()) == pytest.approx(total, abs=1e-6)
    et = a["splits"]["Expectations vs term premium"]
    assert sum(et.values()) == pytest.approx(total, abs=1e-6)   # us_rny = us_10y - us_tp
    assert a["reading"][0].startswith("US 10Y")


def test_fiscal_regression_picks_up_debt(monthly):
    fit = fit_tp_regression(monthly)
    assert "debt_gdp" in fit["drivers"]
    assert fit["coef"]["debt_gdp"] > 0          # synthetic TP rises with debt/GDP
    assert 0 < fit["r2"] <= 1


def test_higher_debt_path_and_addons_raise_tp(monthly, config):
    cfg = copy.deepcopy(config)
    cfg["forecast"]["term_premium"] = {"target": "regression"}
    base = run_forecast(monthly, cfg, with_fan=False)
    cfg["forecast"]["term_premium"] = {"target": "regression", "drivers": {"debt_gdp": 130},
                                       "addons_bps": {"fed_independence": 20}}
    hi = run_forecast(monthly, cfg, with_fan=False)
    assert hi.central["us_tp"].iloc[-1] > base.central["us_tp"].iloc[-1] + 0.2
    assert hi.params["term_premium"]["addons_bps"] == {"fed_independence": 20.0}


def test_scenario_fiscal_drivers(monthly, config):
    cfg = copy.deepcopy(config)
    cfg["forecast"]["scenarios"][1]["tp_drivers"] = {"debt_gdp": 140}
    cfg["forecast"]["scenarios"][1].pop("term_premium_target", None)
    r = run_forecast(monthly, cfg, with_fan=False)
    tp_end = {n: s["us_tp"].iloc[-1] for n, s in r.scenarios.items()}
    assert tp_end["Hawkish"] > tp_end["Base"]
    assert np.isfinite(list(tp_end.values())).all()
