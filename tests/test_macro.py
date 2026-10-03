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


def test_fiscal_elasticity_impact(monthly, config):
    from forecasting.term_premium import fiscal_impact
    cfg = {"deficit_revision_pp": 1.0, "debt_revision_pp": 6.0}
    assert fiscal_impact(monthly, {**cfg, "measure": "deficit"})[0] == pytest.approx(25.0)
    assert fiscal_impact(monthly, {**cfg, "measure": "debt"})[0] == pytest.approx(18.0)
    assert fiscal_impact(monthly, cfg)[0] == pytest.approx(21.5)          # average
    assert fiscal_impact(monthly, {})[0] == 0.0                           # no revision → priced already
    assert fiscal_impact(monthly, {**cfg, "enabled": False})[0] == 0.0
    assert fiscal_impact(monthly, {"deficit_revision_pp": -1.0})[0] == pytest.approx(-25.0)


def test_fiscal_premium_component_and_scenario_override(monthly, config):
    cfg = copy.deepcopy(config)
    cfg["forecast"]["term_premium"]["fiscal"] = {"debt_revision_pp": 10, "measure": "debt"}
    cfg["forecast"]["scenarios"][1]["fiscal"] = {"debt_revision_pp": 20}
    r = run_forecast(monthly, cfg, with_fan=False)
    base, hawk = r.scenarios["Base"], r.scenarios["Hawkish"]
    assert base["us_fiscal"].iloc[0] == 0.0
    assert base["us_fiscal"].iloc[-1] == pytest.approx(0.30)               # 10pp × 3bp, phased in
    assert hawk["us_fiscal"].iloc[-1] == pytest.approx(0.60)
    assert r.params["fiscal_by_scenario_bps"]["Hawkish"] == pytest.approx(60.0)


def test_hold_current_tp_target(monthly, config):
    cfg = copy.deepcopy(config)
    cfg["forecast"]["term_premium"] = {"target": "current"}
    r = run_forecast(monthly, cfg, with_fan=False)
    tp = r.scenarios["Base"]["us_tp"]
    assert tp.iloc[-1] == pytest.approx(tp.iloc[0])
    assert r.params["term_premium"]["method"] == "hold_current"
