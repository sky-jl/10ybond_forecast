import copy

import numpy as np
import pandas as pd
import pytest

from forecasting import run_forecast
from forecasting.backtest import _vintage, run_backtest
from forecasting.expectations import expected_avg_short_rate
from forecasting.utils import anchored_path, horizon_index, overlay_path, parse_period_key


def test_identity_components_sum_to_yield(monthly, config):
    r = run_forecast(monthly, config, with_fan=False)
    for sc in r.scenarios.values():
        us = sc["us_expectations"] + sc["us_basis"] + sc["us_tp"] + sc["us_fiscal"] + sc["us_overlay"]
        np.testing.assert_allclose(us, sc["us_10y"])
        np.testing.assert_allclose(sc["us_10y"] + sc["ca_spread"] + sc["ca_overlay"], sc["canada_10y"])


def test_forecast_starts_at_last_observation(monthly, config):
    r = run_forecast(monthly, config, with_fan=False)
    for col in ("us_10y", "canada_10y", "fed_funds", "boc_rate"):
        assert r.central[col].iloc[0] == pytest.approx(monthly[col].iloc[-1])


def test_scenario_ordering_and_probabilities(monthly, config):
    r = run_forecast(monthly, config)
    assert sum(r.probabilities.values()) == pytest.approx(1.0)
    end = {n: s["us_10y"].iloc[-1] for n, s in r.scenarios.items()}
    assert end["Hawkish"] > end["Base"] > end["Dovish"]
    assert len(r.quarterly) == 8


def test_fan_widens_with_horizon(monthly, config):
    r = run_forecast(monthly, config)
    width = r.fan["us_10y_p90"] - r.fan["us_10y_p10"]
    assert width.iloc[3] < width.iloc[12] < width.iloc[24]
    assert (r.fan["us_10y_p10"] <= r.fan["us_10y_p90"]).all()


def test_user_overlay_and_tp_view_shift_forecast(monthly, config):
    base = run_forecast(monthly, config, with_fan=False)
    cfg = copy.deepcopy(config)
    cfg["forecast"]["overlay"] = {"us_10y_bps": {"2028Q3": 25}}
    cfg["forecast"]["term_premium"]["target"] = 1.5
    alt = run_forecast(monthly, cfg, with_fan=False)
    diff = alt.central["us_10y"] - base.central["us_10y"]
    assert diff.iloc[0] == pytest.approx(0.0)
    assert diff.iloc[-1] > 0.25  # overlay + higher TP


def test_anchored_path_interpolation():
    idx = horizon_index(pd.Timestamp("2026-09-30"), 24)
    path = anchored_path({"2026Q4": 3.5, "2027-12": 3.0}, 4.0, idx)
    assert path[0] == 4.0
    assert path[3] == pytest.approx(3.5)        # Dec 2026
    assert path[15] == pytest.approx(3.0)       # Dec 2027
    assert path[-1] == pytest.approx(3.0)       # held flat after last anchor
    assert 3.5 > path[9] > 3.0
    assert parse_period_key("2027Q2") == pd.Timestamp("2027-06-30")
    assert overlay_path({"2027Q4": 20}, idx)[15] == pytest.approx(0.20)


def test_expected_avg_short_rate_limits():
    flat = expected_avg_short_rate(np.full(25, 3.0), neutral=3.0, convergence_halflife=36)
    np.testing.assert_allclose(flat, 3.0)
    high = expected_avg_short_rate(np.full(25, 5.0), neutral=3.0, convergence_halflife=36)
    assert (high > 3.0).all() and (high < 5.0).all()


def test_backtest_has_no_lookahead(monthly):
    t = monthly.index[200]
    v = _vintage(monthly, t)
    assert v.index[-1] == t
    assert v["core_pce_yoy"].iloc[-1] == pytest.approx(monthly["core_pce_yoy"].iloc[199])


def test_backtest_summary(monthly, config):
    bt = run_backtest(monthly.loc[:"2015-12-31"], config, start="2010-01-31", step=3)
    s = bt["summary"]
    assert {"rw", "model", "ar1", "forward"} <= set(s["model"])
    rw = s[s["model"] == "rw"]
    assert np.allclose(rw["rmse_vs_rw"], 1.0)
    e = bt["errors"]
    assert e.groupby(["origin", "target", "model"])["h"].max().max() <= 24


def test_trailing_partial_month_is_trimmed(monthly):
    from data_fetcher import finalize_monthly_dataset
    raw = monthly.copy()
    nxt = raw.index[-1] + pd.offsets.MonthEnd(1)
    raw.loc[nxt] = raw.iloc[-1]
    raw.loc[nxt, ["canada_10y", "canada_2y"]] = np.nan   # BoC not yet published
    out = finalize_monthly_dataset(raw.drop(columns=["spread_can_us", "policy_diff"]))
    assert out.index[-1] == monthly.index[-1]


def test_short_boc_history_does_not_break_us_backtest(monthly, config):
    df = monthly.copy()
    df.loc[:"2009-03-31", ["boc_rate", "policy_diff", "spread_can_us"]] = np.nan
    bt = run_backtest(df.loc[:"2014-12-31"], config, start="2006-01-31", step=3)
    e = bt["errors"]
    us_model = e[(e["target"] == "us_10y") & (e["model"] == "model")]
    assert us_model["origin"].min() == pd.Timestamp("2006-01-31")
