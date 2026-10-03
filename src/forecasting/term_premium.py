"""
Term-premium component of the 10Y yield — the main channel for fiscal and macro risk.

TP_{t+h} = target + (TP_t - target) * phi^h

target options (config `forecast.term_premium.target`):
  - a number (%)        → subjective view, e.g. 0.75
  - "historical"        → trailing N-year mean of the term premium (default)
  - "regression"        → fiscal "fair value": OLS of TP on
                            debt held by the public / GDP, federal deficit / GDP,
                            Fed balance sheet / GDP (QE absorbs duration — controls the debt link),
                            12m realised 10Y volatility (rate uncertainty),
                          evaluated at YOUR driver assumptions for the horizon end
                          (e.g. CBO debt / deficit path, QT path).
plus named add-ons in bps (`addons_bps`) for risks that don't fit a regression — e.g.
Fed independence, geopolitics, energy-driven inflation risk — added to the target.

The regression is descriptive, not causal: fiscal variables trend slowly and QE distorted the
2010s, so coefficients are shown with R² and sample so the user can judge them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import fit_ar1, ols, phi_from_halflife

logger = logging.getLogger(__name__)

DRIVERS = ["debt_gdp", "deficit_gdp", "fed_bs_gdp", "us_10y_rvol_12m"]
DRIVER_LABELS = {
    "debt_gdp": "Debt held by public (% GDP)",
    "deficit_gdp": "Federal balance (% GDP, − = deficit)",
    "fed_bs_gdp": "Fed balance sheet (% GDP)",
    "us_10y_rvol_12m": "10Y realised vol, 12m (bps/day)",
}


def drivers_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in ("debt_gdp", "deficit_gdp", "fed_bs_gdp"):
        if c in df:
            out[c] = df[c]
    if "us_10y_rvol" in df:
        out["us_10y_rvol_12m"] = df["us_10y_rvol"].rolling(12, min_periods=6).mean()
    return out


def fit_tp_regression(df: pd.DataFrame, min_obs: int = 60) -> dict:
    """OLS of the term premium on available fiscal / uncertainty drivers."""
    X = drivers_frame(df)
    cols = [c for c in DRIVERS if c in X and X[c].notna().sum() >= min_obs]
    data = pd.concat([df["us_tp"], X[cols]], axis=1).dropna()
    if not cols or len(data) < min_obs:
        raise ValueError("insufficient driver data for term-premium regression")
    A = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
    coef, resid = ols(data["us_tp"].to_numpy(), A)
    fitted = pd.Series(A @ coef, index=data.index, name="tp_fair_value")
    latest = X[cols].dropna().iloc[-1]
    fair_now = float(coef[0] + latest.to_numpy() @ coef[1:])
    tp_now = float(df["us_tp"].dropna().iloc[-1])
    return {
        "drivers": cols,
        "coef": {"const": float(coef[0]), **{c: float(b) for c, b in zip(cols, coef[1:])}},
        "r2": float(1 - resid.var() / data["us_tp"].var()),
        "n": int(len(data)),
        "sample": f"{data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m}",
        "latest_drivers": {c: float(latest[c]) for c in cols},
        "fair_value_now": fair_now,
        "tp_now": tp_now,
        "residual_now_bps": (tp_now - fair_now) * 100,
        "fitted": fitted,
    }


def regression_target(df: pd.DataFrame, assumptions: dict | None, fit: dict | None = None) -> tuple[float, dict]:
    fit = fit or fit_tp_regression(df)
    point = {c: float((assumptions or {}).get(c) if (assumptions or {}).get(c) is not None
                      else fit["latest_drivers"][c]) for c in fit["drivers"]}
    target = fit["coef"]["const"] + sum(fit["coef"][c] * point[c] for c in fit["drivers"])
    contrib = {c: fit["coef"][c] * (point[c] - fit["latest_drivers"][c]) * 100 for c in fit["drivers"]}
    info = {"coef": {k: round(v, 4) for k, v in fit["coef"].items()}, "r2": round(fit["r2"], 3),
            "sample": fit["sample"], "driver_assumptions": point,
            "driver_change_contrib_bps": {k: round(v, 1) for k, v in contrib.items()}}
    return float(target), info


def addons_total(cfg: dict) -> tuple[float, dict]:
    adds = {k: float(v) for k, v in (cfg.get("addons_bps") or {}).items() if v}
    return sum(adds.values()) / 100.0, adds


def tp_target(df: pd.DataFrame, cfg: dict, fit: dict | None = None) -> tuple[float, dict]:
    target = cfg.get("target", "historical")
    window = int(cfg.get("historical_window_years", 10))
    if isinstance(target, (int, float)):
        base, info = float(target), {"method": "user"}
    elif target == "regression":
        try:
            base, rinfo = regression_target(df, cfg.get("drivers"), fit)
            info = {"method": "regression", **rinfo}
        except ValueError as exc:
            logger.warning("TP regression unavailable (%s); using historical mean", exc)
            target = "historical"
    if not isinstance(target, (int, float)) and target != "regression":
        tp = df["us_tp"].dropna()
        tp = tp[tp.index >= tp.index[-1] - pd.DateOffset(years=window)]
        base, info = float(tp.mean()), {"method": f"historical_{window}y"}
    add, adds = addons_total(cfg)
    info.update({"base_target": round(base, 3), "addons_bps": adds})
    return base + add, info


def tp_path(df: pd.DataFrame, horizon: int, cfg: dict | None) -> tuple[np.ndarray, dict]:
    cfg = cfg or {}
    tp = df["us_tp"].dropna()
    tp0 = float(tp.iloc[-1])
    try:
        fit = fit_tp_regression(df)
    except ValueError:
        fit = None
    target, info = tp_target(df, cfg, fit)
    if cfg.get("halflife_months"):
        phi = phi_from_halflife(float(cfg["halflife_months"]))
    else:
        phi = fit_ar1(tp)["phi"]
    h = np.arange(horizon + 1)
    path = target + (tp0 - target) * phi ** h
    info.update({"tp0": round(tp0, 3), "target": round(target, 3), "phi": round(phi, 4)})
    if fit:  # always report the fiscal fair value — useful context even for other methods
        info["fiscal_fair_value"] = {k: v for k, v in fit.items() if k != "fitted"}
    return path, info


def scenario_target(df: pd.DataFrame, cfg: dict, sc: dict) -> float | None:
    """Scenario-specific TP destination: explicit number, or fiscal drivers through the regression."""
    add, _ = addons_total({"addons_bps": {**(cfg.get("addons_bps") or {}), **(sc.get("tp_addons_bps") or {})}})
    if sc.get("term_premium_target") is not None:
        return float(sc["term_premium_target"]) + add
    if sc.get("tp_drivers"):
        try:
            t, _ = regression_target(df, {**(cfg.get("drivers") or {}), **sc["tp_drivers"]})
            return t + add
        except ValueError as exc:
            logger.warning("Scenario TP drivers ignored (%s)", exc)
    if sc.get("tp_addons_bps"):
        base, _ = tp_target(df, {**cfg, "addons_bps": {}})
        return base + add
    return None
