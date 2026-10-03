"""
Term-premium component of the 10Y yield — the main channel for fiscal and macro risk.

TP_{t+h} = target + (TP_t - target) * phi^h

target options (config `forecast.term_premium.target`):
  - a number (%)        → subjective view, e.g. 0.75
  - "historical"        → trailing N-year mean of the term premium (default)
  - "current"           → hold today's term premium: keeps the fiscal / risk premia already priced
                          (use with the fiscal-outlook revision for any further change)
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
    X = drivers_frame(df).replace([np.inf, -np.inf], np.nan)
    cols = [c for c in DRIVERS if c in X and X[c].notna().sum() >= min_obs and X[c].std() > 1e-9]
    data = pd.concat([df["us_tp"], X[cols]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if not cols or len(data) < min_obs:
        raise ValueError("insufficient driver data for term-premium regression")
    # Standardise drivers for a well-conditioned fit, then map back to per-unit coefficients
    mu, sd = data[cols].mean(), data[cols].std()
    Z = ((data[cols] - mu) / sd).to_numpy()
    A = np.column_stack([np.ones(len(data)), Z])
    zcoef, resid = ols(data["us_tp"].to_numpy(), A)
    slopes = zcoef[1:] / sd.to_numpy()
    const = zcoef[0] - float(np.sum(slopes * mu.to_numpy()))
    coef = np.concatenate([[const], slopes])
    if not np.all(np.isfinite(coef)):
        raise ValueError("term-premium regression did not converge (non-finite coefficients)")
    with np.errstate(all="ignore"):
        fitted = pd.Series(A @ zcoef, index=data.index, name="tp_fair_value")
    latest = X[cols].dropna().iloc[-1]
    fair_now = float(const + np.sum(latest.to_numpy() * slopes))
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
    elif target == "current":
        # Keep today's term premium — including the fiscal / risk premia the market already prices
        base, info = float(df["us_tp"].dropna().iloc[-1]), {"method": "hold_current"}
    elif target == "regression":
        try:
            base, rinfo = regression_target(df, cfg.get("drivers"), fit)
            info = {"method": "regression", **rinfo}
        except ValueError as exc:
            logger.warning("TP regression unavailable (%s); using historical mean", exc)
            target = "historical"
    if not isinstance(target, (int, float)) and target not in ("regression", "current"):
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


# --------------------------------------------------------------------------- fiscal elasticity
# Markets are forward-looking: the known baseline (e.g. CBO's rising debt path) is already in
# today's yields. Only REVISIONS to the fiscal outlook move yields. Literature elasticities for a
# revision to *projected* deficits / debt (Laubach 2009, JEEA; Gamber & Seliski 2019, CBO):
# ~+20–30 bp per 1 pp of GDP wider projected deficit, ~+2–4 bp per 1 pp higher projected debt/GDP.
# The two measures overlap (deficits accumulate into debt), so "average" uses their mean.
FISCAL_DEFAULTS = {
    "bps_per_pp_deficit": 25.0,
    "bps_per_pp_debt": 3.0,
    "measure": "average",          # "deficit" | "debt" | "average"
}


def _latest(df: pd.DataFrame, col: str) -> float | None:
    if col in df and df[col].notna().any():
        return float(df[col].dropna().iloc[-1])
    return None


def fiscal_impact(df: pd.DataFrame, cfg: dict | None, override: dict | None = None) -> tuple[float, dict]:
    """
    Fiscal premium (bps) at the horizon end from YOUR expected revision to the fiscal outlook
    versus the current baseline that markets already price (e.g. today's CBO projections):
      deficit_revision_pp: + = projected deficit wider than the baseline (pp of GDP)
      debt_revision_pp:    + = projected debt/GDP higher than the baseline (pp)
    No revision → 0 bp.
    """
    p = {**FISCAL_DEFAULTS, **(cfg or {}), **(override or {})}
    if p.get("enabled") is False:
        return 0.0, {"enabled": False}
    d_rev = p.get("deficit_revision_pp")
    debt_rev = p.get("debt_revision_pp")
    from_deficit = p["bps_per_pp_deficit"] * float(d_rev or 0.0)
    from_debt = p["bps_per_pp_debt"] * float(debt_rev or 0.0)
    measure = p.get("measure", "average")
    if measure == "deficit":
        total = from_deficit
    elif measure == "debt":
        total = from_debt
    else:
        used = [v for v, set_ in ((from_deficit, bool(d_rev)), (from_debt, bool(debt_rev))) if set_]
        total = float(np.mean(used)) if used else 0.0
    info = {"measure": measure, "deficit_revision_pp": d_rev, "debt_revision_pp": debt_rev,
            "latest_balance_gdp": _latest(df, "deficit_gdp"), "latest_debt_gdp": _latest(df, "debt_gdp"),
            "from_deficit_bps": round(from_deficit, 1), "from_debt_bps": round(from_debt, 1),
            "bps_per_pp_deficit": p["bps_per_pp_deficit"], "bps_per_pp_debt": p["bps_per_pp_debt"],
            "fiscal_premium_bps": round(total, 1)}
    return float(total), info


def fiscal_path(horizon: int, bps: float) -> np.ndarray:
    """Fiscal premium phased in linearly over the horizon (outlook revisions arrive gradually)."""
    return np.linspace(0.0, bps / 100.0, horizon + 1)
