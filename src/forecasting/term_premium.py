"""
Term-premium component of the 10Y yield.

TP_{t+h} = target + (TP_t - target) * phi^h

target options (config `forecast.term_premium.target`):
  - a number (%)        → subjective view, e.g. 0.75
  - "historical"        → trailing N-year mean of the term premium (default)
  - "regression"        → conditional mean from OLS on drivers (deficit/GDP, rate volatility),
                          evaluated at user-specified driver assumptions
phi comes from an estimated AR(1) unless `halflife_months` is given.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import fit_ar1, ols, phi_from_halflife

logger = logging.getLogger(__name__)

DRIVERS = ["deficit_gdp", "us_10y_rvol_12m"]


def _drivers_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "deficit_gdp" in df:
        out["deficit_gdp"] = df["deficit_gdp"]
    if "us_10y_rvol" in df:
        out["us_10y_rvol_12m"] = df["us_10y_rvol"].rolling(12, min_periods=6).mean()
    return out


def regression_target(df: pd.DataFrame, assumptions: dict | None) -> tuple[float, dict]:
    X = _drivers_frame(df)
    cols = [c for c in DRIVERS if c in X]
    data = pd.concat([df["us_tp"], X[cols]], axis=1).dropna()
    if not cols or len(data) < 60:
        raise ValueError("insufficient driver data for term-premium regression")
    A = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
    coef, resid = ols(data["us_tp"].to_numpy(), A)
    latest = X[cols].dropna().iloc[-1].to_dict()
    point = {c: float((assumptions or {}).get(c, latest[c])) for c in cols}
    target = float(coef[0] + sum(coef[i + 1] * point[c] for i, c in enumerate(cols)))
    info = {"coef": dict(zip(["const"] + cols, np.round(coef, 4).tolist())),
            "driver_assumptions": point, "r2": float(1 - resid.var() / data["us_tp"].var())}
    return target, info


def tp_target(df: pd.DataFrame, cfg: dict) -> tuple[float, dict]:
    target = cfg.get("target", "historical")
    window = int(cfg.get("historical_window_years", 10))
    if isinstance(target, (int, float)):
        return float(target), {"method": "user"}
    if target == "regression":
        try:
            t, info = regression_target(df, cfg.get("drivers"))
            return t, {"method": "regression", **info}
        except ValueError as exc:
            logger.warning("TP regression unavailable (%s); using historical mean", exc)
    tp = df["us_tp"].dropna()
    tp = tp[tp.index >= tp.index[-1] - pd.DateOffset(years=window)]
    return float(tp.mean()), {"method": f"historical_{window}y"}


def tp_path(df: pd.DataFrame, horizon: int, cfg: dict | None) -> tuple[np.ndarray, dict]:
    cfg = cfg or {}
    tp = df["us_tp"].dropna()
    tp0 = float(tp.iloc[-1])
    target, info = tp_target(df, cfg)
    if cfg.get("halflife_months"):
        phi = phi_from_halflife(float(cfg["halflife_months"]))
    else:
        phi = fit_ar1(tp)["phi"]
    h = np.arange(horizon + 1)
    path = target + (tp0 - target) * phi ** h
    info.update({"tp0": round(tp0, 3), "target": round(target, 3), "phi": round(phi, 4)})
    return path, info
