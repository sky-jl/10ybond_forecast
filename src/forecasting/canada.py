"""
Canada 10Y = US 10Y + Canada–US spread.

Spread model (monthly OLS):  s_t = a + rho * s_{t-1} + b * pd_t + e_t,
where pd_t = BoC overnight rate − Fed funds (policy differential). Projected forward along
the scenario policy paths. A numeric `canada_spread.target` replaces the model with simple
mean reversion towards the user's spread view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import ols, phi_from_halflife


def fit_spread_model(df: pd.DataFrame, sample_start: str = "2000-01-01") -> dict:
    d = df.loc[sample_start:, ["spread_can_us", "policy_diff"]].dropna()
    if len(d) < 36:
        raise ValueError(f"only {len(d)} months of Canada spread / policy data")
    s = d["spread_can_us"].to_numpy()
    pdiff = d["policy_diff"].to_numpy()
    X = np.column_stack([np.ones(len(s) - 1), s[:-1], pdiff[1:]])
    (a, rho, b), resid = ols(s[1:], X)
    rho = float(np.clip(rho, 0.0, 0.99))
    return {"a": float(a), "rho": rho, "b": float(b), "sigma": float(resid.std()),
            "long_run_spread_at_current_pd": float((a + b * pdiff[-1]) / (1 - rho))}


def spread_path(df: pd.DataFrame, policy_diff_path: np.ndarray, params: dict,
                cfg: dict | None) -> np.ndarray:
    cfg = cfg or {}
    s0 = float(df["spread_can_us"].dropna().iloc[-1])
    H = len(policy_diff_path) - 1
    target = cfg.get("target", "model")
    if isinstance(target, (int, float)):
        phi = phi_from_halflife(cfg["halflife_months"]) if cfg.get("halflife_months") else params["rho"]
        return target + (s0 - target) * phi ** np.arange(H + 1)
    out = np.empty(H + 1)
    out[0] = s0
    for k in range(1, H + 1):
        out[k] = params["a"] + params["rho"] * out[k - 1] + params["b"] * policy_diff_path[k]
    return out


def rolling_beta(df: pd.DataFrame, window: int = 36) -> pd.Series:
    """Beta of monthly ΔCanada10Y on ΔUS10Y over a rolling window."""
    d = df[["canada_10y", "us_10y"]].diff().dropna()
    cov = d["canada_10y"].rolling(window).cov(d["us_10y"])
    var = d["us_10y"].rolling(window).var()
    return (cov / var).rename("beta_ca_us")
