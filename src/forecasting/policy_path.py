"""
Policy-rate paths for the Fed and the BoC.

Two sources:
  1. User scenarios from config (subjective views, e.g. FOMC dots / OIS-informed paths).
  2. An endogenous "model" path: inertial Taylor rule for the Fed,
     i_t = rho * i_{t-1} + (1 - rho) * rule_t,
     rule_t = r* + pi_t + a_pi (pi_t - pi*) + a_gap * okun * (u*_t - u_t),
     with inflation and unemployment decaying towards target / NAIRU (or following user paths).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import anchored_path, decay, ols

logger = logging.getLogger(__name__)

TAYLOR_DEFAULTS = {
    "r_star": 1.0,               # real neutral rate (%)
    "inflation_target": 2.0,     # PCE target (%)
    "a_pi": 0.5,                 # weight on inflation gap
    "a_gap": 1.0,                # weight on output gap ("balanced approach")
    "okun": 2.0,                 # output gap ≈ -okun * (u - u*)
    "rho": None,                 # monthly inertia; None = estimate from data
    "inflation_halflife": 12,    # months for core PCE to close half its gap to target
    "unemployment_halflife": 18,
    "floor": 0.125,              # effective lower bound
}


def taylor_rule(pi, u, ustar, p: dict) -> np.ndarray:
    pi, u, ustar = map(lambda a: np.asarray(a, dtype=float), (pi, u, ustar))
    rule = (p["r_star"] + pi + p["a_pi"] * (pi - p["inflation_target"])
            + p["a_gap"] * p["okun"] * (ustar - u))
    return np.maximum(rule, p["floor"])


def estimate_inertia(df: pd.DataFrame, p: dict, start: str = "1990-01-01") -> float:
    """OLS of (i_t - rule_t) on (i_{t-1} - rule_t), no constant. Clipped to [0.85, 0.99]."""
    cols = ["fed_funds", "core_pce_yoy", "unrate", "nrou"]
    if not all(c in df for c in cols):
        return 0.95
    d = df.loc[start:, cols].dropna()
    if len(d) < 60:
        return 0.95
    rule = taylor_rule(d["core_pce_yoy"], d["unrate"], d["nrou"], p)
    i = d["fed_funds"].to_numpy()
    y = i[1:] - rule[1:]
    x = i[:-1] - rule[1:]
    (rho,), _ = ols(y, x[:, None])
    return float(np.clip(rho, 0.85, 0.99))


def taylor_path(df: pd.DataFrame, index: pd.DatetimeIndex, cfg: dict) -> tuple[np.ndarray, dict]:
    """Endogenous Fed funds path over `index` (h = 0..H). Returns (path, parameters used)."""
    p = {**TAYLOR_DEFAULTS, **(cfg or {})}
    if p.get("rho") is None:
        p["rho"] = estimate_inertia(df, p)

    last = df.iloc[-1]
    h = np.arange(len(index))
    i0 = float(last["fed_funds"])

    pi0 = float(df["core_pce_yoy"].dropna().iloc[-1]) if "core_pce_yoy" in df else p["inflation_target"]
    u0 = float(df["unrate"].dropna().iloc[-1]) if "unrate" in df else 4.2
    ustar = float(df["nrou"].dropna().iloc[-1]) if "nrou" in df and df["nrou"].notna().any() else u0

    pi_path = (anchored_path(p.get("inflation_path"), pi0, index) if p.get("inflation_path")
               else p["inflation_target"] + (pi0 - p["inflation_target"]) * decay(h, p["inflation_halflife"]))
    u_path = (anchored_path(p.get("unemployment_path"), u0, index) if p.get("unemployment_path")
              else ustar + (u0 - ustar) * decay(h, p["unemployment_halflife"]))

    rule = taylor_rule(pi_path, u_path, ustar, p)
    path = np.empty(len(index))
    path[0] = i0
    for k in range(1, len(index)):
        path[k] = p["rho"] * path[k - 1] + (1 - p["rho"]) * rule[k]

    used = {k: p[k] for k in ("r_star", "inflation_target", "a_pi", "a_gap", "okun", "rho")}
    used.update({"pi0": round(pi0, 2), "u0": u0, "ustar": round(ustar, 2)})
    return path, used


def mean_reverting_path(i0: float, target: float, index: pd.DatetimeIndex, rho: float) -> np.ndarray:
    """Partial adjustment towards a fixed target (used for the model BoC path)."""
    h = np.arange(len(index))
    return target + (i0 - target) * rho ** h


def boc_model_path(df: pd.DataFrame, fed_path: np.ndarray, index: pd.DatetimeIndex,
                   long_run_boc: float, rho: float, fed_passthrough: float = 0.5) -> np.ndarray:
    """
    Model BoC path: partial adjustment towards long-run BoC neutral, plus a share of the
    Fed's projected move (Canadian policy historically co-moves with the Fed).
    """
    base = mean_reverting_path(float(df["boc_rate"].iloc[-1]), long_run_boc, index, rho)
    return np.maximum(base + fed_passthrough * (fed_path - fed_path[0]), 0.125)
