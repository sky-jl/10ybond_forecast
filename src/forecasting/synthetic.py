"""
Synthetic monthly dataset with the same columns as data_fetcher.build_monthly_dataset().
Used for unit tests and for a demo run when FRED / Bank of Canada are unreachable.
Numbers are NOT real market data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_fetcher import finalize_monthly_dataset


def make_synthetic_monthly(start: str = "1995-01-31", end: str = "2026-09-30",
                           seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="ME")
    n = len(idx)

    # Macro block
    infl = np.empty(n); u = np.empty(n)
    infl[0], u[0] = 2.0, 5.0
    for t in range(1, n):
        infl[t] = 2.0 + 0.97 * (infl[t - 1] - 2.0) + rng.normal(0, 0.15)
        u[t] = 5.0 + 0.98 * (u[t - 1] - 5.0) - 0.05 * (infl[t] - 2.0) + rng.normal(0, 0.12)
    nrou = np.full(n, 4.6)

    # Policy rates: inertial Taylor rule
    ffr = np.empty(n); boc = np.empty(n)
    ffr[0], boc[0] = 5.0, 5.0
    for t in range(1, n):
        rule = max(1.0 + infl[t] + 0.5 * (infl[t] - 2.0) + 2.0 * (nrou[t] - u[t]), 0.125)
        ffr[t] = max(0.95 * ffr[t - 1] + 0.05 * rule + rng.normal(0, 0.08), 0.05)
        boc[t] = max(0.95 * boc[t - 1] + 0.05 * (ffr[t] - 0.3) + rng.normal(0, 0.08), 0.25)

    # Term premium AR(1)
    tp = np.empty(n); tp[0] = 1.2
    for t in range(1, n):
        tp[t] = 0.6 + 0.97 * (tp[t - 1] - 0.6) + rng.normal(0, 0.12)

    # Risk-neutral yield: average expected short rate converging to 3%
    k = np.arange(120)
    w = np.exp(-np.log(2) * k / 36).mean()
    rny = 3.0 + (ffr - 3.0) * w
    us10 = rny + tp + rng.normal(0, 0.05, n)

    spread = np.empty(n); spread[0] = 0.2
    pdiff = boc - ffr
    for t in range(1, n):
        spread[t] = -0.05 + 0.93 * spread[t - 1] + 0.04 * pdiff[t] + rng.normal(0, 0.07)
    ca10 = us10 + spread

    df = pd.DataFrame({
        "us_3m": ffr + 0.05, "us_1y": 0.7 * ffr + 0.3 * us10, "us_2y": 0.55 * ffr + 0.45 * us10,
        "us_5y": 0.25 * ffr + 0.75 * us10, "us_7y": 0.1 * ffr + 0.9 * us10, "us_10y": us10,
        "us_30y": us10 + 0.35, "fed_funds": ffr, "kw_tp": tp, "acm_tp": tp, "acm_rny": rny,
        "us_10y_real": us10 - 2.2, "us_10y_bei": np.full(n, 2.2) + 0.1 * (infl - 2),
        "us_10y_rvol": np.abs(rng.normal(6, 1.5, n)),
        "core_pce": 100 * np.exp(np.cumsum(infl / 1200)), "unrate": u, "nrou": nrou,
        "deficit_gdp": -4.0 + rng.normal(0, 0.5, n),
        "canada_10y": ca10, "canada_2y": 0.5 * boc + 0.5 * ca10, "boc_rate": boc,
    }, index=idx)
    df["core_pce_yoy"] = infl
    return finalize_monthly_dataset(df)
