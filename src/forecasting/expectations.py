"""
Expectations (risk-neutral) component of the 10Y yield.

At each future month h, the risk-neutral 10Y yield is approximated by the average expected
short rate over the bond's life (months h .. h+119):
  - months inside the policy-path horizon use the scenario path;
  - beyond it the short rate converges exponentially to the nominal neutral rate
    (long-run Fed funds) with a configurable half-life.

A "basis" term reconciles the construct with the observed risk-neutral yield at h = 0
(observed us_rny - construct) and decays over time, so the forecast starts exactly at the
current market level instead of jumping.
"""

from __future__ import annotations

import numpy as np

from .utils import decay

MATURITY_MONTHS = 120


def expected_avg_short_rate(path: np.ndarray, neutral: float, convergence_halflife: float,
                            maturity: int = MATURITY_MONTHS) -> np.ndarray:
    """
    path: expected policy rate for months 0..H.
    Returns the average expected short rate over the next `maturity` months, for each h = 0..H.
    """
    H = len(path) - 1
    k = np.arange(1, H + maturity + 1)
    tail = neutral + (path[-1] - neutral) * decay(k, convergence_halflife)
    full = np.concatenate([path, tail])
    csum = np.concatenate([[0.0], np.cumsum(full)])
    h = np.arange(H + 1)
    return (csum[h + maturity] - csum[h]) / maturity


def basis_path(observed_rny0: float | None, construct0: float, horizon: int,
               halflife: float) -> np.ndarray:
    """Gap between observed and constructed expectations at h = 0, decayed over the horizon."""
    if observed_rny0 is None or np.isnan(observed_rny0):
        return np.zeros(horizon + 1)
    return (observed_rny0 - construct0) * decay(np.arange(horizon + 1), halflife)
