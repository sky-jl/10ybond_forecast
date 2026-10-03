"""Shared helpers: horizon indexing, decay factors, OLS and quarter-keyed user inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def decay(h, halflife_months: float):
    """Fraction of a gap remaining after h months, given a half-life in months."""
    if halflife_months is None or halflife_months <= 0:
        return np.zeros_like(np.asarray(h, dtype=float))
    return np.exp(-np.log(2) * np.asarray(h, dtype=float) / halflife_months)


def halflife_from_phi(phi: float) -> float:
    phi = float(np.clip(phi, 1e-6, 0.9999))
    return float(np.log(0.5) / np.log(phi))


def phi_from_halflife(halflife_months: float) -> float:
    return float(0.5 ** (1.0 / halflife_months))


def horizon_index(last_obs: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Month-end dates for h = 0..horizon, h = 0 being the last observed month."""
    start = pd.Timestamp(last_obs) + pd.offsets.MonthEnd(0)
    return pd.date_range(start, periods=horizon + 1, freq="ME")


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least squares; returns (coefficients, residuals)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def fit_ar1(series: pd.Series) -> dict:
    """x_t = c + phi x_{t-1} + e_t. Returns mean, phi, half-life (months) and residual sd."""
    x = series.dropna().to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    (c, phi), resid = ols(x[1:], X)
    phi = float(np.clip(phi, 0.0, 0.995))
    mu = float(c / (1 - phi)) if phi < 0.995 else float(x.mean())
    return {"mu": mu, "phi": phi, "halflife": halflife_from_phi(phi), "sigma": float(resid.std())}


def parse_period_key(key) -> pd.Timestamp:
    """Accept '2027Q2', '2027-06', '2027-06-30' (or datetime) and return the month-end date."""
    k = str(key).strip().upper()
    if "Q" in k:
        return pd.Period(k, freq="Q").end_time.normalize() + pd.offsets.MonthEnd(0)
    return pd.Timestamp(k) + pd.offsets.MonthEnd(0)


def anchored_path(
    points: dict | None, start_value: float, index: pd.DatetimeIndex, hold_last: bool = True
) -> np.ndarray:
    """
    Build a monthly path over `index` (h = 0..H) by linear interpolation between user anchor
    points {period: value}. h = 0 is pinned to `start_value`; after the last anchor the value
    is held flat. Anchors before the forecast start are ignored.
    """
    anchors = {index[0]: float(start_value)}
    for k, v in (points or {}).items():
        d = parse_period_key(k)
        if d > index[0]:
            anchors[d] = float(v)
    s = pd.Series(anchors).sort_index()
    full = s.reindex(s.index.union(index)).interpolate(method="time")
    if hold_last:
        full = full.ffill()
    return full.reindex(index).to_numpy(dtype=float)


def overlay_path(points: dict | None, index: pd.DatetimeIndex) -> np.ndarray:
    """Judgmental overlay in bps at anchor periods -> monthly path in percentage points (0 at h=0)."""
    if not points:
        return np.zeros(len(index))
    return anchored_path(points, 0.0, index) / 100.0
