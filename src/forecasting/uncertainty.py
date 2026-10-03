"""
Forecast uncertainty: joint moving-block bootstrap of monthly yield changes.

Monthly changes of (US 10Y, Canada 10Y) are resampled in blocks (preserving autocorrelation
and US–Canada co-movement), demeaned, cumulated, and added to a scenario path drawn with
the scenario probabilities. Quantiles across simulations give the fan chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]


def block_bootstrap(changes: np.ndarray, horizon: int, n_sims: int, block: int,
                    rng: np.random.Generator) -> np.ndarray:
    """changes: (T, k). Returns cumulative shocks of shape (n_sims, horizon + 1, k), zero at h=0."""
    T, k = changes.shape
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, T - block + 1, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_sims, -1)[:, :horizon]
    draws = changes[idx]                      # (n_sims, horizon, k)
    cum = np.cumsum(draws, axis=1)
    return np.concatenate([np.zeros((n_sims, 1, k)), cum], axis=1)


def simulate_fan(df: pd.DataFrame, scenario_paths: dict[str, np.ndarray], probs: dict[str, float],
                 cfg: dict | None) -> dict[str, np.ndarray]:
    """
    scenario_paths: {name: array (H+1, 2)} with columns (us_10y, canada_10y).
    Returns {"us_10y": sims (n, H+1), "canada_10y": sims (n, H+1)}.
    """
    cfg = cfg or {}
    rng = np.random.default_rng(cfg.get("seed", 42))
    n_sims = int(cfg.get("n_sims", 5000))
    block = int(cfg.get("block_months", 6))
    years = int(cfg.get("sample_years", 15))
    scale = float(cfg.get("scale", 1.0))

    hist = df[["us_10y", "canada_10y"]].dropna()
    hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=years)]
    changes = hist.diff().dropna().to_numpy()
    changes = (changes - changes.mean(axis=0)) * scale

    names = list(scenario_paths)
    H = scenario_paths[names[0]].shape[0] - 1
    p = np.array([probs[n] for n in names])
    pick = rng.choice(len(names), size=n_sims, p=p / p.sum())
    base = np.stack([scenario_paths[n] for n in names])[pick]   # (n, H+1, 2)
    sims = base + block_bootstrap(changes, H, n_sims, block, rng)
    return {"us_10y": sims[:, :, 0], "canada_10y": sims[:, :, 1]}


def quantile_frame(sims: np.ndarray, index: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    q = np.quantile(sims, QUANTILES, axis=0).T
    cols = [f"{prefix}_p{int(x * 100):02d}" for x in QUANTILES]
    return pd.DataFrame(q, index=index, columns=cols)
