"""
Pseudo out-of-sample backtest.

At each monthly origin t the model and benchmarks see only data up to t (macro series lagged
one extra month to mimic publication delays) and forecast h = 1..24 months ahead.

The structural model runs with its endogenous Taylor-rule policy path and a trailing,
market-based neutral rate: user scenarios and overlays cannot be backtested.

Benchmarks:
  rw        random walk (no change) — the hard-to-beat standard for 10Y yields
  ar1       AR(1) on the yield level, re-estimated each origin
  forward   forward rate implied by the curve (expectations hypothesis)
  dns       dynamic Nelson–Siegel (Diebold–Li 2006), US only, needs the full curve
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .model import project_scenarios
from .utils import fit_ar1

logger = logging.getLogger(__name__)

EVAL_HORIZONS = [3, 6, 12, 24]
MACRO_COLS = ["core_pce", "core_pce_yoy", "unrate", "nrou", "deficit_gdp"]
DNS_MATURITIES = {"us_3m": 3, "us_1y": 12, "us_2y": 24, "us_5y": 60, "us_7y": 84,
                  "us_10y": 120, "us_30y": 360}
DNS_LAMBDA = 0.0609


# ---------------------------------------------------------------------------- benchmarks
def _ar1_forecast(series: pd.Series, H: int, years: int = 20) -> np.ndarray:
    s = series.dropna()
    s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
    p = fit_ar1(s)
    return p["mu"] + (float(s.iloc[-1]) - p["mu"]) * p["phi"] ** np.arange(1, H + 1)


def _forward_forecast(row: pd.Series, H: int, short: str, mid: str, mid_m: int,
                      long: str = None) -> np.ndarray:
    """10Y yield h months forward from a piecewise-linear curve (0m, mid, [5y], 10y)."""
    knots_m = [0, mid_m] + ([60] if long else []) + [120]
    knots_y = [row[short], row[mid]] + ([row[long]] if long else []) + [row["_y10"]]
    slope_end = (knots_y[-1] - knots_y[-2]) / (knots_m[-1] - knots_m[-2])

    def y(m):
        if m <= 120:
            return float(np.interp(m, knots_m, knots_y))
        return knots_y[-1] + slope_end * (m - 120)

    h = np.arange(1, H + 1)
    return np.array([((120 + k) * y(120 + k) - k * y(k)) / 120 for k in h])


def _ns_loadings(tau: np.ndarray, lam: float = DNS_LAMBDA) -> np.ndarray:
    x = lam * tau
    l1 = (1 - np.exp(-x)) / x
    return np.column_stack([np.ones_like(tau), l1, l1 - np.exp(-x)])


def _dns_forecast(curve: pd.DataFrame, H: int, years: int = 15) -> np.ndarray:
    curve = curve[curve.index >= curve.index[-1] - pd.DateOffset(years=years)].dropna()
    tau = np.array([DNS_MATURITIES[c] for c in curve.columns], dtype=float)
    L = _ns_loadings(tau)
    betas = np.linalg.lstsq(L, curve.to_numpy().T, rcond=None)[0].T   # (T, 3)
    l10 = _ns_loadings(np.array([120.0]))[0]
    b_path = np.empty((H, 3))
    for j in range(3):
        p = fit_ar1(pd.Series(betas[:, j]))
        b_path[:, j] = p["mu"] + (betas[-1, j] - p["mu"]) * p["phi"] ** np.arange(1, H + 1)
    # Anchor to the observed 10Y at origin (remove the cross-sectional fitting error)
    fit_err = curve["us_10y"].iloc[-1] - betas[-1] @ l10
    return b_path @ l10 + fit_err


# ---------------------------------------------------------------------------- engine
def _vintage(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    v = df.loc[:t].copy()
    for c in MACRO_COLS:
        if c in v:
            v[c] = v[c].shift(1)
    return v


def _model_config(config: dict, H: int) -> dict:
    f = dict(config.get("forecast", {}))
    bt = f.get("backtest", {}) or {}
    return {
        "horizon_months": H,
        "neutral_mode": "trailing",
        "trailing_neutral_years": bt.get("trailing_neutral_years", 10),
        "expectations": f.get("expectations", {}),
        "taylor_rule": {k: v for k, v in (f.get("taylor_rule") or {}).items()
                        if k not in ("inflation_path", "unemployment_path")},
        "term_premium": {"target": "historical",
                         "historical_window_years": (f.get("term_premium") or {}).get("historical_window_years", 10)},
        "canada_spread": {"target": "model",
                          "sample_start": (f.get("canada_spread") or {}).get("sample_start", "2000-01-01")},
        "boc_fed_passthrough": f.get("boc_fed_passthrough", 0.5),
        "scenarios": [],
        "model_scenario": {"include": True, "probability": 1.0, "name": "model"},
        "overlay": {},
    }


def run_backtest(df: pd.DataFrame, config: dict, start: str | None = None,
                 horizon: int = 24, step: int = 1) -> dict:
    """Returns {"errors": long DataFrame, "summary": DataFrame}."""
    bt = (config.get("forecast", {}) or {}).get("backtest", {}) or {}
    start = pd.Timestamp(start or bt.get("start", "2005-01-31"))
    mcfg = _model_config(config, horizon)

    df = df.copy()
    df["_y10"] = df["us_10y"]
    dns_cols = [c for c in DNS_MATURITIES if c in df]
    has_dns = len(dns_cols) >= 5
    origins = [t for t in df.index if t >= start][::step]
    last = df.index[-1]

    records = []
    failures: list[tuple[pd.Timestamp, str]] = []
    for t in origins:
        v = _vintage(df, t)
        if len(v) < 120:
            continue
        H = min(horizon, int(round((last - t).days / 30.44)))
        if H < 1:
            continue
        fut = df.loc[df.index > t].iloc[:H]
        preds: dict[tuple[str, str], np.ndarray] = {}
        try:
            sc, _, _, _ = project_scenarios(v, config, {**mcfg, "horizon_months": H})
            m = next(iter(sc.values()))
            preds[("us_10y", "model")] = m["us_10y"].to_numpy()[1:]
            preds[("canada_10y", "model")] = m["canada_10y"].to_numpy()[1:]
            # Variant: US model + today's Canada–US spread held flat
            preds[("canada_10y", "model_flat_spread")] = (
                m["us_10y"].to_numpy()[1:] + float(v["spread_can_us"].iloc[-1]))
        except Exception as exc:
            failures.append((t, repr(exc)))
        row = v.iloc[-1]
        for tgt in ("us_10y", "canada_10y"):
            preds[(tgt, "rw")] = np.full(H, row[tgt])
            preds[(tgt, "ar1")] = _ar1_forecast(v[tgt], H)
        r = row.copy()
        if {"us_2y", "us_5y"} <= set(v.columns) and r[["us_2y", "us_5y"]].notna().all():
            preds[("us_10y", "forward")] = _forward_forecast(r, H, "fed_funds", "us_2y", 24, "us_5y")
        if "canada_2y" in v and pd.notna(r.get("canada_2y")):
            r = r.copy()
            r["_y10"] = r["canada_10y"]
            preds[("canada_10y", "forward")] = _forward_forecast(r, H, "boc_rate", "canada_2y", 24)
        if has_dns:
            try:
                preds[("us_10y", "dns")] = _dns_forecast(v[dns_cols], H)
            except Exception as exc:
                logger.debug("dns failed at %s: %s", t.date(), exc)

        for (tgt, name), p in preds.items():
            actual = fut[tgt].to_numpy()
            for k in range(min(H, len(actual))):
                if np.isnan(actual[k]) or np.isnan(p[k]):
                    continue
                records.append((t, tgt, name, k + 1, p[k], actual[k], float(row[tgt])))

    if failures:
        logger.warning("Structural model failed at %d of %d origins (%s → %s); first error: %s",
                       len(failures), len(origins), failures[0][0].date(), failures[-1][0].date(),
                       failures[0][1])
    errors = pd.DataFrame(records, columns=["origin", "target", "model", "h", "pred", "actual", "origin_value"])
    errors["error"] = errors["pred"] - errors["actual"]
    return {"errors": errors, "summary": summarize(errors)}


# ---------------------------------------------------------------------------- metrics
def _dm_pvalue(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> float:
    """Diebold–Mariano test (squared loss, Newey–West with h-1 lags, HLN small-sample factor)."""
    d = e_model ** 2 - e_bench ** 2
    n = len(d)
    if n < 10:
        return float("nan")
    dc = d - d.mean()
    var = dc @ dc / n
    for lag in range(1, h):
        w = 1 - lag / h
        var += 2 * w * (dc[lag:] @ dc[:-lag]) / n
    if var <= 0:
        return float("nan")
    stat = d.mean() / math.sqrt(var / n)
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(math.erfc(abs(stat) / math.sqrt(2)))  # two-sided normal p-value


def summarize(errors: pd.DataFrame, horizons: list[int] = EVAL_HORIZONS) -> pd.DataFrame:
    rows = []
    for (tgt, h), g in errors[errors["h"].isin(horizons)].groupby(["target", "h"]):
        rw = g[g["model"] == "rw"].set_index("origin")["error"]
        for name, gm in g.groupby("model"):
            e = gm.set_index("origin")["error"]
            common = e.index.intersection(rw.index)
            pred_dir = np.sign(gm["pred"] - gm["origin_value"])
            act_dir = np.sign(gm["actual"] - gm["origin_value"])
            rmse = float(np.sqrt((e ** 2).mean()))
            rw_rmse = float(np.sqrt((rw.loc[common] ** 2).mean()))
            rows.append({
                "target": tgt, "h": h, "model": name, "n": len(e),
                "rmse": rmse, "mae": float(e.abs().mean()), "bias": float(e.mean()),
                "rmse_vs_rw": rmse / rw_rmse if rw_rmse > 0 else np.nan,
                "hit_rate": float((pred_dir == act_dir)[pred_dir != 0].mean()) if name != "rw" else np.nan,
                "dm_pvalue_vs_rw": _dm_pvalue(e.loc[common].to_numpy(), rw.loc[common].to_numpy(), h)
                if name != "rw" else np.nan,
            })
    return pd.DataFrame(rows).sort_values(["target", "h", "rmse"]).reset_index(drop=True)
