"""Compute analytical metrics used in the forecast report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_fetcher import last_quarter_window


def _pct_rank(series: pd.Series, value: float) -> float:
    """Return the percentile rank of `value` within `series` (0–100)."""
    return float((series < value).mean() * 100)


def _last_n_years(series: pd.Series, years: int) -> pd.Series:
    cutoff = series.index[-1] - pd.DateOffset(years=years)
    return series[series.index >= cutoff]


def _stats(series: pd.Series, label: str) -> dict:
    clean = series.dropna()
    return {
        f"{label}_current": round(float(clean.iloc[-1]), 3),
        f"{label}_mean_5y": round(float(_last_n_years(clean, 5).mean()), 3),
        f"{label}_mean_10y": round(float(_last_n_years(clean, 10).mean()), 3),
        f"{label}_mean_20y": round(float(clean.mean()), 3),
        f"{label}_pct_rank_20y": round(_pct_rank(clean, float(clean.iloc[-1])), 1),
    }


def run_analysis(df: pd.DataFrame, config: dict) -> dict:
    """
    Compute all metrics needed for chart annotations and AI commentary.
    Returns a structured dict consumed by charts.py and commentary.py.
    """
    results: dict = {}

    # --- 1. Spread: US 10Y minus Fed Funds Rate ---
    df["spread_10y_ffr"] = df["us_10y"] - df["fed_funds"]
    results.update(_stats(df["spread_10y_ffr"], "spread_10y_ffr"))

    # --- 2. Term premium (Kim-Wright THREEFYTP10; column name kept as acm_tp) ---
    results.update(_stats(df["acm_tp"], "acm_tp"))

    # --- 3. Canada–US spread ---
    df["spread_can_us"] = df["canada_10y"] - df["us_10y"]
    results.update(_stats(df["spread_can_us"], "spread_can_us"))

    # --- 3b. Canada 10Y minus BoC overnight rate ---
    df["spread_canada_10y_boc"] = df["canada_10y"] - df["boc_rate"]
    results.update(_stats(df["spread_canada_10y_boc"], "spread_canada_10y_boc"))

    # --- 4. US 10Y level stats ---
    results.update(_stats(df["us_10y"], "us_10y"))

    # --- 5. Canada 10Y level stats ---
    results.update(_stats(df["canada_10y"], "canada_10y"))

    # --- 6. 10Y Breakeven inflation ---
    results.update(_stats(df["us_10y_bei"], "us_10y_bei"))

    # --- 7. Last-quarter movement ---
    qs, qe = last_quarter_window(df)
    for col, label in [("us_10y", "us_10y_q"), ("canada_10y", "canada_10y_q")]:
        window = df[col].loc[qs:qe].dropna()
        if len(window) > 0:
            results[f"{label}_start"] = round(float(window.iloc[0]), 3)
            results[f"{label}_end"] = round(float(window.iloc[-1]), 3)
            results[f"{label}_high"] = round(float(window.max()), 3)
            results[f"{label}_low"] = round(float(window.min()), 3)
            results[f"{label}_change_bps"] = round(
                (window.iloc[-1] - window.iloc[0]) * 100, 1
            )
    results["last_quarter_start"] = str(qs.date())
    results["last_quarter_end"] = str(qe.date())

    # --- 8. Methodology-implied fair value ---
    long_run_ffr = config.get("long_run_fed_funds", 3.0)
    long_run_boc = config.get("long_run_boc_rate", 2.75)
    long_run_real_gdp = config.get("long_run_real_gdp_growth", 2.15)
    long_run_pce = config.get("long_run_pce_inflation", 2.0)
    us_target = config["us_10y"]["target"]
    results["methodology"] = {
        # Framework 1: r* + historical 10Y–FFR spread (TP implicit in spread)
        "long_run_fed_funds": long_run_ffr,
        "implied_10y_using_5y_avg_spread": round(
            long_run_ffr + results["spread_10y_ffr_mean_5y"], 3
        ),
        "implied_10y_using_10y_avg_spread": round(
            long_run_ffr + results["spread_10y_ffr_mean_10y"], 3
        ),
        "implied_10y_using_20y_avg_spread": round(
            long_run_ffr + results["spread_10y_ffr_mean_20y"], 3
        ),
        # Framework 2: term-premium decomposition — expected short rate + explicit term premium
        # (long-run FFR as the expected short rate converging over 10Y horizon)
        "implied_10y_acm_10y_avg_tp": round(long_run_ffr + results["acm_tp_mean_10y"], 3),
        "implied_10y_acm_20y_avg_tp": round(long_run_ffr + results["acm_tp_mean_20y"], 3),
        "implied_10y_acm_current_tp": round(long_run_ffr + results["acm_tp_current"], 3),
        # Framework 3: Real GDP growth + PCE inflation (Fisher / neutral rate approach)
        "long_run_real_gdp_growth": long_run_real_gdp,
        "long_run_pce_inflation": long_run_pce,
        "implied_10y_gdp_inflation": round(long_run_real_gdp + long_run_pce, 3),
        "user_target_us_10y": us_target,
        "user_target_canada_10y": config["canada_10y"]["target"],
        # Canada anchor 1: BoC r* + historical Canada 10Y–BoC spread
        "long_run_boc_rate": long_run_boc,
        "implied_canada_10y_using_5y_boc_spread": round(
            long_run_boc + results["spread_canada_10y_boc_mean_5y"], 3
        ),
        "implied_canada_10y_using_10y_boc_spread": round(
            long_run_boc + results["spread_canada_10y_boc_mean_10y"], 3
        ),
        "implied_canada_10y_using_20y_boc_spread": round(
            long_run_boc + results["spread_canada_10y_boc_mean_20y"], 3
        ),
        # Canada anchor 2: US 10Y target + historical Canada–US spread
        "implied_canada_10y_using_5y_canus_spread": round(
            us_target + results["spread_can_us_mean_5y"], 3
        ),
        "implied_canada_10y_using_10y_canus_spread": round(
            us_target + results["spread_can_us_mean_10y"], 3
        ),
    }

    # Store the df with derived columns for charting
    results["_df"] = df

    return results
