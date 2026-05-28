"""Fetch yield and rate data from FRED and Bank of Canada VALET API."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd
import requests
from fredapi import Fred

logger = logging.getLogger(__name__)

# FRED series used in this program
FRED_SERIES = {
    "us_10y": "DGS10",          # US 10Y Treasury yield (daily)
    "fed_funds": "DFF",          # Fed Funds Effective Rate (daily)
    "acm_tp": "THREEFYTP10",     # 10Y term premium, Kim-Wright model (daily)
    "us_10y_bei": "T10YIE",      # 10Y Breakeven Inflation (daily)
}

# Bank of Canada VALET series
BOC_SERIES = {
    "canada_10y": "BD.CDN.10YR.DQ.YLD",  # GoC 10Y benchmark bond yield (daily)
    "boc_rate": "B114039",                 # BoC overnight target rate (change dates only, ffilled)
}

BOC_VALET_BASE = "https://www.bankofcanada.ca/valet/observations/{series}/json"


def fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    api_key: str,
) -> pd.Series:
    """Return a daily pd.Series for a FRED series, forward-filled for weekends."""
    fred = Fred(api_key=api_key)
    data = fred.get_series(series_id, observation_start=start, observation_end=end)
    data.name = series_id
    data = data.asfreq("B").ffill()  # align to business days, forward-fill gaps
    return data


def fetch_boc_series(series_id: str, start: str, end: str) -> pd.Series:
    """Return a daily pd.Series from the Bank of Canada VALET API (no auth required)."""
    url = BOC_VALET_BASE.format(series=series_id)
    params = {"start_date": start, "end_date": end}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    observations = payload.get("observations", [])
    records = {}
    for obs in observations:
        dt = obs.get("d")
        val_dict = obs.get(series_id, {})
        val = val_dict.get("v")
        if dt and val is not None:
            try:
                records[pd.Timestamp(dt)] = float(val)
            except (ValueError, TypeError):
                continue

    series = pd.Series(records, name=series_id)
    series.index = pd.DatetimeIndex(series.index)
    series = series.asfreq("B").ffill()
    return series


def build_dataset(config: dict, lookback_years: int = 20) -> pd.DataFrame:
    """
    Fetch all series and return a single DataFrame aligned to business-day index.
    Covers the last `lookback_years` of history for chart/average purposes.
    """
    api_key = config["fred_api_key"]
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=365 * lookback_years)).isoformat()

    logger.info("Fetching data from %s to %s", start, end)

    frames: dict[str, pd.Series] = {}

    for name, series_id in FRED_SERIES.items():
        logger.info("  FRED: %s (%s)", name, series_id)
        frames[name] = fetch_fred_series(series_id, start, end, api_key)

    for name, series_id in BOC_SERIES.items():
        logger.info("  BoC VALET: %s (%s)", name, series_id)
        frames[name] = fetch_boc_series(series_id, start, end)

    df = pd.DataFrame(frames)
    # Drop rows where all yield series are NaN (e.g., early gaps in BoC data)
    df = df.dropna(subset=["us_10y", "fed_funds"], how="all")
    logger.info("Dataset shape: %s", df.shape)
    return df


def last_quarter_window(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) timestamps for the most recently completed calendar quarter."""
    today = pd.Timestamp.today().normalize()
    q = (today.month - 1) // 3  # 0=Q1, 1=Q2, 2=Q3, 3=Q4
    if q == 0:
        qs = pd.Timestamp(today.year - 1, 10, 1)
        qe = pd.Timestamp(today.year - 1, 12, 31)
    else:
        qs = pd.Timestamp(today.year, q * 3 - 2, 1)
        qe = pd.Timestamp(today.year, q * 3, 1) + pd.offsets.MonthEnd(0)
    # Clip to available data range
    qs = max(qs, df.index.min())
    qe = min(qe, df.index.max())
    return qs, qe
