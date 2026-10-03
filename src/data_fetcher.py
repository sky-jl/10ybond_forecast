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
    # NOTE: column kept as "acm_tp" for backward compatibility, but THREEFYTP10 is the
    # Kim-Wright (Fed Board) term premium, not ACM. True ACM comes from fetch_acm().
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


# ============================================================
# Monthly dataset for the 2-year forecast model (src/forecasting)
# ============================================================

# Market series: monthly average of daily observations
MONTHLY_FRED_MARKET = {
    "us_3m": "DGS3MO",
    "us_1y": "DGS1",
    "us_2y": "DGS2",
    "us_5y": "DGS5",
    "us_7y": "DGS7",
    "us_10y": "DGS10",
    "us_30y": "DGS30",
    "fed_funds": "DFF",
    "kw_tp": "THREEFYTP10",      # Kim-Wright 10Y term premium
    "us_10y_real": "DFII10",     # 10Y TIPS real yield
    "us_10y_bei": "T10YIE",      # 10Y breakeven inflation
}

# Macro series: last available value in the month
MONTHLY_FRED_MACRO = {
    "core_pce": "PCEPILFE",      # core PCE price index (monthly)
    "unrate": "UNRATE",          # unemployment rate (monthly)
    "nrou": "NROU",              # CBO natural rate of unemployment (quarterly)
    "deficit_gdp": "FYFSGDA188S",  # federal surplus/deficit % GDP (annual)
}

MONTHLY_BOC_MARKET = {
    "canada_10y": "BD.CDN.10YR.DQ.YLD",
    "canada_2y": "BD.CDN.2YR.DQ.YLD",
    "boc_rate": "B114039",
}

# Series the forecast model cannot run without
REQUIRED_MONTHLY = ["us_10y", "fed_funds", "canada_10y", "boc_rate"]

ACM_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "data_indices/ACMTermPremium.xls"
)


def fetch_acm(start: str) -> pd.DataFrame:
    """
    Download the NY Fed ACM term structure decomposition (daily).
    Returns columns acm_tp (ACMTP10) and acm_rny (ACMRNY10, risk-neutral 10Y yield).
    Requires the `xlrd` package to read the .xls file.
    """
    raw = pd.read_excel(ACM_URL, sheet_name=0)
    raw["DATE"] = pd.to_datetime(raw["DATE"], format="mixed", dayfirst=True)
    raw = raw.set_index("DATE").sort_index()
    out = raw[["ACMTP10", "ACMRNY10"]].rename(
        columns={"ACMTP10": "acm_tp", "ACMRNY10": "acm_rny"}
    )
    return out.loc[start:]


def _try(fetch, name: str, required: bool):
    try:
        return fetch()
    except Exception as exc:  # network / series errors
        if required:
            raise
        logger.warning("  Skipping optional series %s: %s", name, exc)
        return None


def build_monthly_dataset(config: dict, start: str = "1990-01-01") -> pd.DataFrame:
    """
    Fetch all series needed by the forecast model and return a month-end indexed DataFrame.
    Market series are monthly averages; macro series take the last value in each month.
    Optional series that fail to download are skipped with a warning.
    """
    api_key = config["fred_api_key"]
    end = date.today().isoformat()
    logger.info("Fetching monthly forecast dataset from %s to %s", start, end)

    market: dict[str, pd.Series] = {}
    macro: dict[str, pd.Series] = {}

    for name, sid in MONTHLY_FRED_MARKET.items():
        s = _try(lambda: fetch_fred_series(sid, start, end, api_key), name, name in REQUIRED_MONTHLY)
        if s is not None:
            market[name] = s
    for name, sid in MONTHLY_BOC_MARKET.items():
        s = _try(lambda: fetch_boc_series(sid, start, end), name, name in REQUIRED_MONTHLY)
        if s is not None:
            market[name] = s
    for name, sid in MONTHLY_FRED_MACRO.items():
        s = _try(lambda: fetch_fred_series(sid, start, end, api_key), name, False)
        if s is not None:
            macro[name] = s

    acm = _try(lambda: fetch_acm(start), "acm", False)
    if acm is not None:
        for col in acm.columns:
            market[col] = acm[col]

    daily = pd.DataFrame(market)
    monthly = daily.resample("ME").mean()
    # Realized volatility of daily 10Y changes within the month (bps)
    monthly["us_10y_rvol"] = daily["us_10y"].diff().resample("ME").std() * 100

    if macro:
        monthly = monthly.join(pd.DataFrame(macro).resample("ME").last(), how="left")

    return finalize_monthly_dataset(monthly)


def finalize_monthly_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and trim. Shared by the live fetcher, CSV cache and synthetic data."""
    df = df.copy().sort_index()
    df = df.dropna(subset=["us_10y", "fed_funds"], how="any")
    # Slow-moving macro series are published with a lag / at lower frequency
    for col in ("nrou", "deficit_gdp", "core_pce", "unrate"):
        if col in df:
            df[col] = df[col].ffill()
    if "core_pce" in df and "core_pce_yoy" not in df:
        df["core_pce_yoy"] = df["core_pce"].pct_change(12, fill_method=None) * 100
    # Best available term premium: ACM if present, else Kim-Wright
    if "us_tp" not in df:
        if "acm_tp" in df and df["acm_tp"].notna().any():
            df["us_tp"] = df["acm_tp"].fillna(df.get("kw_tp"))
        elif "kw_tp" in df:
            df["us_tp"] = df["kw_tp"]
    if "us_tp" in df:
        df["us_rny"] = df["us_10y"] - df["us_tp"]  # risk-neutral (expectations) component
    if "canada_10y" in df:
        df["spread_can_us"] = df["canada_10y"] - df["us_10y"]
        df["policy_diff"] = df["boc_rate"] - df["fed_funds"]
    return df


def save_monthly_dataset(df: pd.DataFrame, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date")


def load_monthly_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return finalize_monthly_dataset(df)
