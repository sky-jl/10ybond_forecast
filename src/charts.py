"""Generate publication-quality charts for the bond yield forecast report."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# House style
BLUE = "#1B4F8A"
ORANGE = "#E07B39"
GREEN = "#2E7D32"
GREY = "#9E9E9E"
LIGHT_GREY = "#F5F5F5"
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.6,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(path)


def _tail_years(series: pd.Series, years: int) -> pd.Series:
    cutoff = series.index[-1] - pd.DateOffset(years=years)
    return series[series.index >= cutoff].dropna()


def _year_fmt(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")


def chart_us_10y(df: pd.DataFrame, metrics: dict, output_dir: Path, config: dict) -> str:
    """US 10Y yield — 10-year history with last-quarter shaded and target annotated."""
    window = _tail_years(df["us_10y"], 10)
    target = config["us_10y"]["target"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window.index, window.values, color=BLUE, linewidth=1.4, label="US 10Y Yield")

    # Shade last quarter
    qs = pd.Timestamp(metrics["last_quarter_start"])
    qe = pd.Timestamp(metrics["last_quarter_end"])
    ax.axvspan(qs, qe, alpha=0.12, color=ORANGE, label="Last quarter")

    # Target line
    ax.axhline(target, color=GREEN, linewidth=1.2, linestyle="--",
               label=f"LT Target: {target:.1f}%")

    # 10Y average
    avg = metrics["us_10y_mean_10y"]
    ax.axhline(avg, color=GREY, linewidth=0.9, linestyle=":",
               label=f"10Y avg: {avg:.2f}%")

    ax.set_title("US 10-Year Treasury Yield", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Yield (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_us_10y.png")


def chart_canada_10y(df: pd.DataFrame, metrics: dict, output_dir: Path, config: dict) -> str:
    """Canada 10Y GoC yield — 10-year history with target annotated."""
    window = _tail_years(df["canada_10y"], 10)
    target = config["canada_10y"]["target"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(window.index, window.values, color="#C62828", linewidth=1.4,
            label="Canada 10Y GoC Yield")

    qs = pd.Timestamp(metrics["last_quarter_start"])
    qe = pd.Timestamp(metrics["last_quarter_end"])
    ax.axvspan(qs, qe, alpha=0.12, color=ORANGE, label="Last quarter")

    ax.axhline(target, color=GREEN, linewidth=1.2, linestyle="--",
               label=f"LT Target: {target:.1f}%")

    avg = metrics["canada_10y_mean_10y"]
    ax.axhline(avg, color=GREY, linewidth=0.9, linestyle=":",
               label=f"10Y avg: {avg:.2f}%")

    ax.set_title("Canada 10-Year Government Bond Yield", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Yield (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_canada_10y.png")


def chart_spread_10y_ffr(df: pd.DataFrame, metrics: dict, output_dir: Path) -> str:
    """US 10Y–Fed Funds spread vs rolling 10Y average."""
    spread = _tail_years(df["spread_10y_ffr"], 15)
    rolling_avg = spread.rolling(window=252 * 10, min_periods=252).mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spread.index, spread.values, color=BLUE, linewidth=1.2,
            label="10Y–Fed Funds Spread")
    ax.plot(rolling_avg.index, rolling_avg.values, color=ORANGE, linewidth=1.2,
            linestyle="--", label="Rolling 10Y Average")
    ax.axhline(0, color="black", linewidth=0.7)

    current = metrics["spread_10y_ffr_current"]
    ax.annotate(
        f"Current: {current:+.2f}%",
        xy=(spread.index[-1], current),
        xytext=(-80, 15),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_title("US 10Y Yield minus Fed Funds Rate (Spread)", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Spread (pp)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_spread_10y_ffr.png")


def chart_acm_term_premium(df: pd.DataFrame, metrics: dict, output_dir: Path) -> str:
    """ACM 10Y term premium with mean and ±1σ bands."""
    acm = _tail_years(df["acm_tp"], 20)
    mu = acm.mean()
    sigma = acm.std()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(acm.index, mu - sigma, mu + sigma, alpha=0.12, color=BLUE,
                    label="±1σ band")
    ax.plot(acm.index, acm.values, color=BLUE, linewidth=1.2, label="ACM Term Premium")
    ax.axhline(mu, color=ORANGE, linewidth=1.0, linestyle="--",
               label=f"Historical mean: {mu:.2f}%")
    ax.axhline(0, color="black", linewidth=0.7)

    current = metrics["acm_tp_current"]
    pct = metrics["acm_tp_pct_rank_20y"]
    ax.annotate(
        f"Current: {current:.2f}%\n({pct:.0f}th pct)",
        xy=(acm.index[-1], current),
        xytext=(-100, -30),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_title("ACM 10-Year Term Premium", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Term Premium (%)")
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_acm_term_premium.png")


def chart_spread_canada_boc(df: pd.DataFrame, metrics: dict, output_dir: Path) -> str:
    """Canada 10Y–BoC overnight rate spread vs rolling 10Y average."""
    spread = _tail_years(df["spread_canada_10y_boc"], 15)
    rolling_avg = spread.rolling(window=252 * 10, min_periods=252).mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(spread.index, spread.values, color="#C62828", linewidth=1.2,
            label="Canada 10Y–BoC Rate Spread")
    ax.plot(rolling_avg.index, rolling_avg.values, color=ORANGE, linewidth=1.2,
            linestyle="--", label="Rolling 10Y Average")
    ax.axhline(0, color="black", linewidth=0.7)

    current = metrics["spread_canada_10y_boc_current"]
    ax.annotate(
        f"Current: {current:+.2f}%",
        xy=(spread.index[-1], current),
        xytext=(-80, 15),
        textcoords="offset points",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_title("Canada 10Y Yield minus BoC Overnight Rate (Spread)", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Spread (pp)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_spread_canada_boc.png")


def chart_canada_us_spread(df: pd.DataFrame, metrics: dict, output_dir: Path) -> str:
    """Canada–US 10Y spread — 5-year history."""
    spread = _tail_years(df["spread_can_us"], 5)
    avg = metrics["spread_can_us_mean_5y"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(spread.index, spread.values, 0,
                    where=spread.values >= 0, alpha=0.15, color=GREEN, interpolate=True)
    ax.fill_between(spread.index, spread.values, 0,
                    where=spread.values < 0, alpha=0.15, color="#C62828", interpolate=True)
    ax.plot(spread.index, spread.values, color=BLUE, linewidth=1.2,
            label="Canada–US 10Y Spread")
    ax.axhline(avg, color=ORANGE, linewidth=1.0, linestyle="--",
               label=f"5Y avg: {avg:.2f}pp")
    ax.axhline(0, color="black", linewidth=0.7)

    ax.set_title("Canada–US 10-Year Yield Spread", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Spread (pp)")
    _year_fmt(ax)
    ax.legend(fontsize=8, framealpha=0.8)

    return _save(fig, output_dir / "chart_canada_us_spread.png")


def generate_all_charts(df: pd.DataFrame, metrics: dict, output_dir: Path, config: dict) -> dict[str, str]:
    """Generate all charts and return a dict of {chart_name: file_path}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "us_10y": chart_us_10y(df, metrics, output_dir, config),
        "canada_10y": chart_canada_10y(df, metrics, output_dir, config),
        "spread_10y_ffr": chart_spread_10y_ffr(df, metrics, output_dir),
        "acm_term_premium": chart_acm_term_premium(df, metrics, output_dir),
        "canada_us_spread": chart_canada_us_spread(df, metrics, output_dir),
        "spread_canada_boc": chart_spread_canada_boc(df, metrics, output_dir),
    }
