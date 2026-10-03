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
    """Kim-Wright 10Y term premium with mean and ±1σ bands."""
    acm = _tail_years(df["acm_tp"], 20)
    mu = acm.mean()
    sigma = acm.std()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(acm.index, mu - sigma, mu + sigma, alpha=0.12, color=BLUE,
                    label="±1σ band")
    ax.plot(acm.index, acm.values, color=BLUE, linewidth=1.2, label="Kim-Wright Term Premium")
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

    ax.set_title("10-Year Term Premium (Kim-Wright)", fontsize=11, fontweight="bold", loc="left")
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


# ============================================================
# 2-year forecast charts (src/forecasting)
# ============================================================
SCENARIO_COLORS = [BLUE, "#C62828", GREEN, "#6A1B9A", ORANGE, GREY]


def _month_fmt(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def chart_forecast_fan(result, country: str, output_dir: Path, history_years: int = 4) -> str:
    """History + probability-weighted central path + 10–90 / 25–75 percentile bands + scenarios."""
    col = "us_10y" if country == "us" else "canada_10y"
    label = "US 10Y Treasury" if country == "us" else "Canada 10Y GoC"
    hist = _tail_years(result.history[col], history_years)
    fan = result.fan
    color = BLUE if country == "us" else "#B71C1C"

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hist.index, hist.values, color="black", linewidth=1.2, label="History (monthly avg)")
    ax.fill_between(fan.index, fan[f"{col}_p10"], fan[f"{col}_p90"], color=color, alpha=0.12,
                    label="10–90th pct")
    ax.fill_between(fan.index, fan[f"{col}_p25"], fan[f"{col}_p75"], color=color, alpha=0.22,
                    label="25–75th pct")
    for (name, sc), c in zip(result.scenarios.items(), SCENARIO_COLORS[1:]):
        ax.plot(sc.index, sc[col], color=c, linewidth=0.9, linestyle="--",
                label=f"{name} ({result.probabilities[name]:.0%})")
    ax.plot(result.central.index, result.central[col], color=color, linewidth=2.0,
            label="Probability-weighted")
    end = result.central[col].iloc[-1]
    ax.annotate(f"{end:.2f}%", xy=(result.central.index[-1], end), xytext=(6, 0),
                textcoords="offset points", fontsize=8, va="center", color=color)
    ax.axvline(result.as_of, color=GREY, linewidth=0.8, linestyle=":")
    ax.set_title(f"{label} — 2-Year Forecast", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("Yield (%)")
    _year_fmt(ax)
    ax.legend(fontsize=7, framealpha=0.85, loc="best", ncol=2)
    return _save(fig, output_dir / f"chart_forecast_fan_{country}.png")


def chart_forecast_decomposition(result, output_dir: Path) -> str:
    """Stacked components of the central US 10Y path: expectations + basis + TP + overlay."""
    c = result.central
    fig, ax = plt.subplots(figsize=(9, 4))
    exp_basis = c["us_expectations"] + c["us_basis"]
    ax.fill_between(c.index, 0, exp_basis, color=BLUE, alpha=0.35,
                    label="Expected avg short rate (+ basis)")
    if "us_fiscal" in c and c["us_fiscal"].abs().max() > 1e-9:
        top = exp_basis + c["us_tp"]
        ax.fill_between(c.index, top, top + c["us_fiscal"], color="#E34948", alpha=0.45,
                        label="Fiscal premium")
    ax.fill_between(c.index, exp_basis, exp_basis + c["us_tp"], color=ORANGE, alpha=0.45,
                    label="Term premium")
    if c["us_overlay"].abs().max() > 1e-9:
        ax.plot(c.index, c["us_10y"] - c["us_overlay"], color=GREY, linewidth=0.9,
                linestyle="--", label="Before judgmental overlay")
    ax.plot(c.index, c["us_10y"], color="black", linewidth=1.8, label="US 10Y forecast")
    ax.plot(c.index, c["fed_funds"], color=GREEN, linewidth=1.2, label="Fed funds path")
    lo = min(c["fed_funds"].min(), exp_basis.min())
    ax.set_ylim(max(0.0, lo - 0.75), c["us_10y"].max() + 0.5)
    ax.set_title("US 10Y Forecast Decomposition (probability-weighted)", fontsize=11,
                 fontweight="bold", loc="left")
    ax.set_ylabel("%")
    _month_fmt(ax)
    ax.legend(fontsize=7, framealpha=0.85, loc="lower left", ncol=2)
    return _save(fig, output_dir / "chart_forecast_decomposition.png")


def chart_policy_scenarios(result, country: str, output_dir: Path, history_years: int = 3) -> str:
    """Policy-rate scenario paths (Fed or BoC)."""
    col = "fed_funds" if country == "us" else "boc_rate"
    label = "Fed Funds Rate" if country == "us" else "BoC Overnight Rate"
    hist = _tail_years(result.history[col], history_years)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(hist.index, hist.values, color="black", linewidth=1.2, label="History")
    for (name, sc), c in zip(result.scenarios.items(), SCENARIO_COLORS[1:]):
        ax.plot(sc.index, sc[col], color=c, linewidth=1.2,
                label=f"{name} ({result.probabilities[name]:.0%})")
    ax.plot(result.central.index, result.central[col], color=BLUE, linewidth=2.0,
            label="Probability-weighted")
    ax.axvline(result.as_of, color=GREY, linewidth=0.8, linestyle=":")
    ax.set_title(f"{label} — Scenario Paths", fontsize=11, fontweight="bold", loc="left")
    ax.set_ylabel("%")
    _year_fmt(ax)
    ax.legend(fontsize=7, framealpha=0.85, loc="best")
    return _save(fig, output_dir / f"chart_policy_scenarios_{country}.png")


def generate_forecast_charts(result, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "forecast_fan_us": chart_forecast_fan(result, "us", output_dir),
        "forecast_fan_canada": chart_forecast_fan(result, "canada", output_dir),
        "forecast_decomposition": chart_forecast_decomposition(result, output_dir),
        "policy_scenarios_us": chart_policy_scenarios(result, "us", output_dir),
        "policy_scenarios_canada": chart_policy_scenarios(result, "canada", output_dir),
    }
