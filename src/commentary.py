"""Use Claude API to draft each report section from quantitative metrics."""

from __future__ import annotations

import json
import logging

import anthropic

logger = logging.getLogger(__name__)

MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a senior quantitative economist drafting a long-term bond yield forecast
report. Write in a professional, concise style suitable for a central bank or financial institution
publication. Use bullet points as instructed. Be analytically precise — cite the specific numbers
from the data provided. Do not pad with generic statements."""


def _call_claude(client: anthropic.Anthropic, prompt: str) -> str:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _fmt_metrics(metrics: dict) -> str:
    safe = {k: v for k, v in metrics.items() if not k.startswith("_")}
    return json.dumps(safe, indent=2)


def draft_background_research(
    client: anthropic.Anthropic, metrics: dict, config: dict, country: str
) -> str:
    quarter = config["quarter"]

    if country == "us":
        prompt = f"""
Draft the **Background Research** section for a {quarter} US 10-Year Treasury yield forecast report.

KEY DATA:
{_fmt_metrics(metrics)}

Write 6–8 bullet points covering:
- How US 10Y yield moved during the last quarter ({metrics['last_quarter_start']} to {metrics['last_quarter_end']}),
  including start ({metrics.get('us_10y_q_start', 'N/A')}%), end ({metrics.get('us_10y_q_end', 'N/A')}%),
  high ({metrics.get('us_10y_q_high', 'N/A')}%), change ({metrics.get('us_10y_q_change_bps', 'N/A')} bps)
- Key macro drivers of US yield movements last quarter (Fed policy, inflation prints, growth data,
  fiscal/supply dynamics, geopolitical factors)
- Where US 10Y sits relative to its 5Y ({metrics['us_10y_mean_5y']:.2f}%) and 10Y ({metrics['us_10y_mean_10y']:.2f}%) historical averages
- Current ACM term premium context ({metrics['acm_tp_current']:.2f}% at {metrics['acm_tp_pct_rank_20y']:.0f}th percentile of 20Y history)

Begin each bullet with "•". No headers within this section. Mark the section:
[DRAFT — please review]
"""
    else:
        prompt = f"""
Draft the **Background Research** section for a {quarter} Canada 10-Year Government Bond yield forecast report.

KEY DATA:
{_fmt_metrics(metrics)}

Write 6–8 bullet points covering:
- How Canada 10Y yield moved during the last quarter ({metrics['last_quarter_start']} to {metrics['last_quarter_end']}),
  including start ({metrics.get('canada_10y_q_start', 'N/A')}%), end ({metrics.get('canada_10y_q_end', 'N/A')}%),
  high ({metrics.get('canada_10y_q_high', 'N/A')}%), change ({metrics.get('canada_10y_q_change_bps', 'N/A')} bps)
- Key macro drivers of Canadian yield movements last quarter (Bank of Canada policy decisions,
  Canadian inflation, housing market, trade dynamics, CAD/USD)
- Where Canada 10Y sits relative to its 5Y ({metrics['canada_10y_mean_5y']:.2f}%) and 10Y ({metrics['canada_10y_mean_10y']:.2f}%) historical averages
- Canada–US 10Y spread dynamics: current {metrics['spread_can_us_current']:+.2f}pp vs 5Y avg {metrics['spread_can_us_mean_5y']:+.2f}pp,
  and what this implies for relative BoC vs Fed policy paths

Begin each bullet with "•". No headers within this section. Mark the section:
[DRAFT — please review]
"""
    return _call_claude(client, prompt)


def draft_sources(metrics: dict, config: dict) -> str:
    """Auto-generate the sources section (no LLM needed)."""
    quarter = config["quarter"]
    lines = [
        "[DRAFT — please review]",
        "",
        "**Data Sources:**",
        "• Federal Reserve Economic Data (FRED) — fred.stlouisfed.org",
        "  - DGS10: US 10-Year Treasury Constant Maturity Rate (daily)",
        "  - DFF: Federal Funds Effective Rate (daily)",
        "  - ACMTP10: Adrian, Crump & Moench (ACM) 10-Year Term Premium (daily)",
        "  - T10YIE: 10-Year Breakeven Inflation Rate (daily)",
        "• Bank of Canada VALET API — bankofcanada.ca/valet",
        "  - V39056: Government of Canada 10-Year Benchmark Bond Yield (daily)",
        "",
        "**Analytical References:**",
        "• Adrian, T., Crump, R. K., & Moench, E. (2013). 'Pricing the Term Structure with "
        "Linear Regressions.' Journal of Financial Economics.",
        "• Federal Open Market Committee (FOMC) — Summary of Economic Projections (SEP), "
        "long-run Federal Funds rate estimates.",
        "• Bank of Canada — Monetary Policy Report (MPR), most recent edition.",
        f"• Report generated: {config['run_date']} for {quarter}.",
    ]
    return "\n".join(lines)


def draft_forecasting_rationale(
    client: anthropic.Anthropic, metrics: dict, config: dict, country: str
) -> str:
    quarter = config["quarter"]
    us_target = config["us_10y"]["target"]
    ca_target = config["canada_10y"]["target"]
    methodology = metrics["methodology"]
    macro_themes = "\n".join(f"  - {t}" for t in config.get("macro_themes", []))

    if country == "us":
        m = methodology
        prompt = f"""
Draft the **Forecasting Rationale** section for a {quarter} US 10-Year Treasury yield forecast.

TARGET: US 10Y = {us_target}%

The forecaster triangulates this target using THREE independent frameworks:

FRAMEWORK 1 — Spread-based (r* + historical 10Y–FFR spread):
The 10Y–FFR spread is used directly; term premium is already embedded in the historical spread average.
- Long-run Fed Funds r* = {m['long_run_fed_funds']}% (FOMC SEP long-run dot)
- 5Y avg 10Y–FFR spread: {metrics['spread_10y_ffr_mean_5y']:+.2f}pp → implied 10Y: {m['implied_10y_using_5y_avg_spread']:.2f}%
- 10Y avg 10Y–FFR spread: {metrics['spread_10y_ffr_mean_10y']:+.2f}pp → implied 10Y: {m['implied_10y_using_10y_avg_spread']:.2f}%
- 20Y avg 10Y–FFR spread: {metrics['spread_10y_ffr_mean_20y']:+.2f}pp → implied 10Y: {m['implied_10y_using_20y_avg_spread']:.2f}%

FRAMEWORK 2 — ACM decomposition (expected short rate + explicit term premium):
10Y yield = expected avg short rate converging to long-run FFR + term premium.
- Long-run FFR ({m['long_run_fed_funds']}%) + 10Y avg TP ({metrics['acm_tp_mean_10y']:.2f}%) → implied 10Y: {m['implied_10y_acm_10y_avg_tp']:.2f}%
- Long-run FFR ({m['long_run_fed_funds']}%) + 20Y avg TP ({metrics['acm_tp_mean_20y']:.2f}%) → implied 10Y: {m['implied_10y_acm_20y_avg_tp']:.2f}%
- Current TP is {metrics['acm_tp_current']:.2f}% (at {metrics['acm_tp_pct_rank_20y']:.0f}th percentile of 20Y history) → used as a risk signal, not additive
- NOTE: Framework 1 and 2 should NOT be combined — TP is already in the spread average in Framework 1

FRAMEWORK 3 — GDP + Inflation (Fisher / neutral rate approach):
Nominal neutral rate ≈ long-run real GDP growth + long-run inflation (classical Fisher equation).
- FOMC SEP long-run real GDP growth: {m['long_run_real_gdp_growth']:.2f}%
- Fed PCE inflation target: {m['long_run_pce_inflation']:.2f}%
- Implied 10Y: {m['long_run_real_gdp_growth']:.2f}% + {m['long_run_pce_inflation']:.2f}% = {m['implied_10y_gdp_inflation']:.2f}%
- This uses real GDP growth as a proxy for real r*; the gap vs FOMC's lower r* estimate reflects structural headwinds (demographics, savings glut)

CURRENT LEVELS:
- US 10Y: {metrics['us_10y_current']:.2f}% (10Y avg: {metrics['us_10y_mean_10y']:.2f}%)
- 10Y breakeven inflation: {metrics['us_10y_bei_current']:.2f}%

MACRO THEMES DRIVING THE VIEW:
{macro_themes}

Write 8–10 bullet points covering:
- Framework 1 derivation: show the arithmetic for the spread-based implied range
- Framework 2 derivation: show how ACM decomposition gives a consistent cross-check (and clarify it should not be stacked on top of Framework 1)
- Framework 3 derivation: GDP + inflation as the classical anchor, and why it converges with Frameworks 1 and 2
- Why {us_target}% sits within / at the intersection of the three implied ranges
- Key structural factors (neutral rate, fiscal, inflation regime) supporting the long-term anchor

Begin each bullet with "•". No sub-headers. Mark the section:
[DRAFT — please review]
"""
    else:
        spread_assumption = ca_target - us_target
        m = methodology
        prompt = f"""
Draft the **Forecasting Rationale** section for a {quarter} Canada 10-Year Government Bond yield forecast.

TARGET: Canada 10Y = {ca_target}%

The forecaster cross-checks the Canada target using TWO independent spread anchors that should converge:

ANCHOR 1 — BoC Rate + Canada 10Y–BoC Spread (domestic anchor):
- Long-run BoC overnight rate assumption (r*): {m['long_run_boc_rate']:.2f}%
- Historical Canada 10Y–BoC spread averages:
    5Y avg: {metrics['spread_canada_10y_boc_mean_5y']:+.2f}pp → implied Canada 10Y: {m['implied_canada_10y_using_5y_boc_spread']:.2f}%
   10Y avg: {metrics['spread_canada_10y_boc_mean_10y']:+.2f}pp → implied Canada 10Y: {m['implied_canada_10y_using_10y_boc_spread']:.2f}%
   20Y avg: {metrics['spread_canada_10y_boc_mean_20y']:+.2f}pp → implied Canada 10Y: {m['implied_canada_10y_using_20y_boc_spread']:.2f}%
- Current Canada 10Y–BoC spread: {metrics['spread_canada_10y_boc_current']:+.2f}pp

ANCHOR 2 — US 10Y Target + Canada–US Spread (cross-border anchor):
- US 10Y target: {us_target}%
- Historical Canada–US 10Y spread averages:
    5Y avg: {metrics['spread_can_us_mean_5y']:+.2f}pp → implied Canada 10Y: {m['implied_canada_10y_using_5y_canus_spread']:.2f}%
   10Y avg: {metrics['spread_can_us_mean_10y']:+.2f}pp → implied Canada 10Y: {m['implied_canada_10y_using_10y_canus_spread']:.2f}%
- Current Canada–US spread: {metrics['spread_can_us_current']:+.2f}pp

CURRENT LEVELS:
- Canada 10Y: {metrics['canada_10y_current']:.2f}% (10Y avg: {metrics['canada_10y_mean_10y']:.2f}%)
- US 10Y: {metrics['us_10y_current']:.2f}%
- Canada–US spread: {metrics['spread_can_us_current']:+.2f}pp (5Y avg: {metrics['spread_can_us_mean_5y']:+.2f}pp)

MACRO THEMES DRIVING THE VIEW:
{macro_themes}

Write 8–10 bullet points covering:
- Anchor 1 derivation: how long-run BoC r* + historical Canada 10Y–BoC spread range implies Canada 10Y target (show arithmetic)
- Anchor 2 derivation: how US 10Y anchor + Canada–US spread assumption implies the same target (show arithmetic)
- Why the two anchors converge on {ca_target}% and what that consistency implies
- Why the Canada–US spread assumption of {spread_assumption:+.1f}pp is reasonable given BoC vs Fed policy divergence
- Key structural factors affecting both anchors (BoC neutral rate vs Fed neutral rate, fiscal, housing)

Begin each bullet with "•". No sub-headers. Mark the section:
[DRAFT — please review]
"""
    return _call_claude(client, prompt)


def draft_medium_term_risks(
    client: anthropic.Anthropic, metrics: dict, config: dict, country: str
) -> str:
    quarter = config["quarter"]
    risks = config.get("medium_term_risks", {})
    upside = "\n".join(f"  - {r}" for r in risks.get("upside", []))
    downside = "\n".join(f"  - {r}" for r in risks.get("downside", []))
    country_label = "US 10-Year Treasury" if country == "us" else "Canada 10-Year Government Bond"
    target = config["us_10y"]["target"] if country == "us" else config["canada_10y"]["target"]
    current = metrics["us_10y_current"] if country == "us" else metrics["canada_10y_current"]

    prompt = f"""
Draft the **Medium-Term Risks (3–5 Year)** section for a {quarter} {country_label} yield forecast.

CURRENT CONTEXT:
- {country_label}: {current:.2f}% | Forecast target: {target}%
- ACM term premium: {metrics['acm_tp_current']:.2f}% ({metrics['acm_tp_pct_rank_20y']:.0f}th pct)
- 10Y breakeven inflation: {metrics['us_10y_bei_current']:.2f}%

USER-IDENTIFIED UPSIDE RISKS (yields higher than forecast):
{upside}

USER-IDENTIFIED DOWNSIDE RISKS (yields lower than forecast):
{downside}

Write 6–8 bullet points framed specifically for the {country_label}. Structure as:
- First 3–4 bullets: upside risks with specific mechanism and rough yield impact (e.g., "+30–50 bps")
- Next 3–4 bullets: downside risks with specific mechanism and yield impact

Begin each bullet with "↑ [Upside]" or "↓ [Downside]" as appropriate. Mark the section:
[DRAFT — please review]
"""
    return _call_claude(client, prompt)


def draft_long_term_risks(
    client: anthropic.Anthropic, metrics: dict, config: dict, country: str
) -> str:
    quarter = config["quarter"]
    risks = config.get("long_term_risks", {})
    upside = "\n".join(f"  - {r}" for r in risks.get("upside", []))
    downside = "\n".join(f"  - {r}" for r in risks.get("downside", []))
    country_label = "US 10-Year Treasury" if country == "us" else "Canada 10-Year Government Bond"
    target = config["us_10y"]["target"] if country == "us" else config["canada_10y"]["target"]
    mean_20y = metrics["us_10y_mean_20y"] if country == "us" else metrics["canada_10y_mean_20y"]

    prompt = f"""
Draft the **Long-Term Risks (5–10 Year)** section for a {quarter} {country_label} yield forecast.

These are structural, secular forces that could shift the terminal level of bond yields over a
decade-long horizon — beyond the medium-term forecast.

CURRENT CONTEXT:
- {country_label} target: {target}%
- 20Y historical avg: {mean_20y:.2f}%
- Current ACM term premium: {metrics['acm_tp_current']:.2f}% vs 20Y avg: {metrics['acm_tp_mean_20y']:.2f}%

USER-IDENTIFIED UPSIDE RISKS (yields structurally higher):
{upside}

USER-IDENTIFIED DOWNSIDE RISKS (yields structurally lower):
{downside}

Write 6–8 bullet points framed specifically for the {country_label}. Structure as:
- First 3–4 bullets: structural upside risks with the transmission channel and magnitude
- Next 3–4 bullets: structural downside risks

Begin each bullet with "↑ [Upside]" or "↓ [Downside]" as appropriate. Mark the section:
[DRAFT — please review]
"""
    return _call_claude(client, prompt)


def generate_all_commentary(metrics: dict, config: dict) -> dict[str, dict[str, str]]:
    """Draft all report sections for US and Canada. Returns {"us": {...}, "canada": {...}}."""
    api_key = config.get("anthropic_api_key") or None
    client = anthropic.Anthropic(api_key=api_key)

    logger.info("Drafting commentary with Claude (%s)...", MODEL)

    result: dict[str, dict[str, str]] = {}
    for country in ("us", "canada"):
        logger.info("  Country: %s", country.upper())
        sections: dict[str, str] = {}

        logger.info("    Section 1: Background research")
        sections["background"] = draft_background_research(client, metrics, config, country)

        logger.info("    Section 2: Sources")
        sections["sources"] = draft_sources(metrics, config)

        logger.info("    Section 3: Forecasting rationale")
        sections["rationale"] = draft_forecasting_rationale(client, metrics, config, country)

        logger.info("    Section 4: Medium-term risks")
        sections["medium_term_risks"] = draft_medium_term_risks(client, metrics, config, country)

        logger.info("    Section 5: Long-term risks")
        sections["long_term_risks"] = draft_long_term_risks(client, metrics, config, country)

        result[country] = sections

    return result
