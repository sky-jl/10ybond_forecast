"""
Concise (≤ 2 page) PDF forecast brief for the US and Canada 10Y yields.

Follows the structure of the quarterly Word reports (background → rationale → risks →
conclusion) in a compact form, built on the 2-year model output (forecasting.ForecastResult).
Commentary is drafted by Claude (structured JSON output); without an API key a factual
template is used instead so the PDF can always be generated.
"""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"

NAVY = colors.HexColor("#1B4F8A")
INK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d4")
FILL = colors.HexColor("#f3f6fa")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence, the core call with numbers"},
        "summary": {"type": "string", "description": "3-4 sentences: overall view for both markets"},
        "market_narrative": {"type": "string", "description":
                             "2-3 sentences: what the market traded last quarter (Fed path, inflation/energy, "
                             "term premium/fiscal) based on the attribution numbers"},
        "macro_backdrop": {"type": "array", "items": {"type": "string"}, "description":
                           "3-4 bullets on macro drivers of the outlook: fiscal deficits/debt and issuance "
                           "(use the fiscal fair value), Fed policy and independence, energy, geopolitics"},
        "us_analysis": {"type": "array", "items": {"type": "string"},
                        "description": "3-4 bullets on US 10Y drivers: policy path, term premium, anchors"},
        "canada_analysis": {"type": "array", "items": {"type": "string"},
                            "description": "2-3 bullets on Canada 10Y and the Canada-US spread"},
        "upside_risks": {"type": "array", "items": {"type": "string"},
                         "description": "2-3 risks that would push yields higher, with rough bps impact"},
        "downside_risks": {"type": "array", "items": {"type": "string"},
                           "description": "2-3 risks that would push yields lower, with rough bps impact"},
        "conclusion": {"type": "string", "description": "2-3 sentences: forecast call and what would change it"},
    },
    "required": ["headline", "summary", "market_narrative", "macro_backdrop", "us_analysis", "canada_analysis", "upside_risks",
                 "downside_risks", "conclusion"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a senior fixed-income strategist writing a concise two-page outlook for
the US 10-year Treasury and Government of Canada 10-year yields. Every claim must be grounded in
the model output provided; cite specific numbers (levels in %, changes in bps, probabilities).
The forecast comes from a structural model (10Y = expected average short rate + term premium;
Canada 10Y = US 10Y + Canada–US spread) with forecaster-chosen policy scenarios, so describe it as
a scenario-weighted view, not a certainty. Be candid that 10Y yields rarely beat a random walk in
backtests. Give the macro view equal weight to the technical one: explain the move through fiscal
deficits, debt and Treasury supply (the term premium vs its fiscal fair value), Fed policy and
independence (5y5y inflation expectations), energy prices and geopolitics, using the attribution
of last quarter's move. You have no live news feed: rely on the numbers and the forecaster's macro
themes, and do not invent specific events or dates. Write in plain professional English, no
headers or markdown inside strings. The brief must fit two pages: keep the whole commentary under
about 450 words — summary and narrative under 60 words each, each bullet under 30 words."""


# --------------------------------------------------------------------------- inputs
def build_context(result, config: dict, backtest_summary: pd.DataFrame | None = None) -> dict:
    """Compact, JSON-serialisable facts for the commentary (and the template fallback)."""
    from forecasting.attribution import attribute

    c, f, q = result.central, result.fan, result.quarterly
    end = result.index[-1]
    ctx = {
        "quarter": config.get("quarter"),
        "as_of_month": f"{result.as_of:%b %Y}",
        "horizon_end": f"{end:%b %Y}",
        "current": {k: round(float(c[k].iloc[0]), 2) for k in
                    ("us_10y", "canada_10y", "fed_funds", "boc_rate", "us_tp", "ca_spread")},
        "forecast_end_weighted": {k: round(float(c[k].iloc[-1]), 2) for k in
                                  ("us_10y", "canada_10y", "fed_funds", "boc_rate", "us_tp", "ca_spread")},
        "range_10_90_at_end": {
            "us_10y": [round(float(f["us_10y_p10"].iloc[-1]), 2), round(float(f["us_10y_p90"].iloc[-1]), 2)],
            "canada_10y": [round(float(f["canada_10y_p10"].iloc[-1]), 2),
                           round(float(f["canada_10y_p90"].iloc[-1]), 2)],
        },
        "us_decomposition_change_bps": {
            k: round(float((c[k].iloc[-1] - c[k].iloc[0]) * 100))
            for k in ("us_expectations", "us_basis", "us_tp", "us_overlay", "us_10y")
        },
        "quarterly_weighted": {
            idx: {"us_10y": round(float(r["us_10y"]), 2), "canada_10y": round(float(r["canada_10y"]), 2),
                  "fed_funds": round(float(r["fed_funds"]), 2), "boc_rate": round(float(r["boc_rate"]), 2)}
            for idx, r in q.iterrows()
        },
        "scenarios": {
            n: {"probability": round(result.probabilities[n], 2),
                "fed_funds_end": round(float(s["fed_funds"].iloc[-1]), 2),
                "boc_rate_end": round(float(s["boc_rate"].iloc[-1]), 2),
                "us_10y_end": round(float(s["us_10y"].iloc[-1]), 2),
                "canada_10y_end": round(float(s["canada_10y"].iloc[-1]), 2)}
            for n, s in result.scenarios.items()
        },
        "anchors": {
            "neutral_fed_funds": result.params.get("neutral_fed_funds"),
            "neutral_boc_rate": result.params.get("neutral_boc_rate"),
            "term_premium_target": result.params.get("term_premium", {}).get("target"),
            "term_premium_method": result.params.get("term_premium", {}).get("method"),
            "long_run_real_gdp_growth": config.get("long_run_real_gdp_growth"),
            "long_run_pce_inflation": config.get("long_run_pce_inflation"),
            "beta_ca_us_36m": result.params.get("beta_ca_us_36m"),
        },
        "last_quarter_attribution": _round(attribute(result.history)),
        "term_premium_fiscal": _round({k: v for k, v in (result.params.get("term_premium", {})
                                                          .get("fiscal_fair_value") or {}).items()}),
        "term_premium_addons_bps": result.params.get("term_premium", {}).get("addons_bps", {}),
        "macro_themes": config.get("macro_themes", []),
        "forecaster_risks": config.get("medium_term_risks", {}),
    }
    if backtest_summary is not None and len(backtest_summary):
        s = backtest_summary[backtest_summary["model"] == "model"]
        ctx["backtest_rmse_vs_random_walk"] = {
            f"{r.target}_h{int(r.h)}": round(float(r.rmse_vs_rw), 3) for r in s.itertuples()}
    return ctx


def _round(x, nd: int = 2):
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


# --------------------------------------------------------------------------- commentary
def draft_with_claude(ctx: dict, api_key: str | None = None) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or None)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high",
                       "format": {"type": "json_schema", "schema": BRIEF_SCHEMA}},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content":
                   "Draft the outlook brief from this model output:\n\n"
                   + json.dumps(ctx, indent=1, default=float)}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined to draft the commentary")
    text = next(b.text for b in response.content if b.type == "text")
    out = json.loads(text)
    out["_source"] = f"Commentary drafted by Claude ({response.model})"
    return out


def template_commentary(ctx: dict) -> dict:
    """Factual fallback when no API key / network: numbers only, no judgment."""
    cur, end, rng = ctx["current"], ctx["forecast_end_weighted"], ctx["range_10_90_at_end"]
    d = ctx["us_decomposition_change_bps"]
    us_chg = (end["us_10y"] - cur["us_10y"]) * 100
    ca_chg = (end["canada_10y"] - cur["canada_10y"]) * 100
    sc = ctx["scenarios"]
    top = max(sc, key=lambda n: sc[n]["probability"])
    return {
        "headline": (f"Scenario-weighted model sees the US 10Y at {end['us_10y']:.2f}% and Canada 10Y at "
                     f"{end['canada_10y']:.2f}% by {ctx['horizon_end']} ({us_chg:+.0f} / {ca_chg:+.0f} bps)."),
        "summary": (f"From {cur['us_10y']:.2f}% (US) and {cur['canada_10y']:.2f}% (Canada) in {ctx['as_of_month']}, "
                    f"the probability-weighted path ends at {end['us_10y']:.2f}% and {end['canada_10y']:.2f}%. "
                    f"The 10–90th percentile range at the horizon is {rng['us_10y'][0]:.2f}–{rng['us_10y'][1]:.2f}% "
                    f"for the US and {rng['canada_10y'][0]:.2f}–{rng['canada_10y'][1]:.2f}% for Canada. "
                    f"The {top} scenario carries the largest weight ({sc[top]['probability']:.0%})."),
        "market_narrative": " ".join((ctx.get("last_quarter_attribution") or {}).get("reading", [])[:3]),
        "macro_backdrop": _template_macro(ctx),
        "us_analysis": [
            f"Fed funds path: {cur['fed_funds']:.2f}% now to {end['fed_funds']:.2f}% (weighted) by {ctx['horizon_end']}.",
            f"Expected short-rate component changes {d['us_expectations']:+d} bps; the market-vs-neutral gap "
            f"contributes {d['us_basis']:+d} bps as it converges toward the {ctx['anchors']['neutral_fed_funds']}% neutral rate.",
            f"Term premium moves from {cur['us_tp']:.2f}% to {end['us_tp']:.2f}% ({d['us_tp']:+d} bps).",
        ],
        "canada_analysis": [
            f"BoC path: {cur['boc_rate']:.2f}% to {end['boc_rate']:.2f}%; Canada–US spread from "
            f"{cur['ca_spread']:+.2f} to {end['ca_spread']:+.2f} pp.",
            f"36-month US→Canada yield beta: {ctx['anchors'].get('beta_ca_us_36m')}.",
        ],
        "upside_risks": list((ctx.get("forecaster_risks") or {}).get("upside", []))[:3],
        "downside_risks": list((ctx.get("forecaster_risks") or {}).get("downside", []))[:3],
        "conclusion": (f"Base call: US 10Y {end['us_10y']:.2f}%, Canada 10Y {end['canada_10y']:.2f}% by "
                       f"{ctx['horizon_end']}. The neutral-rate and term-premium assumptions are the key swing factors."),
        "_source": "Template commentary (no AI) — set ANTHROPIC_API_KEY for an analytical draft",
    }


def _template_macro(ctx: dict) -> list[str]:
    out = []
    fis = ctx.get("term_premium_fiscal") or {}
    if fis:
        d = fis.get("latest_drivers", {})
        out.append(f"Term premium {fis.get('tp_now', 0):.2f}% vs fiscal fair value {fis.get('fair_value_now', 0):.2f}% "
                   f"({fis.get('residual_now_bps', 0):+.0f} bps); debt held by public {d.get('debt_gdp', float('nan')):.0f}% "
                   f"of GDP, federal balance {d.get('deficit_gdp', float('nan')):+.1f}% of GDP.")
    adds = ctx.get("term_premium_addons_bps") or {}
    if adds:
        out.append("Forecaster risk-premium add-ons: " + ", ".join(f"{k.replace('_', ' ')} {v:+.0f} bps"
                                                                   for k, v in adds.items()) + ".")
    out += list(ctx.get("macro_themes", []))[:2]
    return out


def draft_commentary(ctx: dict, use_ai: bool = True, api_key: str | None = None) -> dict:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if use_ai and key:
        try:
            return draft_with_claude(ctx, key)
        except Exception as exc:  # network, auth, refusal → still produce a PDF
            logger.warning("Claude commentary failed (%s); using template", exc)
            out = template_commentary(ctx)
            out["_source"] = f"Template commentary (Claude call failed: {type(exc).__name__})"
            return out
    return template_commentary(ctx)


# --------------------------------------------------------------------------- charts
def _style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color="#e6e5e1", linewidth=0.6)
    ax.tick_params(labelsize=7)


def _fan_panel(ax, result, col: str, title: str, years: int = 3):
    hist = result.history[col].dropna()
    hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=years)]
    f, c = result.fan, result.central
    ax.plot(hist.index, hist.values, color="#7a7975", lw=1.0, label="History")
    ax.fill_between(f.index, f[f"{col}_p10"], f[f"{col}_p90"], color=SERIES[0], alpha=0.13, lw=0, label="10–90th pct")
    ax.fill_between(f.index, f[f"{col}_p25"], f[f"{col}_p75"], color=SERIES[0], alpha=0.25, lw=0, label="25–75th pct")
    for i, (name, s) in enumerate(result.scenarios.items()):
        ax.plot(s.index, s[col], color=SERIES[(i + 1) % len(SERIES)], lw=0.9, ls="--", label=name)
    ax.plot(c.index, c[col], color="#0b0b0b", lw=1.8, label="Weighted")
    ax.annotate(f"{c[col].iloc[-1]:.2f}%", (c.index[-1], c[col].iloc[-1]), xytext=(3, 0),
                textcoords="offset points", fontsize=7, va="center")
    ax.axvline(result.as_of, color="#9e9e9e", lw=0.7, ls=":")
    ax.set_title(title, fontsize=8.5, loc="left", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _style_ax(ax)


def chart_fans(result) -> io.BytesIO:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55), sharey=False)
    _fan_panel(axes[0], result, "us_10y", "US 10Y (%)")
    _fan_panel(axes[1], result, "canada_10y", "Canada 10Y (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=6.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _png(fig)


def chart_decomposition(result) -> io.BytesIO:
    c = result.central
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3))
    ax = axes[0]
    base = c["us_expectations"] + c["us_basis"]
    ax.fill_between(c.index, 0, base, color=SERIES[0], alpha=0.45, lw=0, label="Expected short rate + gap")
    ax.fill_between(c.index, base, base + c["us_tp"], color=SERIES[1], alpha=0.55, lw=0, label="Term premium")
    ax.plot(c.index, c["us_10y"], color="#0b0b0b", lw=1.6, label="US 10Y")
    ax.plot(c.index, c["fed_funds"], color=SERIES[2], lw=1.2, label="Fed funds")
    lo = min(c["fed_funds"].min(), base.min())
    ax.set_ylim(max(0, lo - 0.6), c["us_10y"].max() + 0.4)
    ax.set_title("US 10Y decomposition (%)", fontsize=8.5, loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.legend(fontsize=6, frameon=False, loc="lower left", ncol=2)
    _style_ax(ax)

    ax = axes[1]
    for i, (name, s) in enumerate(result.scenarios.items()):
        col = SERIES[(i + 1) % len(SERIES)]
        ax.plot(s.index, s["fed_funds"], color=col, lw=1.3, label=f"{name} – Fed")
        ax.plot(s.index, s["boc_rate"], color=col, lw=1.0, ls=":", label=f"{name} – BoC")
    ax.set_title("Policy-rate scenarios (%): Fed solid, BoC dotted", fontsize=8.5, loc="left", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::2], [l.replace(" – Fed", "") for l in labels[::2]], fontsize=6, frameon=False, loc="lower left", ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    return _png(fig)


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------- PDF
def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16,
                                leading=19, textColor=NAVY, alignment=TA_LEFT, spaceAfter=1),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontSize=8, textColor=MUTED, leading=10),
        "headline": ParagraphStyle("h", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=10.5,
                                   leading=13.5, textColor=INK, spaceBefore=6, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Normal"], fontName="Helvetica-Bold", fontSize=9.5,
                             leading=12, textColor=NAVY, spaceBefore=7, spaceAfter=2),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontSize=8.5, leading=11, textColor=INK),
        "bullet": ParagraphStyle("bl", parent=ss["Normal"], fontSize=8.3, leading=10.6, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontSize=6.8, leading=8.4, textColor=MUTED),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontSize=7.4, leading=9, textColor=INK),
    }


def _bullets(items, st):
    return ListFlowable([ListItem(Paragraph(_esc(t), st["bullet"]), leftIndent=9, value="•")
                         for t in items if t], bulletType="bullet", start="•", leftIndent=9,
                        bulletFontSize=7)


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _table(data, col_widths, header_rows=1, zebra=True, font=7.4):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font),
        ("FONT", (0, 0), (-1, header_rows - 1), "Helvetica-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), NAVY),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, RULE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]
    if zebra:
        for r in range(header_rows, len(data)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), FILL))
    t.setStyle(TableStyle(style))
    return t


def build_pdf(result, commentary: dict, config: dict, ctx: dict | None = None) -> bytes:
    """Render the brief; if long commentary spills past two pages, re-render in compact mode."""
    ctx = ctx or build_context(result, config)
    pdf, pages = _render(result, commentary, config, ctx, compact=False)
    if pages > 2:
        logger.info("Brief ran to %d pages; re-rendering compact", pages)
        pdf, pages = _render(result, commentary, config, ctx, compact=True)
    return pdf


def _render(result, commentary: dict, config: dict, ctx: dict, compact: bool) -> tuple[bytes, int]:
    if compact:  # trim lists, shrink charts
        commentary = {k: (v[:3] if isinstance(v, list) else v) for k, v in commentary.items()}
    st = _styles()
    buf = io.BytesIO()
    W = letter[0] - 1.2 * inch
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title=f"US & Canada 10Y Outlook {config.get('quarter', '')}")
    cur, end, rng = ctx["current"], ctx["forecast_end_weighted"], ctx["range_10_90_at_end"]
    story = []

    story.append(Paragraph(f"US &amp; Canada 10-Year Yield Outlook — {_esc(config.get('quarter', ''))}", st["title"]))
    story.append(Paragraph(
        f"Two-year scenario-weighted forecast · data through {ctx['as_of_month']} · generated "
        f"{date.today():%d %b %Y}", st["sub"]))
    story.append(Paragraph(_esc(commentary["headline"]), st["headline"]))

    kpi = [["", "Current", f"{ctx['horizon_end']} (weighted)", "Change", "10–90th pct", "Policy rate now → end"],
           ["US 10Y", f"{cur['us_10y']:.2f}%", f"{end['us_10y']:.2f}%",
            f"{(end['us_10y'] - cur['us_10y']) * 100:+.0f} bps", f"{rng['us_10y'][0]:.2f}–{rng['us_10y'][1]:.2f}%",
            f"Fed {cur['fed_funds']:.2f}% → {end['fed_funds']:.2f}%"],
           ["Canada 10Y", f"{cur['canada_10y']:.2f}%", f"{end['canada_10y']:.2f}%",
            f"{(end['canada_10y'] - cur['canada_10y']) * 100:+.0f} bps",
            f"{rng['canada_10y'][0]:.2f}–{rng['canada_10y'][1]:.2f}%",
            f"BoC {cur['boc_rate']:.2f}% → {end['boc_rate']:.2f}%"]]
    story.append(_table(kpi, [0.9 * inch, 0.8 * inch, 1.25 * inch, 0.8 * inch, 1.05 * inch, W - 4.8 * inch],
                        zebra=False, font=8))
    story.append(Spacer(1, 5))
    story.append(Paragraph(_esc(commentary["summary"]), st["body"]))
    story.append(Spacer(1, 4))
    fan_h = W * 2.55 / 7.4 * (0.82 if compact else 1.0)
    story.append(Image(chart_fans(result), width=W * (0.82 if compact else 1.0), height=fan_h))

    # Quarterly table
    q = result.quarterly
    rows = [["Quarter", "Fed funds", "US 10Y", "US 10–90th", "BoC", "Canada 10Y", "CA 10–90th", "CA–US spread"]]
    for idx, r in q.iterrows():
        rows.append([idx, f"{r['fed_funds']:.2f}", f"{r['us_10y']:.2f}",
                     f"{r['us_10y_p10']:.2f}–{r['us_10y_p90']:.2f}", f"{r['boc_rate']:.2f}",
                     f"{r['canada_10y']:.2f}", f"{r['canada_10y_p10']:.2f}–{r['canada_10y_p90']:.2f}",
                     f"{r['ca_spread']:+.2f}"])
    story.append(KeepTogether([Paragraph("Quarterly forecast (quarterly averages, %)", st["h2"]),
                               _table(rows, [W / 8] * 8)]))

    # Last quarter attribution
    att = ctx.get("last_quarter_attribution") or {}
    if att.get("splits"):
        arows = [["US 10Y change split", "Component A", "bps", "Component B", "bps"]]
        for name, parts in att["splits"].items():
            (la, va), (lb, vb) = list(parts.items())
            arows.append([name, la, f"{va:+.0f}", lb, f"{vb:+.0f}"])
        story.append(KeepTogether([
            Paragraph(f"What moved yields: {att['start']} → {att['end']} "
                      f"(US 10Y {att['us_10y_change_bps']:+.0f} bps)", st["h2"]),
            _table(arows, [1.75 * inch, 1.85 * inch, 0.5 * inch, 1.85 * inch, W - 5.95 * inch]),
            Spacer(1, 3),
            Paragraph(_esc(commentary.get("market_narrative", "")), st["body"]),
        ]))

    # Page 2 — analysis
    story.append(PageBreak())
    story.append(Paragraph("Macro backdrop — fiscal, Fed, energy, geopolitics", st["h2"]))
    story.append(_bullets(commentary.get("macro_backdrop", []), st))
    story.append(Paragraph("US 10Y — drivers", st["h2"]))
    story.append(_bullets(commentary["us_analysis"], st))
    k = 0.78 if compact else 1.0
    story.append(Image(chart_decomposition(result), width=W * k, height=W * 2.3 / 7.4 * k))
    story.append(Paragraph("Canada 10Y — spread and policy", st["h2"]))
    story.append(_bullets(commentary["canada_analysis"], st))

    # Scenarios
    srows = [["Scenario", "Weight", "Fed end", "BoC end", "US 10Y end", "Canada 10Y end"]]
    for n, s in ctx["scenarios"].items():
        srows.append([n, f"{s['probability']:.0%}", f"{s['fed_funds_end']:.2f}%", f"{s['boc_rate_end']:.2f}%",
                      f"{s['us_10y_end']:.2f}%", f"{s['canada_10y_end']:.2f}%"])
    story.append(KeepTogether([Paragraph("Scenarios", st["h2"]),
                               _table(srows, [1.6 * inch] + [(W - 1.6 * inch) / 5] * 5)]))

    # Risks side by side
    up = [Paragraph("<b>↑ Upside (higher yields)</b>", st["cell"])] + \
         [Paragraph("• " + _esc(x), st["cell"]) for x in commentary["upside_risks"]]
    dn = [Paragraph("<b>↓ Downside (lower yields)</b>", st["cell"])] + \
         [Paragraph("• " + _esc(x), st["cell"]) for x in commentary["downside_risks"]]
    risk_tbl = Table([[up, dn]], colWidths=[W / 2, W / 2])
    risk_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("BACKGROUND", (0, 0), (-1, -1), FILL),
                                  ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
    story.append(KeepTogether([Paragraph("Risks to the outlook", st["h2"]), risk_tbl]))

    story.append(Paragraph("Conclusion", st["h2"]))
    story.append(Paragraph(_esc(commentary["conclusion"]), st["body"]))

    a = ctx["anchors"]
    bt = ctx.get("backtest_rmse_vs_random_walk")
    bt_txt = (f" Backtest RMSE vs random walk (US, 12m/24m): {bt.get('us_10y_h12', '–')} / {bt.get('us_10y_h24', '–')}."
              if bt else "")
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Methodology: monthly structural model, US 10Y = expected average short rate (policy scenarios, "
        f"convergence to a {a['neutral_fed_funds']}% neutral Fed rate) + term premium (target "
        f"{a['term_premium_target']}%, {a['term_premium_method']}); Canada 10Y = US 10Y + Canada–US spread "
        f"driven by the BoC–Fed policy differential (BoC neutral {a['neutral_boc_rate']}%). Bands from a "
        f"block bootstrap of historical monthly changes around the probability-weighted scenarios.{bt_txt} "
        f"Data: FRED, Bank of Canada, NY Fed ACM. {_esc(commentary.get('_source', ''))}. "
        f"Draft for review — not investment advice.", st["small"]))

    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(letter[0] - 0.6 * inch, 0.3 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue(), doc.page


def generate_brief(result, config: dict, backtest_summary: pd.DataFrame | None = None,
                   use_ai: bool = True) -> tuple[bytes, dict]:
    """Context → commentary → PDF. Returns (pdf_bytes, commentary)."""
    ctx = build_context(result, config, backtest_summary)
    commentary = draft_commentary(ctx, use_ai=use_ai, api_key=config.get("anthropic_api_key"))
    return build_pdf(result, commentary, config, ctx), commentary
