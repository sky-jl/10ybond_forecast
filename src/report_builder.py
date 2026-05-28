"""Assemble the Word (.docx) forecast report from charts, metrics, and commentary."""

from __future__ import annotations

import logging
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor, Cm

logger = logging.getLogger(__name__)

DARK_BLUE = RGBColor(0x1B, 0x4F, 0x8A)
DARK_GREEN = RGBColor(0x2E, 0x7D, 0x32)
DARK_RED = RGBColor(0xC6, 0x28, 0x28)
ORANGE = RGBColor(0xE0, 0x7B, 0x39)
GREY = RGBColor(0x75, 0x75, 0x75)


def _heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    if level == 1:
        p.runs[0].font.color.rgb = DARK_BLUE
        p.runs[0].font.size = Pt(14)
    elif level == 2:
        p.runs[0].font.color.rgb = DARK_BLUE
        p.runs[0].font.size = Pt(11)


def _parse_bullets(doc: Document, text: str, indent: int = 0) -> None:
    """
    Render a block of bullet text into the doc.
    Lines starting with •, ↑, or ↓ become bullet paragraphs; others are normal text.
    Strips the [DRAFT — please review] marker out as a styled callout.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("[DRAFT"):
            p = doc.add_paragraph()
            run = p.add_run("⚠ DRAFT — Please review and edit before distributing")
            run.font.color.rgb = ORANGE
            run.font.bold = True
            run.font.size = Pt(9)
            continue

        if stripped.startswith(("•", "↑", "↓", "-")):
            p = doc.add_paragraph(style="List Bullet")
            marker = stripped[0] if stripped[0] in ("↑", "↓") else ""
            body = stripped.lstrip("•↑↓- ").strip()

            if marker == "↑":
                run = p.add_run("↑ ")
                run.font.color.rgb = DARK_RED
                run.font.bold = True
            elif marker == "↓":
                run = p.add_run("↓ ")
                run.font.color.rgb = DARK_GREEN
                run.font.bold = True

            if "[Upside]" in body or "[Downside]" in body:
                tag = "[Upside]" if "[Upside]" in body else "[Downside]"
                parts = body.split(tag, 1)
                run2 = p.add_run(tag)
                run2.font.bold = True
                p.add_run(parts[1] if len(parts) > 1 else "")
            else:
                p.add_run(body)

            p.paragraph_format.left_indent = Cm(0.5 + indent * 0.5)
        else:
            doc.add_paragraph(stripped)


def _target_box(doc: Document, label: str, target: float, color: RGBColor, fill: str) -> None:
    """Add a single-country shaded target box."""
    table = doc.add_table(rows=1, cols=1)
    table.style = "Table Grid"

    cell = table.rows[0].cells[0]
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)

    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_label = p.add_run(label + "\n")
    run_label.font.size = Pt(9)
    run_label.font.color.rgb = GREY

    run_val = p.add_run(f"{target:.1f}%")
    run_val.font.size = Pt(22)
    run_val.font.bold = True
    run_val.font.color.rgb = color

    doc.add_paragraph()


def _add_chart(doc: Document, chart_path: str, caption: str, width: float = 6.0) -> None:
    if not chart_path or not Path(chart_path).exists():
        doc.add_paragraph(f"[Chart not available: {caption}]")
        return
    doc.add_picture(chart_path, width=Inches(width))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(8)
    cap.runs[0].font.color.rgb = GREY
    doc.add_paragraph()


def _metrics_snapshot_table(doc: Document, metrics: dict) -> None:
    """Append a compact table of key quantitative metrics."""
    rows = [
        ("Metric", "Current", "5Y Avg", "10Y Avg", "20Y Avg"),
        ("US 10Y Yield (%)", metrics["us_10y_current"],
         metrics["us_10y_mean_5y"], metrics["us_10y_mean_10y"], metrics["us_10y_mean_20y"]),
        ("Canada 10Y Yield (%)", metrics["canada_10y_current"],
         metrics["canada_10y_mean_5y"], metrics["canada_10y_mean_10y"], metrics["canada_10y_mean_20y"]),
        ("Fed Funds Rate (%)", metrics.get("fed_funds_current", "—"), "—", "—", "—"),
        ("BoC Overnight Rate (%)", metrics.get("boc_rate_current", "—"), "—", "—", "—"),
        ("US 10Y–FFR Spread (pp)", metrics["spread_10y_ffr_current"],
         metrics["spread_10y_ffr_mean_5y"], metrics["spread_10y_ffr_mean_10y"], metrics["spread_10y_ffr_mean_20y"]),
        ("Canada 10Y–BoC Spread (pp)", metrics["spread_canada_10y_boc_current"],
         metrics["spread_canada_10y_boc_mean_5y"], metrics["spread_canada_10y_boc_mean_10y"], metrics["spread_canada_10y_boc_mean_20y"]),
        ("Canada–US 10Y Spread (pp)", metrics["spread_can_us_current"],
         metrics["spread_can_us_mean_5y"], metrics["spread_can_us_mean_10y"], metrics["spread_can_us_mean_20y"]),
        ("ACM Term Premium (%)", metrics["acm_tp_current"],
         metrics["acm_tp_mean_5y"], metrics["acm_tp_mean_10y"], metrics["acm_tp_mean_20y"]),
        ("10Y Breakeven Inflation (%)", metrics["us_10y_bei_current"],
         metrics["us_10y_bei_mean_5y"], metrics["us_10y_bei_mean_10y"], metrics["us_10y_bei_mean_20y"]),
    ]

    table = doc.add_table(rows=len(rows), cols=5)
    table.style = "Light Shading Accent 1"

    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.rows[r_idx].cells[c_idx]
            if isinstance(val, float):
                cell.text = f"{val:.2f}"
            else:
                cell.text = str(val)
            if r_idx == 0:
                cell.paragraphs[0].runs[0].font.bold = True


def build_report(
    metrics: dict,
    commentary: dict[str, str],
    chart_paths: dict[str, str],
    config: dict,
    output_path: Path,
    country: str = "us",
) -> str:
    """Assemble and save the Word report for one country. Returns the output file path."""
    is_us = country == "us"

    title_text = "US Long-Term Bond Yield Forecast" if is_us else "Canada Long-Term Bond Yield Forecast"
    target_label = "US 10-Year Yield Target" if is_us else "Canada 10-Year Yield Target"
    target_value = config["us_10y"]["target"] if is_us else config["canada_10y"]["target"]
    title_color = DARK_BLUE if is_us else DARK_RED
    target_fill = "F0F4FA" if is_us else "FFF3F3"

    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # =====================================================================
    # TITLE PAGE
    # =====================================================================
    title = doc.add_heading(title_text, 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.runs[0].font.color.rgb = title_color

    sub = doc.add_paragraph(config["quarter"])
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(14)
    sub.runs[0].font.color.rgb = GREY

    date_p = doc.add_paragraph(f"Generated: {config['run_date']}    |    Author: _______________")
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.runs[0].font.size = Pt(9)
    date_p.runs[0].font.color.rgb = GREY

    doc.add_paragraph()
    _target_box(doc, target_label, target_value, title_color, target_fill)
    doc.add_page_break()

    # =====================================================================
    # SECTION 1 — BACKGROUND RESEARCH
    # =====================================================================
    _heading(doc, "1.  Background Research", level=1)
    _heading(doc, "Recent History and Last-Quarter Movements", level=2)
    _parse_bullets(doc, commentary.get("background", "[Commentary not generated]"))
    doc.add_paragraph()

    if is_us:
        _add_chart(doc, chart_paths.get("us_10y", ""),
                   "Figure 1: US 10-Year Treasury Yield (10-Year History)")
    else:
        _add_chart(doc, chart_paths.get("canada_10y", ""),
                   "Figure 1: Canada 10-Year GoC Bond Yield (10-Year History)")

    doc.add_page_break()

    # =====================================================================
    # SECTION 2 — SOURCES
    # =====================================================================
    _heading(doc, "2.  Sources Used", level=1)
    _parse_bullets(doc, commentary.get("sources", "[Sources not generated]"))
    doc.add_paragraph()
    doc.add_page_break()

    # =====================================================================
    # SECTION 3 — FORECASTING RATIONALE
    # =====================================================================
    _heading(doc, "3.  Forecasting Rationale", level=1)
    _heading(doc, "Factors Affecting Long-Term Yield Dynamics", level=2)

    m = metrics["methodology"]
    callout = doc.add_paragraph()
    if is_us:
        callout.add_run("Three-framework convergence  |  Target: ").font.bold = True
        callout.add_run(f"{m['user_target_us_10y']:.1f}%\n").font.bold = True
        t = doc.add_table(rows=4, cols=3)
        t.style = "Light Shading Accent 1"
        headers = ["Framework", "Approach", "Implied 10Y Range"]
        for c, h in enumerate(headers):
            cell = t.rows[0].cells[c]
            cell.text = h
            cell.paragraphs[0].runs[0].font.bold = True
        data = [
            ("1 — Spread-based",
             f"r* ({m['long_run_fed_funds']:.2f}%) + hist. 10Y–FFR spread (TP implicit)",
             f"{m['implied_10y_using_5y_avg_spread']:.2f}% – {m['implied_10y_using_20y_avg_spread']:.2f}%"),
            ("2 — ACM decomposition",
             f"r* ({m['long_run_fed_funds']:.2f}%) + avg term premium (explicit TP)",
             f"{m['implied_10y_acm_20y_avg_tp']:.2f}% – {m['implied_10y_acm_10y_avg_tp']:.2f}%"),
            ("3 — GDP + Inflation",
             f"Real GDP ({m['long_run_real_gdp_growth']:.2f}%) + PCE ({m['long_run_pce_inflation']:.2f}%)",
             f"{m['implied_10y_gdp_inflation']:.2f}%"),
        ]
        for r, (fw, approach, rng) in enumerate(data, start=1):
            t.rows[r].cells[0].text = fw
            t.rows[r].cells[1].text = approach
            t.rows[r].cells[2].text = rng
        doc.add_paragraph()
    else:
        callout.add_run("Canada 10Y implied range — Anchor 1 (BoC r* + spread):  ").font.bold = True
        callout.add_run(
            f"{m['implied_canada_10y_using_5y_boc_spread']:.2f}% – {m['implied_canada_10y_using_20y_boc_spread']:.2f}%  "
            f"|  Anchor 2 (US target + CAN–US spread):  "
            f"{m['implied_canada_10y_using_5y_canus_spread']:.2f}% – {m['implied_canada_10y_using_10y_canus_spread']:.2f}%  "
            f"|  Target: {m['user_target_canada_10y']:.1f}%"
        )
    callout.paragraph_format.left_indent = Cm(0.5)

    doc.add_paragraph()
    _parse_bullets(doc, commentary.get("rationale", "[Commentary not generated]"))
    doc.add_paragraph()

    if is_us:
        _add_chart(doc, chart_paths.get("spread_10y_ffr", ""),
                   "Figure 2: US 10Y–Fed Funds Rate Spread vs Historical Average")
        _add_chart(doc, chart_paths.get("acm_term_premium", ""),
                   "Figure 3: ACM 10-Year Term Premium")
    else:
        _add_chart(doc, chart_paths.get("spread_canada_boc", ""),
                   "Figure 2: Canada 10Y–BoC Overnight Rate Spread vs Historical Average")
        _add_chart(doc, chart_paths.get("canada_us_spread", ""),
                   "Figure 3: Canada–US 10-Year Yield Spread (5-Year History)")

    doc.add_page_break()

    # =====================================================================
    # SECTION 4 — MEDIUM-TERM RISKS
    # =====================================================================
    _heading(doc, "4.  Medium-Term Risks and Impact on Outlook (3–5 Years)", level=1)
    _parse_bullets(doc, commentary.get("medium_term_risks", "[Commentary not generated]"))
    doc.add_paragraph()
    doc.add_page_break()

    # =====================================================================
    # SECTION 5 — LONG-TERM RISKS
    # =====================================================================
    _heading(doc, "5.  Long-Term Risks and Impact on Outlook (5–10 Years)", level=1)
    _parse_bullets(doc, commentary.get("long_term_risks", "[Commentary not generated]"))
    doc.add_paragraph()
    doc.add_page_break()

    # =====================================================================
    # APPENDIX
    # =====================================================================
    _heading(doc, "Appendix: Supporting Data", level=1)
    doc.add_paragraph()
    _heading(doc, "Key Metrics Snapshot", level=2)
    _metrics_snapshot_table(doc, metrics)

    # =====================================================================
    # SAVE
    # =====================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    logger.info("Report saved to %s", output_path)
    return str(output_path)
