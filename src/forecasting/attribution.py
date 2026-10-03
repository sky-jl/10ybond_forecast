"""
What moved the 10Y? Attribution of a yield change over a window (default: last completed quarter).

Three complementary splits of the US 10Y change (bps, monthly averages):
  1. Expectations vs term premium   Δ10Y = Δ risk-neutral yield (ACM/KW) + Δ term premium
  2. Real vs inflation compensation Δ10Y = Δ TIPS real yield + Δ breakeven
  3. Front end vs curve             Δ10Y = Δ 2Y (policy expectations) + Δ 2s10s slope

plus macro context (Fed funds, 5y5y inflation, oil, dollar, VIX, realised vol) and a rule-based
reading of the dominant theme the market traded: Fed policy, inflation/energy, or
term premium (fiscal / supply / risk premium). The reading is a heuristic summary of the numbers,
not a causal identification — it is meant to frame the narrative, which the analyst confirms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = {
    "Expectations vs term premium": [("us_rny", "Expected short rates (risk-neutral)"),
                                     ("us_tp", "Term premium")],
    "Real vs inflation": [("us_10y_real", "Real yield (TIPS)"),
                          ("us_10y_bei", "Breakeven inflation")],
    "Front end vs curve": [("us_2y", "2Y yield (Fed expectations)"),
                           ("_2s10s", "2s10s slope")],
}

CONTEXT = [  # (column, label, kind) — kind: "bps" level change, "pct" % change, "pts" point change
    ("fed_funds", "Fed funds", "bps"),
    ("us_5y5y_bei", "5y5y inflation expectations", "bps"),
    ("oil_brent", "Brent oil", "pct"),
    ("usd_broad", "Broad USD", "pct"),
    ("vix", "VIX", "pts"),
    ("us_10y_rvol", "10Y realised vol (bps/day)", "pts"),
    ("canada_10y", "Canada 10Y", "bps"),
    ("spread_can_us", "Canada–US spread", "bps"),
]


def last_completed_quarter(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(start, end) month-ends: end = latest quarter-end month with data, start = 3 months before."""
    q_ends = [d for d in df.index if d.month in (3, 6, 9, 12) and pd.notna(df.loc[d, "us_10y"])]
    end = q_ends[-1] if q_ends else df.index[-1]
    start = end - pd.offsets.MonthEnd(3)
    return start, end


def window(df: pd.DataFrame, months: int | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not months:
        return last_completed_quarter(df)
    end = df.index[-1]
    return end - pd.offsets.MonthEnd(months), end


def _chg(df: pd.DataFrame, col: str, start, end, kind: str = "bps") -> float | None:
    if col not in df or start not in df.index or end not in df.index:
        return None
    a, b = df.loc[start, col], df.loc[end, col]
    if pd.isna(a) or pd.isna(b):
        return None
    if kind == "pct":
        return float((b / a - 1) * 100)
    if kind == "pts":
        return float(b - a)
    return float((b - a) * 100)


def attribute(df: pd.DataFrame, months: int | None = None) -> dict:
    d = df.copy()
    if {"us_10y", "us_2y"} <= set(d.columns):
        d["_2s10s"] = d["us_10y"] - d["us_2y"]
    start, end = window(d, months)
    total = _chg(d, "us_10y", start, end)
    out = {"start": f"{start:%b %Y}", "end": f"{end:%b %Y}", "us_10y_change_bps": total,
           "us_10y_start": float(d.loc[start, "us_10y"]) if start in d.index else None,
           "us_10y_end": float(d.loc[end, "us_10y"]) if end in d.index else None,
           "splits": {}, "context": {}}
    for name, parts in SPLITS.items():
        vals = {label: _chg(d, col, start, end) for col, label in parts}
        if all(v is not None for v in vals.values()):
            out["splits"][name] = vals
    for col, label, kind in CONTEXT:
        v = _chg(d, col, start, end, kind)
        if v is not None:
            out["context"][label] = {"change": v, "unit": {"bps": "bps", "pct": "%", "pts": "pts"}[kind],
                                     "level": float(d.loc[end, col])}
    out["reading"] = read_drivers(out)
    return out


def read_drivers(a: dict) -> list[str]:
    """Heuristic narrative: which theme dominated the move."""
    total = a.get("us_10y_change_bps")
    if total is None:
        return []
    s, ctx = a["splits"], a["context"]
    lines = []
    direction = "rose" if total > 0 else "fell"
    lines.append(f"US 10Y {direction} {abs(total):.0f} bps ({a['start']} → {a['end']}, monthly averages).")
    if abs(total) < 5:
        lines.append("Move was small; no single theme dominated.")

    et = s.get("Expectations vs term premium")
    if et:
        exp, tp = et["Expected short rates (risk-neutral)"], et["Term premium"]
        if abs(abs(tp) - abs(exp)) < 0.3 * max(abs(total), 1) and np.sign(tp) == np.sign(exp):
            lines.append(f"Expected short rates ({exp:+.0f} bps) and term premium ({tp:+.0f} bps) contributed "
                         "about equally — both the Fed outlook and risk/fiscal premia moved.")
        elif abs(tp) > abs(exp) and np.sign(tp) == np.sign(total):
            lines.append(f"Term premium drove the move ({tp:+.0f} bps vs {exp:+.0f} bps from expected short "
                         "rates) — consistent with fiscal/supply, uncertainty or risk-premium repricing "
                         "rather than a change in the Fed outlook.")
        elif abs(exp) >= abs(tp) and np.sign(exp) == np.sign(total):
            lines.append(f"Policy expectations drove the move ({exp:+.0f} bps vs {tp:+.0f} bps term premium) "
                         "— the market repriced the expected Fed path.")
    ri = s.get("Real vs inflation")
    if ri:
        real, bei = ri["Real yield (TIPS)"], ri["Breakeven inflation"]
        oil = ctx.get("Brent oil", {}).get("change")
        if abs(bei) > abs(real) and np.sign(bei) == np.sign(total):
            msg = f"Inflation compensation led ({bei:+.0f} bps breakeven vs {real:+.0f} bps real)"
            if oil is not None and abs(oil) >= 10 and np.sign(oil) == np.sign(bei):
                msg += f", alongside a {oil:+.0f}% move in oil — an energy/inflation story"
            lines.append(msg + ".")
        elif np.sign(real) == np.sign(total):
            lines.append(f"Real yields led ({real:+.0f} bps vs {bei:+.0f} bps breakeven) — growth, policy or "
                         "term-premium forces rather than inflation fears.")
    fc = s.get("Front end vs curve")
    if fc:
        two, slope = fc["2Y yield (Fed expectations)"], fc["2s10s slope"]
        if abs(slope) > 10:
            shape = "steepened" if slope > 0 else "flattened"
            kind = ("bear" if total > 0 else "bull") + ("-steepening" if slope > 0 else "-flattening")
            lines.append(f"The 2s10s curve {shape} {abs(slope):.0f} bps ({kind}; 2Y {two:+.0f} bps).")
    lr = ctx.get("5y5y inflation expectations", {}).get("change")
    if lr is not None and abs(lr) >= 10:
        lines.append(f"Long-run inflation expectations (5y5y) moved {lr:+.0f} bps — watch Fed credibility / "
                     "independence and inflation-risk premium.")
    return lines


def attribution_table(a: dict) -> pd.DataFrame:
    rows = []
    for name, parts in a["splits"].items():
        for label, v in parts.items():
            rows.append({"split": name, "component": label, "change_bps": v})
    return pd.DataFrame(rows)
