"""
Interactive 10Y forecast lab (Streamlit).

    streamlit run app.py

Adjust the subjective inputs in the sidebar / editors and the 2-year US and Canada 10Y
forecast is recomputed with the same model as forecast.py. Uses the monthly dataset cached
by `python forecast.py` (output/forecast/monthly_dataset.csv).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")  # ANTHROPIC_API_KEY for the PDF report commentary

from data_fetcher import load_monthly_dataset  # noqa: E402
from forecasting import run_forecast  # noqa: E402
from forecasting.attribution import attribute  # noqa: E402
from forecasting.backtest import run_backtest  # noqa: E402
from forecasting.term_premium import DRIVER_LABELS, fit_tp_regression  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "quarterly_config.yaml"
DEFAULT_DATA = ROOT / "output" / "forecast" / "monthly_dataset.csv"

# Categorical slots (fixed order; colour follows the scenario, never its rank)
SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

st.set_page_config(page_title="10Y Forecast Lab", page_icon="📈", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")  # research progress shows in the terminal


# ------------------------------------------------------------------------------ theme
def _dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


DARK = _dark()
INK = "#ffffff" if DARK else "#0b0b0b"
MUTED = "#c3c2b7" if DARK else "#52514e"
SLOTS = SLOTS_DARK if DARK else SLOTS_LIGHT
BAND = "57,135,229" if DARK else "42,120,214"


# ------------------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str, mtime: float, synthetic: bool) -> pd.DataFrame:
    if synthetic:
        from forecasting.synthetic import make_synthetic_monthly
        return make_synthetic_monthly()
    return load_monthly_dataset(path)


@st.cache_data(show_spinner="Running forecast…")
def cached_forecast(_df: pd.DataFrame, data_key: str, cfg_json: str):
    return run_forecast(_df, json.loads(cfg_json))


@st.cache_data(show_spinner=False)
def cached_tp_fit(_df: pd.DataFrame, data_key: str):
    try:
        return fit_tp_regression(_df)
    except ValueError:
        return None


@st.cache_data(show_spinner="Running backtest (≈5–20 s)…")
def cached_backtest(_df: pd.DataFrame, data_key: str, cfg_json: str):
    return run_backtest(_df, json.loads(cfg_json))["summary"]


# ------------------------------------------------------------------------------ helpers
def anchors_to_frame(scenarios: list[dict], key: str) -> pd.DataFrame:
    periods = sorted({str(p) for s in scenarios for p in (s.get(key) or {})})
    data = {s["name"]: [(s.get(key) or {}).get(p, np.nan) for p in periods] for s in scenarios}
    return pd.DataFrame(data, index=pd.Index(periods, name="period"))


def rate_columns(frame: pd.DataFrame) -> dict:
    """Decimal number columns (3 dp) so rate paths like 3.375 can be typed in any Streamlit version."""
    return {c: st.column_config.NumberColumn(c, min_value=-1.0, max_value=20.0, step=0.001, format="%.3f")
            for c in frame.columns}


def frame_to_anchors(frame: pd.DataFrame, name: str) -> dict:
    col = frame[name] if name in frame else pd.Series(dtype=float)
    return {str(p).strip(): float(v) for p, v in col.items()
            if str(p).strip() and str(p) != "nan" and pd.notna(v)}


def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def base_layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15, x=0),
        yaxis=dict(title="%", ticksuffix="", zeroline=False),
        xaxis=dict(showgrid=False),
    )
    return fig


def color_map(names: list[str]) -> dict[str, str]:
    return {n: SLOTS[i % len(SLOTS)] for i, n in enumerate(names)}


# ------------------------------------------------------------------------------ sidebar: data
st.sidebar.title("📈 10Y Forecast Lab")
cfg_path = st.sidebar.text_input("Config file", str(DEFAULT_CONFIG.relative_to(ROOT)))
base_cfg = load_config(str(ROOT / cfg_path))
fbase = base_cfg.get("forecast", {})

data_path = ROOT / st.sidebar.text_input("Monthly dataset (CSV)", str(DEFAULT_DATA.relative_to(ROOT)))
has_data = data_path.exists()
synthetic = st.sidebar.toggle("Use synthetic demo data", value=not has_data,
                              help="Fake data for trying the app before running forecast.py")
if not has_data and not synthetic:
    st.error(f"Dataset not found: `{data_path}`. Run `python forecast.py` once to download and "
             "cache it, or switch on synthetic demo data.")
    st.stop()
df = load_data(str(data_path), data_path.stat().st_mtime if has_data else 0.0, synthetic)
data_key = f"{data_path}|{synthetic}|{df.index[-1]}|{len(df)}"
if synthetic:
    st.warning("Synthetic demo data — numbers are NOT real market data.", icon="⚠️")

if st.sidebar.button("↺ Reset all inputs to config", width="stretch"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ------------------------------------------------------------------------------ sidebar: views
sb = st.sidebar
H = int(fbase.get("horizon_months", 24))
last_obs = df.index[-1]
horizon_end = (last_obs + pd.offsets.MonthEnd(H)).strftime("%Y-%m")

with sb.expander("Neutral rates & expectations", expanded=True):
    neutral_fed = st.number_input("Fed neutral rate (%)", 0.0, 10.0,
                            float(fbase.get("neutral_fed_funds") or base_cfg.get("long_run_fed_funds", 3.0)), 0.125,
                            format="%.3f",
                            help="Long-run nominal Fed funds rate the short rate converges to beyond the horizon")
    neutral_boc = st.number_input("BoC neutral rate (%)", 0.0, 10.0,
                            float(fbase.get("neutral_boc_rate") or base_cfg.get("long_run_boc_rate", 2.75)), 0.125,
                            format="%.3f")
    exp_cfg = fbase.get("expectations", {})
    conv_hl = st.slider("Convergence half-life beyond horizon (months)", 6, 120,
                        int(exp_cfg.get("convergence_halflife_months", 36)), 6)
    basis_hl = st.slider("Market-vs-neutral gap half-life (months)", 3, 120,
                         int(exp_cfg.get("basis_halflife_months", 36)), 3,
                         help="How fast today's gap between market-implied expectations and your "
                              "neutral view disappears. Short = market converges to your view quickly.")
    if "us_rny" in df and pd.notna(df["us_rny"].dropna().iloc[-1]):
        st.caption(f"Market risk-neutral 10Y today: **{df['us_rny'].dropna().iloc[-1]:.2f}%** "
                   f"(10Y − term premium)")

tp_fit = cached_tp_fit(df, f"{data_path}|{synthetic}|{df.index[-1]}|{len(df)}")

with sb.expander("Term premium (fiscal & macro risk)", expanded=True):
    tp_cfg = fbase.get("term_premium", {})
    tp_default = tp_cfg.get("target", "historical")
    modes = ["Your view (number)", "Historical mean", "Fiscal fair value (regression)"]
    mode_idx = 0 if isinstance(tp_default, (int, float)) else (2 if tp_default == "regression" else 1)
    tp_mode = st.radio("Target", modes, index=mode_idx, horizontal=False)
    tp_now = float(df["us_tp"].dropna().iloc[-1]) if "us_tp" in df else np.nan
    st.caption(f"Current term premium: **{tp_now:.2f}%**")
    tp_value = st.number_input("Term premium target (%)", -3.0, 5.0,
                         float(tp_default) if isinstance(tp_default, (int, float)) else round(tp_now, 2), 0.05,
                         format="%.2f", disabled=tp_mode != modes[0])
    tp_window = st.slider("Historical window (years)", 5, 35, int(tp_cfg.get("historical_window_years", 30)),
                          disabled=tp_mode != modes[1])
    if tp_mode == modes[1] and "us_tp" in df:
        tp = df["us_tp"].dropna()
        st.caption(f"→ mean over {tp_window}y: **{tp[tp.index >= tp.index[-1] - pd.DateOffset(years=tp_window)].mean():.2f}%**")
    tp_drivers = {}
    if tp_fit:
        st.caption(f"Fiscal fair value today: **{tp_fit['fair_value_now']:.2f}%** "
                   f"(actual {tp_fit['tp_now']:.2f}%, {tp_fit['residual_now_bps']:+.0f} bps gap)")
    if tp_mode == modes[2]:
        if not tp_fit:
            st.warning("Fiscal driver data not available — falls back to the historical mean.")
        else:
            st.caption("Driver values you expect at the horizon end:")
            saved = tp_cfg.get("drivers") or {}
            for c in tp_fit["drivers"]:
                tp_drivers[c] = st.number_input(
                    DRIVER_LABELS.get(c, c), -50.0, 300.0,
                    float(saved.get(c, round(tp_fit["latest_drivers"][c], 1))), 0.5, format="%.1f",
                    key=f"drv_{c}", help=f"Latest: {tp_fit['latest_drivers'][c]:.1f} · "
                                         f"TP sensitivity: {tp_fit['coef'][c] * 100:+.1f} bps per unit")
    st.markdown("**Macro risk-premium add-ons (bps)**")
    saved_adds = tp_cfg.get("addons_bps") or {}
    add_names = list(dict.fromkeys(["fed_independence", "geopolitics", "energy_inflation", "fiscal_supply",
                                    *saved_adds]))
    addons = {}
    ac = st.columns(2)
    for i, nm in enumerate(add_names):
        addons[nm] = ac[i % 2].number_input(nm.replace("_", " ").capitalize(), -100, 200,
                                            int(saved_adds.get(nm, 0) or 0), 5, key=f"add_{nm}")
    tp_hl_on = st.checkbox("Set adjustment half-life manually", value=bool(tp_cfg.get("halflife_months")))
    tp_hl = st.slider("TP half-life (months)", 3, 120, int(tp_cfg.get("halflife_months") or 24), 3,
                      disabled=not tp_hl_on)

with sb.expander("Canada–US spread"):
    sp_cfg = fbase.get("canada_spread", {})
    sp_now = float(df["spread_can_us"].dropna().iloc[-1])
    sp_fixed = st.toggle("Use your spread view instead of the model",
                         value=isinstance(sp_cfg.get("target"), (int, float)),
                         help="Model: spread follows the BoC − Fed policy differential of each scenario")
    sp_value = st.number_input("Spread target (pp)", -5.0, 5.0,
                         float(sp_cfg["target"]) if isinstance(sp_cfg.get("target"), (int, float)) else round(sp_now, 2),
                         0.05, format="%.2f", disabled=not sp_fixed)
    st.caption(f"Current Canada–US spread: **{sp_now:+.2f} pp**")
    boc_pass = st.slider("Fed→BoC pass-through (when BoC path not set)", 0.0, 1.0,
                         float(fbase.get("boc_fed_passthrough", 0.5)), 0.05)

with sb.expander("Judgmental overlay"):
    st.caption(f"Added on top of the model, phased in linearly to {horizon_end}.")
    us_ov = st.slider("US 10Y overlay (bps)", -100, 100, 0, 5)
    ca_ov = st.slider("Canada 10Y overlay (bps)", -100, 100, 0, 5)

with sb.expander("Uncertainty & display"):
    unc_cfg = fbase.get("uncertainty", {})
    unc_scale = st.slider("Band width scale", 0.5, 2.0, float(unc_cfg.get("scale", 1.0)), 0.1)
    n_sims = st.select_slider("Simulations", [1000, 2000, 5000, 10000], int(unc_cfg.get("n_sims", 5000)))
    hist_years = st.slider("History shown (years)", 1, 15, 4)

# ------------------------------------------------------------------------------ main: scenarios
st.title("US & Canada 10Y — 2-Year Forecast Lab")
st.caption(f"Forecast origin: **{last_obs:%b %Y}** (monthly averages) · horizon {H} months · "
           "edit the tables below or the sidebar; everything recomputes instantly.")

scen_cfg = copy.deepcopy(fbase.get("scenarios", []) or [])
ms_cfg = fbase.get("model_scenario", {}) or {}
model_name = ms_cfg.get("name", "Model (Taylor rule)")

st.subheader("1 · Policy-rate scenarios")
c1, c1b = st.columns([1.2, 1])
with c1:
    st.markdown("**Scenario weights & term-premium views**")
    meta = pd.DataFrame({
        "probability": [float(s.get("probability", 0)) for s in scen_cfg],
        "TP target (%)": [float(s.get("term_premium_target", np.nan)) for s in scen_cfg],
    }, index=pd.Index([s["name"] for s in scen_cfg], name="scenario"))
    meta = st.data_editor(
        meta, key="meta", width="stretch",
        column_config={
            "probability": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
            "TP target (%)": st.column_config.NumberColumn(
                help="Optional scenario-specific term premium; blank = global setting", format="%.2f"),
        })
with c1b:
    st.markdown("**Objective cross-check**")
    model_prob = st.slider(f"Weight on '{model_name}' (endogenous Taylor-rule path)", 0.0, 1.0,
                           float(ms_cfg.get("probability", 0.1)), 0.05)
    total = float(meta["probability"].fillna(0).sum()) + model_prob
    st.caption(f"Weights sum to **{total:.2f}** — rescaled to 1 automatically.")
c2, c3 = st.columns(2)
with c2:
    st.markdown("**Fed funds path (% at period end)**")
    fed_in = anchors_to_frame(scen_cfg, "fed_funds").astype(float)
    fed_tbl = st.data_editor(fed_in, key="fed", num_rows="dynamic", width="stretch",
                             column_config=rate_columns(fed_in))
    st.caption(f"Latest: **{df['fed_funds'].iloc[-1]:.2f}%** · periods like 2027Q2 or 2027-06")
with c3:
    st.markdown("**BoC overnight path (% at period end)**")
    boc_in = anchors_to_frame(scen_cfg, "boc_rate").astype(float)
    boc_tbl = st.data_editor(boc_in, key="boc", num_rows="dynamic", width="stretch",
                             column_config=rate_columns(boc_in))
    st.caption(f"Latest: **{df['boc_rate'].iloc[-1]:.2f}%** · blank column = model BoC path")

# ------------------------------------------------------------------------------ assemble config
cfg = copy.deepcopy(base_cfg)
f = cfg.setdefault("forecast", {})
new_scen = []
for s in scen_cfg:
    name = s["name"]
    sc = {"name": name, "probability": float(meta.loc[name, "probability"] or 0) if name in meta.index else 0.0,
          "fed_funds": frame_to_anchors(fed_tbl, name)}
    boc = frame_to_anchors(boc_tbl, name)
    if boc:
        sc["boc_rate"] = boc
    tpv = meta.loc[name, "TP target (%)"] if name in meta.index else np.nan
    if pd.notna(tpv):
        sc["term_premium_target"] = float(tpv)
    if sc["probability"] > 0:
        new_scen.append(sc)
f["scenarios"] = new_scen
f["model_scenario"] = {"include": True, "name": model_name, "probability": model_prob}
f["neutral_fed_funds"] = neutral_fed
f["neutral_boc_rate"] = neutral_boc
f["expectations"] = {"convergence_halflife_months": conv_hl, "basis_halflife_months": basis_hl}
f["term_premium"] = {
    **(fbase.get("term_premium") or {}),
    "target": tp_value if tp_mode == modes[0] else ("regression" if tp_mode == modes[2] else "historical"),
    "historical_window_years": tp_window,
    "halflife_months": tp_hl if tp_hl_on else None,
    "drivers": tp_drivers or (fbase.get("term_premium") or {}).get("drivers"),
    "addons_bps": {k: v for k, v in addons.items() if v},
}
f["canada_spread"] = {**(fbase.get("canada_spread") or {}), "target": sp_value if sp_fixed else "model"}
f["boc_fed_passthrough"] = boc_pass
f["overlay"] = {"us_10y_bps": {horizon_end: us_ov} if us_ov else {},
                "canada_10y_bps": {horizon_end: ca_ov} if ca_ov else {}}
f["uncertainty"] = {**(fbase.get("uncertainty") or {}), "scale": unc_scale, "n_sims": n_sims}

if not new_scen and model_prob <= 0:
    st.error("All scenario weights are zero — give at least one scenario a positive weight.")
    st.stop()

cfg_json = json.dumps(cfg, default=str, sort_keys=True)
try:
    res = cached_forecast(df, data_key, cfg_json)
except Exception as exc:
    st.error(f"Forecast failed: {exc}")
    st.stop()

colors = color_map(list(res.scenarios))

# ------------------------------------------------------------------------------ KPIs
st.subheader("2 · Results")
end_label = res.index[-1].strftime("%b %Y")
k = st.columns(4)
for i, (col, label) in enumerate([("us_10y", "US 10Y"), ("canada_10y", "Canada 10Y")]):
    now = float(res.central[col].iloc[0])
    end = float(res.central[col].iloc[-1])
    k[2 * i].metric(f"{label} · now ({last_obs:%b %Y})", fmt_pct(now))
    k[2 * i + 1].metric(f"{label} · {end_label} (weighted)", fmt_pct(end), f"{(end - now) * 100:+.0f} bps",
                        delta_color="off",
                        help=f"10–90th pct: {res.fan[col + '_p10'].iloc[-1]:.2f}% – {res.fan[col + '_p90'].iloc[-1]:.2f}%")


def fan_chart(col: str, label: str) -> go.Figure:
    hist = res.history[col].dropna()
    hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=hist_years)]
    fan = res.fan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color=MUTED, width=1.5),
                             hovertemplate="%{y:.2f}%"))
    for lo, hi, a, nm in [("p10", "p90", 0.13, "10–90th pct"), ("p25", "p75", 0.25, "25–75th pct")]:
        fig.add_trace(go.Scatter(x=fan.index, y=fan[f"{col}_{hi}"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fan.index, y=fan[f"{col}_{lo}"], fill="tonexty", line=dict(width=0),
                                 fillcolor=f"rgba({BAND},{a})", name=nm, hoverinfo="skip"))
    for name, sc in res.scenarios.items():
        fig.add_trace(go.Scatter(x=sc.index, y=sc[col], name=f"{name} ({res.probabilities[name]:.0%})",
                                 line=dict(color=colors[name], width=1.5, dash="dash"),
                                 hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=res.central.index, y=res.central[col], name="Probability-weighted",
                             line=dict(color=INK, width=2.5), hovertemplate="%{y:.2f}%"))
    fig.add_vline(x=res.as_of, line=dict(color=MUTED, width=1, dash="dot"))
    return base_layout(fig, f"{label} — history and 2-year forecast")


def decomposition_chart() -> go.Figure:
    c = res.central
    fig = go.Figure()
    parts = [("us_expectations", "Expected avg short rate", SLOTS[0]),
             ("us_basis", "Market-vs-neutral gap", SLOTS[6]),
             ("us_tp", "Term premium", SLOTS[1]),
             ("us_overlay", "Judgmental overlay", SLOTS[3])]
    for colname, nm, colr in parts:
        if colname == "us_overlay" and c[colname].abs().max() < 1e-9:
            continue
        fig.add_trace(go.Bar(x=c.index, y=c[colname], name=nm, marker=dict(color=colr),
                             hovertemplate="%{y:.2f}"))
    fig.add_trace(go.Scatter(x=c.index, y=c["us_10y"], name="US 10Y forecast",
                             line=dict(color=INK, width=2.5), hovertemplate="%{y:.2f}%"))
    fig.update_layout(barmode="relative", bargap=0.15)
    return base_layout(fig, "US 10Y = expectations + gap + term premium (+ overlay)")


def policy_chart(col: str, label: str) -> go.Figure:
    hist = res.history[col].dropna()
    hist = hist[hist.index >= hist.index[-1] - pd.DateOffset(years=min(hist_years, 5))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="History", line=dict(color=MUTED, width=1.5),
                             hovertemplate="%{y:.2f}%"))
    for name, sc in res.scenarios.items():
        fig.add_trace(go.Scatter(x=sc.index, y=sc[col], name=name, line=dict(color=colors[name], width=2),
                                 hovertemplate="%{y:.2f}%"))
    fig.add_trace(go.Scatter(x=res.central.index, y=res.central[col], name="Weighted",
                             line=dict(color=INK, width=2.5), hovertemplate="%{y:.2f}%"))
    return base_layout(fig, label, height=360)


tabs = st.tabs(["US 10Y", "Canada 10Y", "Decomposition", "Policy paths", "Tables", "Backtest",
                "PDF report", "Export config", "Macro drivers"])
with tabs[0]:
    st.plotly_chart(fan_chart("us_10y", "US 10Y Treasury"), theme="streamlit")
with tabs[1]:
    st.plotly_chart(fan_chart("canada_10y", "Canada 10Y GoC"), theme="streamlit")
    beta = res.params.get("beta_ca_us_36m")
    st.caption(f"Canada–US spread at {end_label}: **{res.central['ca_spread'].iloc[-1]:+.2f} pp** "
               f"(now {res.central['ca_spread'].iloc[0]:+.2f}) · 36-month US→Canada yield beta: "
               f"**{beta:.2f}**" if beta is not None else "")
with tabs[2]:
    st.plotly_chart(decomposition_chart(), theme="streamlit")
    e = res.central.iloc[[0, -1]][["us_expectations", "us_basis", "us_tp", "us_overlay", "us_10y"]].T
    e.columns = ["Now", end_label]
    e["Change (bps)"] = (e[end_label] - e["Now"]) * 100
    st.dataframe(e.rename(index={"us_expectations": "Expected avg short rate", "us_basis": "Market-vs-neutral gap",
                                 "us_tp": "Term premium", "us_overlay": "Overlay", "us_10y": "US 10Y"})
                 .style.format({"Now": "{:.2f}", end_label: "{:.2f}", "Change (bps)": "{:+.0f}"}),
                 width="stretch")
with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(policy_chart("fed_funds", "Fed funds rate"), theme="streamlit")
    b.plotly_chart(policy_chart("boc_rate", "BoC overnight rate"), theme="streamlit")
with tabs[4]:
    q = res.quarterly
    show = ["fed_funds", "us_10y", "us_10y_p10", "us_10y_p90", "boc_rate", "canada_10y",
            "canada_10y_p10", "canada_10y_p90", "us_tp", "ca_spread"]
    st.markdown("**Quarterly averages (%)**")
    st.dataframe(q[show].style.format("{:.2f}"), width="stretch")
    st.download_button("Download quarterly CSV", q.to_csv().encode(), "forecast_quarterly.csv", "text/csv")
    st.markdown("**By scenario (quarterly averages, %)**")
    st.dataframe(q[[c for c in q.columns if "[" in c]].style.format("{:.2f}"), width="stretch")
    with st.expander("Model parameters (estimated / used)"):
        st.json(json.loads(json.dumps(res.params, default=float)))
with tabs[5]:
    st.markdown("Pseudo real-time backtest of the **objective** model (endogenous Taylor-rule path, no "
                "scenarios or overlay) vs benchmarks. `rmse_vs_rw` < 1 beats a random walk.")
    if st.button("Run backtest"):
        st.session_state["bt"] = cached_backtest(df, data_key, cfg_json)
    if "bt" in st.session_state:
        s = st.session_state["bt"]
        target = st.radio("Target", ["us_10y", "canada_10y"], horizontal=True)
        sub = s[s["target"] == target]
        piv = sub[sub["model"] != "rw"].pivot(index="model", columns="h", values="rmse_vs_rw")
        fig = go.Figure()
        for i, m in enumerate(piv.index):
            fig.add_trace(go.Bar(x=[f"{h}m" for h in piv.columns], y=piv.loc[m], name=m,
                                 marker=dict(color=INK if m == "model" else SLOTS[(i + 1) % len(SLOTS)]),
                                 hovertemplate="%{y:.3f}"))
        fig.add_hline(y=1.0, line=dict(color=MUTED, dash="dot"), annotation_text="random walk = 1")
        fig.update_layout(barmode="group")
        st.plotly_chart(base_layout(fig, "RMSE relative to random walk (lower is better)", 360)
                        .update_yaxes(title="RMSE / RW RMSE"), theme="streamlit")
        st.dataframe(sub.drop(columns="target").style.format(
            {"rmse": "{:.3f}", "mae": "{:.3f}", "bias": "{:+.3f}", "rmse_vs_rw": "{:.3f}",
             "hit_rate": "{:.2f}", "dm_pvalue_vs_rw": "{:.2f}"}), width="stretch", hide_index=True)
with tabs[8]:
    st.markdown("### What moved yields — what was the market trading?")
    win_opts = {"Last completed quarter": None, "Last 1 month": 1, "Last 3 months": 3,
                "Last 6 months": 6, "Last 12 months": 12}
    win = st.radio("Window (monthly averages)", list(win_opts), horizontal=True)
    att = attribute(df, win_opts[win])
    if att["us_10y_change_bps"] is None:
        st.info("Not enough data in this window.")
    else:
        for line in att["reading"]:
            st.markdown(f"- {line}")
        labels, vals, cols = [], [], []
        for i, (name, parts) in enumerate(att["splits"].items()):
            for label, v in parts.items():
                labels.append(f"{label}  ·  {name}")
                vals.append(v)
                cols.append(SLOTS[i])
        short = [lab.split("  ·  ")[0] for lab in labels]
        fig = go.Figure(go.Bar(
            y=short[::-1], x=vals[::-1], orientation="h", marker=dict(color=cols[::-1]),
            text=[f"{v:+.0f}" for v in vals[::-1]], textposition="outside", cliponaxis=False,
            customdata=[lab.split("  ·  ")[1] for lab in labels][::-1],
            hovertemplate="%{y} (%{customdata}): %{x:+.1f} bps<extra></extra>"))
        tot = att["us_10y_change_bps"]
        fig.add_vline(x=tot, line=dict(color=INK, dash="dot", width=1.5))
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        lim = max(abs(v) for v in vals + [tot]) * 1.35 + 2
        fig = base_layout(fig, f"US 10Y {tot:+.0f} bps ({att['start']} → {att['end']}) — each pair sums to the "
                               "total (dotted line)", 340)
        fig.update_layout(showlegend=False, hovermode="closest", margin=dict(l=10, r=30, t=50, b=30))
        fig.update_xaxes(title="bps", range=[-lim, lim], zeroline=False)
        fig.update_yaxes(title="")
        st.plotly_chart(fig, theme="streamlit")
        st.caption("Pairs: blue = expectations vs term premium · orange = real yield vs breakeven · "
                   "green = 2Y vs 2s10s slope.")
        ctx_rows = [{"Indicator": k, "Change": f"{v['change']:+.1f} {v['unit']}", "Level now": f"{v['level']:.2f}"}
                    for k, v in att["context"].items()]
        st.dataframe(pd.DataFrame(ctx_rows), hide_index=True, width="stretch")
        st.caption("Heuristic reading of the numbers, not causal identification — confirm against the "
                   "news flow (auctions, Fed communication, oil/geopolitics, fiscal announcements).")

    st.markdown("### Fiscal fair value of the term premium")
    if not tp_fit:
        st.info("Debt/deficit/Fed balance-sheet data not in the dataset — rerun `python forecast.py` to fetch it.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Term premium now", fmt_pct(tp_fit["tp_now"]))
        m2.metric("Fiscal fair value", fmt_pct(tp_fit["fair_value_now"]))
        m3.metric("Gap", f"{tp_fit['residual_now_bps']:+.0f} bps", delta_color="off")
        m4.metric("R²", f"{tp_fit['r2']:.2f}", help=f"{tp_fit['n']} months, {tp_fit['sample']}")
        fitted = tp_fit["fitted"]
        actual = res.history["us_tp"].reindex(fitted.index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual.index, y=actual, name="Term premium (actual)",
                                 line=dict(color=INK, width=1.6), hovertemplate="%{y:.2f}%"))
        fig.add_trace(go.Scatter(x=fitted.index, y=fitted, name="Fiscal fair value (fitted)",
                                 line=dict(color=SLOTS[1], width=2), hovertemplate="%{y:.2f}%"))
        st.plotly_chart(base_layout(fig, "Term premium vs fiscal fair value", 340), theme="streamlit")
        coef_rows = [{"Driver": DRIVER_LABELS.get(c, c), "Latest": f"{tp_fit['latest_drivers'][c]:.1f}",
                      "TP sensitivity (bps per unit)": f"{tp_fit['coef'][c] * 100:+.2f}"}
                     for c in tp_fit["drivers"]]
        st.dataframe(pd.DataFrame(coef_rows), hide_index=True, width="stretch")
        st.caption("Descriptive regression (not causal): fiscal series trend slowly and QE distorted the 2010s, "
                   "which is why the Fed balance sheet is included as a control. Use it to frame the fiscal "
                   "channel; set your horizon driver values in the sidebar (Fiscal fair value mode) and "
                   "add named risk premia for Fed independence, geopolitics or energy.")

    # ---------------------------------------------------------------- macro themes (AI research + edit)
    st.markdown("### Macro themes for the report")
    st.caption("Claude researches the period with web search — every theme must cite a page it actually "
               "retrieved, unsourced themes are dropped. Review, edit or untick themes, and add your own; "
               "the final list feeds the PDF / Word commentary.")
    import os as _os

    from macro_research import (MODEL as R_MODEL, ResearchError, load_latest_research,
                                research_macro_themes, save_research)

    research_dir = ROOT / "output" / "forecast"
    if "research" not in st.session_state:
        st.session_state["research"] = load_latest_research(research_dir)
    key_ok = bool(_os.environ.get("ANTHROPIC_API_KEY") or base_cfg.get("anthropic_api_key"))
    att_q = attribute(df)
    rc1, rc2 = st.columns([1, 2])
    if rc1.button(f"🔎 Research {att_q['start']} – {att_q['end']} with Claude", disabled=not key_ok,
                  help=f"{R_MODEL} + web search; takes 1–3 minutes and uses API credits"):
        with st.spinner("Searching the news and drafting cited themes (1–3 min)…"):
            try:
                r = research_macro_themes(att_q, cfg, log_dir=research_dir / "logs")
                save_research(r, research_dir)
                st.session_state["research"] = r
                for k in [k for k in st.session_state if str(k).startswith(("th_", "inc_"))]:
                    del st.session_state[k]
            except ResearchError as exc:
                st.error(f"Research failed: {exc}")
                if exc.diagnostics:
                    with st.expander("Diagnostics", expanded=True):
                        st.json(exc.diagnostics)
                if exc.log_path:
                    st.caption(f"Full log (search queries, raw tool results): `{exc.log_path}`")
    if not key_ok:
        rc2.caption("Add ANTHROPIC_API_KEY to `.env` to enable research. You can still type themes below.")

    final_themes: list[str] = []
    r = st.session_state.get("research")
    if r:
        rc2.caption(f"Research for **{r.get('period')}** · generated {r.get('generated')} · {r.get('model')} · "
                    f"{len(r.get('sources', []))} sources · {r.get('dropped_unsourced', 0)} unsourced theme(s) dropped")
        st.info(r.get("quarter_summary", ""))
        if r.get("diagnostics"):
            d = r["diagnostics"]
            st.caption(f"{len(d.get('searches', []))} searches · {d.get('results_returned', 0)} results · "
                       f"{d.get('text_citations', 0)} inline citations · log: `{r.get('log_path')}`")
        for i, t in enumerate(r.get("themes", [])):
            with st.container(border=True):
                cc1, cc2 = st.columns([0.07, 0.93])
                inc = cc1.checkbox("Use", value=True, key=f"inc_{i}", label_visibility="collapsed")
                txt = cc2.text_area(f"{t['driver'].replace('_', ' ')} · {t['direction'].replace('_', ' ')}",
                                    t["theme"], key=f"th_{i}", height=68)
                cc2.caption("Evidence: " + t.get("evidence", ""))
                cc2.markdown(" · ".join(f"[{s_['title'][:70]}]({s_['url']})" for s_ in t.get("sources", [])))
                if inc and txt.strip():
                    final_themes.append(txt.strip())
        with st.expander("All sources and research notes"):
            for s_ in r.get("sources", []):
                st.markdown(f"[{s_['id']}] [{s_['title']}]({s_['url']})")
            st.text(r.get("notes", ""))
    manual = st.text_area("Your own themes (one per line)",
                          "\n".join(base_cfg.get("macro_themes", []) if not r else []), height=120,
                          key="manual_themes")
    final_themes += [ln.strip() for ln in manual.splitlines() if ln.strip()]
    cfg["macro_themes"] = final_themes
    st.caption(f"**{len(final_themes)} theme(s)** will be used in the reports.")

with tabs[6]:
    import os

    from brief_report import MODEL, generate_brief

    st.markdown("A concise **1–2 page PDF brief** of the current settings: headline call, KPI table, "
                "fan charts, quarterly table, US decomposition, policy scenarios, Canada view, risks "
                "and conclusion.")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or base_cfg.get("anthropic_api_key"))
    use_ai = st.toggle(f"Draft the commentary with Claude (`{MODEL}`)", value=has_key, disabled=not has_key,
                       help="Needs ANTHROPIC_API_KEY in .env. Off = factual template text from the numbers.")
    if not has_key:
        st.caption("No ANTHROPIC_API_KEY found in `.env` — the PDF will use template commentary.")
    include_bt = "bt" in st.session_state
    st.caption("Backtest results will be included." if include_bt
               else "Tip: run the Backtest tab first to include its accuracy summary.")
    if st.button("Generate PDF report", type="primary"):
        with st.spinner("Drafting commentary and building the PDF…" if use_ai else "Building the PDF…"):
            pdf, commentary = generate_brief(res, cfg, st.session_state.get("bt"), use_ai=use_ai)
        st.session_state["pdf"] = pdf
        st.session_state["pdf_commentary"] = commentary
    if "pdf" in st.session_state:
        slug = str(cfg.get("quarter", "")).replace(" ", "_")
        st.download_button("⬇ Download PDF", st.session_state["pdf"], f"10Y_Outlook_{slug}.pdf",
                           "application/pdf")
        com = st.session_state["pdf_commentary"]
        st.caption(com.get("_source", ""))
        with st.expander("Preview commentary", expanded=True):
            st.markdown(f"**{com['headline']}**\n\n{com['summary']}")
            st.markdown("**Conclusion:** " + com["conclusion"])
with tabs[7]:
    st.markdown("Copy this into `config/quarterly_config.yaml` (replace the `macro_themes:` list and the "
                "`forecast:` section) to make these settings the defaults for `forecast.py` and the Word report.")
    out = {k: v for k, v in f.items() if k not in ("enabled",)}
    text = yaml.safe_dump({"macro_themes": cfg.get("macro_themes", []),
                           "forecast": {"enabled": fbase.get("enabled", True), **out}},
                          sort_keys=False, allow_unicode=True)
    st.code(text, language="yaml")
    st.download_button("Download forecast settings (YAML)", text.encode(), "forecast_settings.yaml", "text/yaml")

