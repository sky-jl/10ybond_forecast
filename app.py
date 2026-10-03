"""
Interactive 10Y forecast lab (Streamlit).

    streamlit run app.py

Adjust the subjective inputs in the sidebar / editors and the 2-year US and Canada 10Y
forecast is recomputed with the same model as forecast.py. Uses the monthly dataset cached
by `python forecast.py` (output/forecast/monthly_dataset.csv).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_fetcher import load_monthly_dataset  # noqa: E402
from forecasting import run_forecast  # noqa: E402
from forecasting.backtest import run_backtest  # noqa: E402

DEFAULT_CONFIG = ROOT / "config" / "quarterly_config.yaml"
DEFAULT_DATA = ROOT / "output" / "forecast" / "monthly_dataset.csv"

# Categorical slots (fixed order; colour follows the scenario, never its rank)
SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

st.set_page_config(page_title="10Y Forecast Lab", page_icon="📈", layout="wide")


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

if st.sidebar.button("↺ Reset all inputs to config", use_container_width=True):
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

with sb.expander("Term premium", expanded=True):
    tp_cfg = fbase.get("term_premium", {})
    tp_default = tp_cfg.get("target", "historical")
    modes = ["Your view (number)", "Historical mean", "Regression on drivers"]
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
        meta, key="meta", use_container_width=True,
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
    fed_tbl = st.data_editor(fed_in, key="fed", num_rows="dynamic", use_container_width=True,
                             column_config=rate_columns(fed_in))
    st.caption(f"Latest: **{df['fed_funds'].iloc[-1]:.2f}%** · periods like 2027Q2 or 2027-06")
with c3:
    st.markdown("**BoC overnight path (% at period end)**")
    boc_in = anchors_to_frame(scen_cfg, "boc_rate").astype(float)
    boc_tbl = st.data_editor(boc_in, key="boc", num_rows="dynamic", use_container_width=True,
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


tabs = st.tabs(["US 10Y", "Canada 10Y", "Decomposition", "Policy paths", "Tables", "Backtest", "Export config"])
with tabs[0]:
    st.plotly_chart(fan_chart("us_10y", "US 10Y Treasury"), use_container_width=True, theme="streamlit")
with tabs[1]:
    st.plotly_chart(fan_chart("canada_10y", "Canada 10Y GoC"), use_container_width=True, theme="streamlit")
    beta = res.params.get("beta_ca_us_36m")
    st.caption(f"Canada–US spread at {end_label}: **{res.central['ca_spread'].iloc[-1]:+.2f} pp** "
               f"(now {res.central['ca_spread'].iloc[0]:+.2f}) · 36-month US→Canada yield beta: "
               f"**{beta:.2f}**" if beta is not None else "")
with tabs[2]:
    st.plotly_chart(decomposition_chart(), use_container_width=True, theme="streamlit")
    e = res.central.iloc[[0, -1]][["us_expectations", "us_basis", "us_tp", "us_overlay", "us_10y"]].T
    e.columns = ["Now", end_label]
    e["Change (bps)"] = (e[end_label] - e["Now"]) * 100
    st.dataframe(e.rename(index={"us_expectations": "Expected avg short rate", "us_basis": "Market-vs-neutral gap",
                                 "us_tp": "Term premium", "us_overlay": "Overlay", "us_10y": "US 10Y"})
                 .style.format({"Now": "{:.2f}", end_label: "{:.2f}", "Change (bps)": "{:+.0f}"}),
                 use_container_width=True)
with tabs[3]:
    a, b = st.columns(2)
    a.plotly_chart(policy_chart("fed_funds", "Fed funds rate"), use_container_width=True, theme="streamlit")
    b.plotly_chart(policy_chart("boc_rate", "BoC overnight rate"), use_container_width=True, theme="streamlit")
with tabs[4]:
    q = res.quarterly
    show = ["fed_funds", "us_10y", "us_10y_p10", "us_10y_p90", "boc_rate", "canada_10y",
            "canada_10y_p10", "canada_10y_p90", "us_tp", "ca_spread"]
    st.markdown("**Quarterly averages (%)**")
    st.dataframe(q[show].style.format("{:.2f}"), use_container_width=True)
    st.download_button("Download quarterly CSV", q.to_csv().encode(), "forecast_quarterly.csv", "text/csv")
    st.markdown("**By scenario (quarterly averages, %)**")
    st.dataframe(q[[c for c in q.columns if "[" in c]].style.format("{:.2f}"), use_container_width=True)
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
                        .update_yaxes(title="RMSE / RW RMSE"), use_container_width=True, theme="streamlit")
        st.dataframe(sub.drop(columns="target").style.format(
            {"rmse": "{:.3f}", "mae": "{:.3f}", "bias": "{:+.3f}", "rmse_vs_rw": "{:.3f}",
             "hit_rate": "{:.2f}", "dm_pvalue_vs_rw": "{:.2f}"}), use_container_width=True, hide_index=True)
with tabs[6]:
    st.markdown("Copy this into the `forecast:` section of `config/quarterly_config.yaml` to make these "
                "settings the defaults for `forecast.py` and the Word report.")
    out = {k: v for k, v in f.items() if k not in ("enabled",)}
    text = yaml.safe_dump({"forecast": {"enabled": fbase.get("enabled", True), **out}},
                          sort_keys=False, allow_unicode=True)
    st.code(text, language="yaml")
    st.download_button("Download forecast settings (YAML)", text.encode(), "forecast_settings.yaml", "text/yaml")
