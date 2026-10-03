"""
Orchestrates the 2-year forecast.

For each scenario (user scenarios from config + optional endogenous Taylor-rule scenario):
  fed_funds path ─┬─> expectations component ─┐
                  │   term premium path ───────┼─> US 10Y (+ basis + user overlay)
  boc_rate path ──┴─> policy differential ─> Canada–US spread ─> Canada 10Y (+ overlay)

Scenarios are probability-weighted into a central path; a block bootstrap around the scenario
mixture gives the fan (10/25/50/75/90 percentiles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import canada, expectations, policy_path, term_premium, uncertainty
from .utils import anchored_path, horizon_index, overlay_path

logger = logging.getLogger(__name__)

SCENARIO_COLUMNS = [
    "fed_funds", "boc_rate", "policy_diff",
    "us_expectations", "us_basis", "us_tp", "us_overlay", "us_10y",
    "ca_spread", "ca_overlay", "canada_10y",
]


@dataclass
class ForecastResult:
    as_of: pd.Timestamp
    index: pd.DatetimeIndex                     # month-ends, h = 0..H
    scenarios: dict[str, pd.DataFrame]          # per-scenario component paths
    probabilities: dict[str, float]
    central: pd.DataFrame                       # probability-weighted components
    fan: pd.DataFrame                           # quantiles for us_10y and canada_10y
    quarterly: pd.DataFrame                     # quarterly averages (h >= 1)
    params: dict
    history: pd.DataFrame = field(repr=False)
    backtest: dict | None = None

    def summary(self) -> dict:
        """Compact JSON-friendly summary for commentary / report."""
        q = self.quarterly.round(2)
        return {
            "as_of": str(self.as_of.date()),
            "current_us_10y": round(float(self.history["us_10y"].iloc[-1]), 2),
            "current_canada_10y": round(float(self.history["canada_10y"].iloc[-1]), 2),
            "probabilities": self.probabilities,
            "quarterly": {str(k): v for k, v in q.to_dict(orient="index").items()},
            "scenario_end_us_10y": {n: round(float(s["us_10y"].iloc[-1]), 2) for n, s in self.scenarios.items()},
            "scenario_end_canada_10y": {n: round(float(s["canada_10y"].iloc[-1]), 2) for n, s in self.scenarios.items()},
            "end_components_central": self.central.iloc[-1][
                ["fed_funds", "boc_rate", "us_expectations", "us_basis", "us_tp", "us_overlay", "ca_spread"]
            ].round(2).to_dict(),
            "params": self.params,
        }


def _neutral_rates(df: pd.DataFrame, cfg: dict, config: dict) -> tuple[float, float]:
    """Nominal neutral policy rates for Fed and BoC (anchor for expectations beyond the horizon)."""
    mode = cfg.get("neutral_mode", "config")
    if mode == "trailing":  # backtest: market-based anchor known at the time
        years = int(cfg.get("trailing_neutral_years", 10))
        recent = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
        fed_n = float(recent["us_rny"].mean()) if "us_rny" in recent else float(recent["us_10y"].mean())
        boc_n = fed_n + float(recent["policy_diff"].mean())
        return fed_n, boc_n
    fed_n = cfg.get("neutral_fed_funds") or config.get("long_run_fed_funds", 3.0)
    boc_n = cfg.get("neutral_boc_rate") or config.get("long_run_boc_rate", 2.75)
    return float(fed_n), float(boc_n)


def _normalise_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("Scenario probabilities must sum to a positive number")
    if abs(total - 1.0) > 1e-6:
        logger.warning("Scenario probabilities sum to %.3f; rescaling to 1", total)
    return {k: v / total for k, v in probs.items()}


def project_scenarios(df: pd.DataFrame, config: dict, fcfg: dict | None = None) -> tuple[
        dict[str, pd.DataFrame], dict[str, float], dict, pd.DatetimeIndex]:
    """Deterministic component paths for each scenario. `fcfg` defaults to config['forecast']."""
    fcfg = fcfg if fcfg is not None else config.get("forecast", {})
    H = int(fcfg.get("horizon_months", 24))
    idx = horizon_index(df.index[-1], H)
    last = df.iloc[-1]

    fed_n, boc_n = _neutral_rates(df, fcfg, config)
    exp_cfg = fcfg.get("expectations", {})
    conv_hl = float(exp_cfg.get("convergence_halflife_months", 36))
    basis_hl = float(exp_cfg.get("basis_halflife_months", 12))

    tp_cfg = dict(fcfg.get("term_premium", {}))
    tp_base, tp_info = term_premium.tp_path(df, H, tp_cfg)

    sp_cfg = fcfg.get("canada_spread", {})
    try:
        sp_params = canada.fit_spread_model(df, sp_cfg.get("sample_start", "2000-01-01"))
    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
        logger.debug("Canada spread model unavailable: %s", exc)
        sp_params = None

    ov = fcfg.get("overlay", {}) or {}
    us_ov = overlay_path(ov.get("us_10y_bps"), idx)
    ca_ov = overlay_path(ov.get("canada_10y_bps"), idx)

    # ---- policy paths per scenario ---------------------------------------------------
    taylor_cfg = dict(fcfg.get("taylor_rule", {}))
    if fcfg.get("neutral_mode") == "trailing":
        taylor_cfg["r_star"] = fed_n - taylor_cfg.get("inflation_target", 2.0)
    fed_model, taylor_used = policy_path.taylor_path(df, idx, taylor_cfg)
    boc_model = policy_path.boc_model_path(
        df, fed_model, idx, boc_n, taylor_used["rho"],
        float(fcfg.get("boc_fed_passthrough", 0.5)))

    paths: dict[str, dict] = {}
    probs: dict[str, float] = {}
    for sc in fcfg.get("scenarios", []) or []:
        name = sc["name"]
        fed = anchored_path(sc.get("fed_funds"), float(last["fed_funds"]), idx)
        boc = anchored_path(sc.get("boc_rate"), float(last["boc_rate"]), idx) if sc.get("boc_rate") \
            else policy_path.boc_model_path(df, fed, idx, boc_n, taylor_used["rho"],
                                            float(fcfg.get("boc_fed_passthrough", 0.5)))
        paths[name] = {"fed": fed, "boc": boc,
                       "tp_target": term_premium.scenario_target(df, tp_cfg, sc)}
        probs[name] = float(sc.get("probability", 1.0))

    ms = fcfg.get("model_scenario", {"include": True, "probability": 0.0})
    if ms.get("include", True) and (ms.get("probability", 0) > 0 or not paths):
        name = ms.get("name", "Model (Taylor rule)")
        paths[name] = {"fed": fed_model, "boc": boc_model, "tp_target": None}
        probs[name] = float(ms.get("probability", 1.0)) if paths.keys() - {name} else 1.0
    probs = _normalise_probs(probs)

    # ---- components ------------------------------------------------------------------
    rny0 = float(last["us_rny"]) if "us_rny" in df and pd.notna(last.get("us_rny")) else None
    h = np.arange(H + 1)
    out: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        exp = expectations.expected_avg_short_rate(p["fed"], fed_n, conv_hl)
        basis = expectations.basis_path(rny0, exp[0], H, basis_hl)
        if p["tp_target"] is not None:
            # Scenario-specific TP view: same speed of adjustment, different destination
            tp = p["tp_target"] + (tp_base[0] - p["tp_target"]) * tp_info["phi"] ** h
        else:
            tp = tp_base
        us = exp + basis + tp + us_ov
        if rny0 is None:  # no term-premium data: pin h=0 to the observed yield via the basis
            basis = basis + (float(last["us_10y"]) - us[0]) * expectations.decay(h, basis_hl)
            us = exp + basis + tp + us_ov
        pdiff = p["boc"] - p["fed"]
        spread = (canada.spread_path(df, pdiff, sp_params, sp_cfg) if sp_params
                  else np.full(H + 1, np.nan))
        ca = us + spread + ca_ov
        out[name] = pd.DataFrame({
            "fed_funds": p["fed"], "boc_rate": p["boc"], "policy_diff": pdiff,
            "us_expectations": exp, "us_basis": basis, "us_tp": tp, "us_overlay": us_ov,
            "us_10y": us, "ca_spread": spread, "ca_overlay": ca_ov, "canada_10y": ca,
        }, index=idx)[SCENARIO_COLUMNS]

    params = {
        "neutral_fed_funds": round(fed_n, 3), "neutral_boc_rate": round(boc_n, 3),
        "convergence_halflife_months": conv_hl, "basis_halflife_months": basis_hl,
        "taylor_rule": taylor_used, "term_premium": tp_info,
        "canada_spread": {k: round(v, 4) for k, v in (sp_params or {}).items()},
    }
    return out, probs, params, idx


def _quarterly_table(central: pd.DataFrame, fan: pd.DataFrame,
                     scenarios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    monthly = central[["fed_funds", "boc_rate", "us_tp", "us_10y", "ca_spread", "canada_10y"]].join(
        fan[["us_10y_p10", "us_10y_p90", "canada_10y_p10", "canada_10y_p90"]])
    for name, s in scenarios.items():
        monthly[f"us_10y [{name}]"] = s["us_10y"]
        monthly[f"canada_10y [{name}]"] = s["canada_10y"]
    monthly = monthly.iloc[1:]  # forecast months only
    q = monthly.groupby(monthly.index.to_period("Q")).mean()
    q.index = q.index.astype(str)
    q.index.name = "quarter"
    return q


def run_forecast(df: pd.DataFrame, config: dict, with_fan: bool = True) -> ForecastResult:
    fcfg = config.get("forecast", {})
    scenarios, probs, params, idx = project_scenarios(df, config, fcfg)

    central = sum(scenarios[n] * probs[n] for n in scenarios)

    if with_fan:
        sims = uncertainty.simulate_fan(
            df, {n: s[["us_10y", "canada_10y"]].to_numpy() for n, s in scenarios.items()},
            probs, fcfg.get("uncertainty"))
        fan = uncertainty.quantile_frame(sims["us_10y"], idx, "us_10y").join(
            uncertainty.quantile_frame(sims["canada_10y"], idx, "canada_10y"))
    else:
        fan = pd.DataFrame(index=idx)
        for c in ("us_10y", "canada_10y"):
            for q in uncertainty.QUANTILES:
                fan[f"{c}_p{int(q * 100):02d}"] = central[c]

    beta = canada.rolling_beta(df).dropna()
    params["beta_ca_us_36m"] = round(float(beta.iloc[-1]), 3) if len(beta) else None

    return ForecastResult(
        as_of=df.index[-1], index=idx, scenarios=scenarios, probabilities=probs,
        central=central, fan=fan, quarterly=_quarterly_table(central, fan, scenarios),
        params=params, history=df,
    )


def export_result(result: ForecastResult, out_dir) -> dict[str, str]:
    """Write monthly / quarterly / scenario tables to CSV and a combined Excel workbook."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monthly = result.central.join(result.fan)
    paths = {
        "monthly": out / "forecast_monthly.csv",
        "quarterly": out / "forecast_quarterly.csv",
    }
    monthly.to_csv(paths["monthly"], index_label="month")
    result.quarterly.to_csv(paths["quarterly"])
    scen = pd.concat(result.scenarios, names=["scenario", "month"])
    scen.to_csv(out / "forecast_scenarios.csv")
    paths["scenarios"] = out / "forecast_scenarios.csv"
    try:
        xlsx = out / "forecast.xlsx"
        with pd.ExcelWriter(xlsx) as xw:
            result.quarterly.round(3).to_excel(xw, sheet_name="quarterly")
            monthly.round(3).to_excel(xw, sheet_name="monthly_central")
            for name, s in result.scenarios.items():
                s.round(3).to_excel(xw, sheet_name=f"sc_{name}"[:31])
            if result.backtest:
                result.backtest["summary"].round(3).to_excel(xw, sheet_name="backtest")
        paths["excel"] = xlsx
    except ImportError:
        logger.warning("openpyxl not installed; skipping Excel export")
    return {k: str(v) for k, v in paths.items()}
