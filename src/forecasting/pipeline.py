"""End-to-end forecast pipeline shared by forecast.py (standalone) and main.py (report)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .model import ForecastResult, export_result, run_forecast

logger = logging.getLogger(__name__)


def load_dataset(config: dict, data_file: str | None = None, synthetic: bool = False,
                 cache_path: Path | None = None) -> pd.DataFrame:
    from data_fetcher import build_monthly_dataset, load_monthly_dataset, save_monthly_dataset
    if synthetic:
        from .synthetic import make_synthetic_monthly
        logger.warning("Using SYNTHETIC data — results are for testing only")
        return make_synthetic_monthly()
    if data_file:
        logger.info("Loading monthly dataset from %s", data_file)
        return load_monthly_dataset(data_file)
    df = build_monthly_dataset(config)
    if cache_path:
        save_monthly_dataset(df, cache_path)
        logger.info("Cached monthly dataset to %s", cache_path)
    return df


def run_pipeline(df: pd.DataFrame, config: dict, out_dir: Path,
                 backtest: bool = True) -> ForecastResult:
    result = run_forecast(df, config)
    if backtest:
        logger.info("Running out-of-sample backtest...")
        result.backtest = run_backtest(df, config)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        result.backtest["errors"].to_csv(Path(out_dir) / "backtest_errors.csv", index=False)
        result.backtest["summary"].to_csv(Path(out_dir) / "backtest_summary.csv", index=False)
    paths = export_result(result, out_dir)
    for name, p in paths.items():
        logger.info("  Saved %s: %s", name, p)
    return result


def attach_to_metrics(result: ForecastResult, metrics: dict) -> None:
    """Store forecast outputs under private keys (excluded from the generic metrics JSON)."""
    from .attribution import attribute

    metrics["_forecast"] = result
    metrics["_forecast_summary"] = result.summary()
    metrics["_forecast_summary"]["last_quarter_attribution"] = attribute(result.history)
    if result.backtest:
        s = result.backtest["summary"]
        metrics["_backtest_summary_df"] = s
        metrics["_backtest_summary"] = (
            s[s["model"].isin(["model", "rw"])][["target", "h", "model", "rmse", "rmse_vs_rw"]]
            .round(3).to_dict(orient="records"))
