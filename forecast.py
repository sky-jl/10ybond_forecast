#!/usr/bin/env python3
"""
2-Year US / Canada 10Y Yield Forecast (monthly model, quarterly output)

Usage:
  python forecast.py --config config/quarterly_config.yaml            # fetch data, forecast
  python forecast.py --backtest                                       # + out-of-sample backtest
  python forecast.py --data-file output/forecast/monthly_dataset.csv  # reuse cached data
  python forecast.py --synthetic --backtest                           # offline demo (fake data)

Subjective views (policy paths, probabilities, term premium, overlays) live in the
`forecast:` section of the config file.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent / "src"))

from charts import generate_forecast_charts  # noqa: E402
from forecasting.pipeline import load_dataset, run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/quarterly_config.yaml")
    ap.add_argument("--output-dir", default="output/forecast")
    ap.add_argument("--data-file", help="Monthly dataset CSV (skips downloading)")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data (offline demo/testing)")
    ap.add_argument("--backtest", action="store_true", help="Run the out-of-sample backtest")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if not args.synthetic and not args.data_file:
        from main import load_config  # resolves FRED key from env / .env
        config = load_config(args.config)

    out = Path(args.output_dir)
    df = load_dataset(config, args.data_file, args.synthetic, cache_path=out / "monthly_dataset.csv")
    from data_fetcher import coverage_report
    pd.set_option("display.width", 160)
    print("\nData coverage (monthly):")
    print(coverage_report(df).to_string())
    print(f"Forecast origin (h = 0): {df.index[-1]:%Y-%m}\n")
    result = run_pipeline(df, config, out, backtest=args.backtest)
    charts = generate_forecast_charts(result, out / "charts")

    pd.set_option("display.width", 160)
    cols = ["fed_funds", "us_10y", "us_10y_p10", "us_10y_p90",
            "boc_rate", "canada_10y", "canada_10y_p10", "canada_10y_p90"]
    print("\nQuarterly forecast (quarterly averages, %):")
    print(result.quarterly[cols].round(2).to_string())
    print("\nScenario probabilities:", {k: round(v, 2) for k, v in result.probabilities.items()})
    if result.backtest:
        s = result.backtest["summary"]
        print("\nBacktest (RMSE in pp; rmse_vs_rw < 1 beats random walk):")
        print(s[["target", "h", "model", "n", "rmse", "rmse_vs_rw", "dm_pvalue_vs_rw"]].round(3).to_string(index=False))
    print("\nCharts:", *charts.values(), sep="\n  ")


if __name__ == "__main__":
    main()
