#!/usr/bin/env python3
"""
Bond Yield Long-Term Forecast Generator
Usage: python main.py --config config/quarterly_config.yaml [--output-dir output/]
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ before config is read

# Add src/ to path so modules can import each other
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetcher import build_dataset
from analysis import run_analysis
from charts import generate_all_charts
from commentary import generate_all_commentary
from report_builder import build_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Allow API keys to come from environment variables as fallback
    if not cfg.get("fred_api_key") or cfg["fred_api_key"] == "YOUR_FRED_API_KEY_HERE":
        cfg["fred_api_key"] = os.environ.get("FRED_API_KEY", "")
    if not cfg.get("anthropic_api_key"):
        cfg["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")

    if not cfg["fred_api_key"]:
        logger.error(
            "FRED API key not set. Add 'fred_api_key' to the config or set FRED_API_KEY env var."
        )
        sys.exit(1)

    return cfg


def open_file(path: str) -> None:
    """Open the output file with the default OS application."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", path], check=False)
        elif sys.platform == "win32":
            os.startfile(path)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bond yield forecast report.")
    parser.add_argument("--config", default="config/quarterly_config.yaml",
                        help="Path to quarterly config YAML")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for the generated report and charts")
    parser.add_argument("--no-open", action="store_true",
                        help="Do not auto-open the report after generation")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    chart_dir = output_dir / "charts"
    quarter_slug = config["quarter"].replace(" ", "_")
    us_report_path = output_dir / f"US_Bond_Yield_Forecast_{quarter_slug}.docx"
    canada_report_path = output_dir / f"Canada_Bond_Yield_Forecast_{quarter_slug}.docx"

    logger.info("=== Bond Yield Forecast Generator — %s ===", config["quarter"])

    # Step 1 — Fetch data
    logger.info("Step 1/5  Fetching market data...")
    df = build_dataset(config)

    # Step 2 — Analyse
    logger.info("Step 2/5  Running analysis...")
    metrics = run_analysis(df, config)
    # Pull out df from metrics to keep it separate
    df_enriched = metrics.pop("_df")

    # Step 3 — Charts
    logger.info("Step 3/5  Generating charts...")
    chart_paths = generate_all_charts(df_enriched, metrics, chart_dir, config)
    for name, path in chart_paths.items():
        logger.info("  Saved: %s", path)

    # Step 4 — Commentary
    logger.info("Step 4/5  Drafting commentary with Claude API...")
    commentary = generate_all_commentary(metrics, config)

    # Step 5 — Build reports
    logger.info("Step 5/5  Assembling Word reports...")
    build_report(metrics, commentary["us"], chart_paths, config, us_report_path, country="us")
    build_report(metrics, commentary["canada"], chart_paths, config, canada_report_path, country="canada")

    logger.info("")
    logger.info("Done!  US report:     %s", us_report_path)
    logger.info("       Canada report: %s", canada_report_path)

    if not args.no_open:
        open_file(str(us_report_path))
        open_file(str(canada_report_path))


if __name__ == "__main__":
    main()
