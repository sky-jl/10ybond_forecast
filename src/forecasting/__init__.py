"""
2-year (monthly, h = 1..24) forecast model for the US and Canada 10Y government bond yields.

US 10Y   = expected average short rate (risk-neutral yield) + term premium + basis + overlay
Canada 10Y = US 10Y + Canada–US spread (driven by the BoC–Fed policy differential) + overlay
"""

from .model import ForecastResult, run_forecast  # noqa: F401
