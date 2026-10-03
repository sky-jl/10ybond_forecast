# Bond Yield Forecast — US & Canada 10Y

Two tools share the same data layer:

| Entry point | Purpose |
|---|---|
| `main.py` | Quarterly Word reports (long-term fair value + **Section 6: 2-year forecast path**) |
| `forecast.py` | Standalone 2-year monthly forecast + backtest (CSV / Excel / charts) |

```bash
pip install -r requirements.txt
export FRED_API_KEY=...            # or put it in .env
python forecast.py --backtest      # forecast + out-of-sample backtest → output/forecast/
python main.py                     # full Word reports (needs ANTHROPIC_API_KEY for commentary)
python forecast.py --synthetic --backtest   # offline demo with FAKE data
pytest tests/
streamlit run app.py               # interactive web page: adjust views, see results live
```

### Interactive forecast lab (`app.py`)

`streamlit run app.py` opens a local web page (http://localhost:8501). Edit scenario
probabilities and Fed / BoC paths in tables, move sliders for the neutral rate, term premium,
Canada spread and overlays — the forecast, fan charts, decomposition and quarterly table update
instantly. The **Backtest** tab runs the out-of-sample test; **Export config** gives the YAML to
paste into `config/quarterly_config.yaml` so `forecast.py` / `main.py` use the same settings.
It reads the dataset cached by `python forecast.py` (run that first; or toggle synthetic demo data).

The **PDF report** tab (or `python forecast.py --pdf`) writes a concise 1–2 page brief: headline
call, KPI table, US/Canada fan charts, quarterly table, decomposition, policy scenarios, risks and
conclusion. Commentary is drafted by Claude (`claude-opus-5-5`) when `ANTHROPIC_API_KEY` is in
`.env`; otherwise a factual template is used. Always review before sharing.

## 2-year forecast model (`src/forecasting/`)

Monthly model, h = 1…24 months, reported as quarterly averages.

```
US 10Y     = expected average short rate (risk-neutral)  +  term premium  +  basis  +  overlay
Canada 10Y = US 10Y  +  Canada–US spread  +  overlay
```

| Component | Module | How it is projected |
|---|---|---|
| Policy rate paths (Fed, BoC) | `policy_path.py` | **User scenarios** (config) + endogenous inertial Taylor rule scenario |
| Expectations component | `expectations.py` | Average expected short rate over the next 120 months: scenario path, then convergence to the neutral rate |
| Basis | `expectations.py` | Today's gap between the observed risk-neutral yield (ACM / Kim-Wright) and the construct, decays away |
| Term premium | `term_premium.py` | AR(1) mean reversion to a target: user number, trailing mean, or regression on deficit / rate volatility |
| Canada–US spread | `canada.py` | `s_t = a + ρ s_{t-1} + b (BoC − Fed)_t`, driven by the scenario policy paths (or user target) |
| Uncertainty | `uncertainty.py` | Scenario mixture + joint block bootstrap of monthly US/Canada yield changes → 10/25/50/75/90 pct |
| Backtest | `backtest.py` | Real-time style, 2005→: model vs random walk, AR(1), forward rates, Diebold–Li; RMSE, DM test |

Why expectations + term premium (not real + breakeven) as the main split: over 2 years most 10Y
variation comes from the expected policy path, which has explicit, forecastable drivers. TIPS real
yields and breakevens each contain their own risk/liquidity premia and are not easier to forecast;
they remain useful as a consistency check.

## Your subjective inputs (`config/quarterly_config.yaml` → `forecast:`)

| Parameter | What it controls | Example |
|---|---|---|
| `scenarios[].fed_funds` / `boc_rate` | Policy-rate path, end-of-period levels, interpolated monthly | `{"2026Q4": 3.375, "2027Q4": 3.125}` |
| `scenarios[].probability` | Scenario weights (rescaled to 1) | `0.55` |
| `scenarios[].term_premium_target` | Scenario-specific TP view (e.g. fiscal stress) | `1.25` |
| `model_scenario.probability` | Weight on the objective Taylor-rule path (0 = reference only) | `0.10` |
| `neutral_fed_funds`, `neutral_boc_rate` | Long-run nominal neutral (defaults to `long_run_fed_funds` / `long_run_boc_rate`) | `3.125` |
| `expectations.convergence_halflife_months` | How fast rates converge to neutral beyond 2 years | `36` |
| `term_premium.target` | Number = your TP view; `"historical"`; `"regression"` | `0.75` |
| `term_premium.drivers` | Deficit / volatility assumptions for `"regression"` | `deficit_gdp: -6.5` |
| `canada_spread.target` | Number = your Canada–US spread view (pp); `"model"` | `-0.60` |
| `taylor_rule.inflation_path` / `unemployment_path` | Macro views feeding the model scenario | `{"2027Q2": 2.6}` |
| `overlay.us_10y_bps` / `canada_10y_bps` | Final judgmental add-on in bps, phased in from 0 | `{"2027Q2": 10}` |
| `uncertainty.scale` | Widen / narrow the fan | `1.2` |

Period keys accept `"2027Q2"`, `"2027-06"` or `"2027-06-30"`. Every path starts at the latest observed value.
The backtest deliberately ignores scenarios and overlays (judgment cannot be backtested); it measures the
objective model only.

## Data

FRED (Treasury curve, Fed funds, Kim-Wright TP `THREEFYTP10`, TIPS, breakevens, core PCE, unemployment,
CBO NAIRU, deficit), Bank of Canada VALET (GoC 2Y/10Y, overnight rate), NY Fed ACM (`ACMTP10`, `ACMRNY10`,
optional). The monthly dataset is cached to `output/forecast/monthly_dataset.csv`; reuse it with
`--data-file` / `--forecast-data-file`.

Note: the `acm_tp` column in the daily report dataset is the **Kim-Wright** term premium (`THREEFYTP10`);
labels in the report now say so. True ACM data is used by the forecast model when the download succeeds.
