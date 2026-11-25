# Long-term Torn stock ML

Dedicated backend repository for long-horizon stock history, weekly aggregation,
multi-model forecasting and rolling-origin backtesting. It is intentionally
separate from the minute-based worker in
[Torn-Helper-ML](https://github.com/unnyanr11/Torn-Helper-ML).

The main application is [Torn-Helper](https://github.com/unnyanr11/Torn-Helper).
It reads published long-term history and forecasts through a server-side
Supabase Edge Function. This repository does not host a website and does not
write into the short-term stock shards or the main application database.

## Data design

- Universe: all 35 Torn stocks.
- Canonical interval: UTC calendar weeks, Monday 00:00 through the following Monday.
- Target historical coverage: five years where the upstream provider has it; actual per-stock coverage is recorded and may be shorter.
- Each weekly row stores OHLC, arithmetic average, weekly return, volatility when calculable, observation count, source timestamps, source name and completeness.
- The current incomplete week is refreshed as new observations arrive; only completed weeks are used for model origins and the main app's weekly chart history.
- Missing weeks are not fabricated, prices are not interpolated, and source observations retain their provenance.

The weekly importer is resumable at the stock level and upserts by `(stock_id, week_start)`. A successful request does not imply that the requested five years were all available; inspect `long_term.backfill_state` for earliest/latest coverage and status.

## Database ownership and security

The dedicated long-term Supabase project is the source of record for:
- `long_term.weekly_prices` — weekly aggregates;
- `long_term.backfill_state` — per-stock ingestion coverage and errors;
- `long_term.model_runs` — model versions, cutoffs and evaluation metadata;
- `long_term.forecasts` — origin, target, price, expected return and validation details.

The schema is private and has RLS enabled. Browser roles have no schema/table grants. Keep these permissions and policies intact. The main app's `long-term-stock-forecast` Edge Function connects server-side using `LONG_TERM_DATABASE_URL`; the connection string must never be exposed to Vite/browser code.

Provide `LONG_TERM_DATABASE_URL` as a GitHub Actions repository secret for this worker. Never commit connection strings, passwords, service-role keys or API credentials.

## Weekly history ingestion

`.github/workflows/weekly-history.yml` runs every Monday at 02:23 UTC and can also be started manually. The manual `years` input accepts 1–10 years and defaults to five.

```sh
python -m pip install -r requirements.txt
python -m long_term.ingest_weekly --years 5
```

To refresh one stock locally:

```sh
python -m long_term.ingest_weekly --years 5 --stock-id 1
```

The importer obtains weekly candles from Tornsy's weekly endpoint, validates OHLC values and duplicate timestamps, groups them into UTC calendar weeks and upserts the results. Provider coverage and rate limits determine the actual history available. A partial history is retained and reported rather than padded.

## Long-term prediction configuration

The daily prediction worker evaluates 13 lookback windows independently:

| Window | Days used |
|---|---:|
| 1 month | 30 |
| 2 months | 60 |
| 3 months | 90 |
| 6 months | 180 |
| 9 months | 270 |
| 1 year | 365 |
| 1 year 6 months | 548 |
| 2 years | 730 |
| 2 years 6 months | 913 |
| 3 years | 1,095 |
| 3 years 6 months | 1,278 |
| 4 years | 1,461 |
| 5 years | 1,826 |

Each eligible window is evaluated against seven requested horizons: 7, 15, 30, 60, 90, 180 and 365 days. This is up to 91 configurations per stock (3,185 across 35 stocks). Requested horizons are aligned to weekly observations using `ceil(days / 7)`; for example, 60 days maps to nine weekly steps. These are weekly-close projections, not daily price paths.

A window needs at least 26 complete weekly observations before the prediction worker considers it. The feature engine also requires sufficient lag/rolling history and supervised examples. Combinations that cannot support chronological validation and at least three non-overlapping holdout folds do not receive fabricated predictions. Five years of weekly history cannot necessarily support multiple independent one-year holdout folds after feature warm-up and horizon purging.

## Multi-model forecasting engine

The daily worker uses the feature-based model competition in `long_term/ml_forecaster.py`. Candidate estimators are Ridge regression, Random Forest, Extra Trees and Histogram Gradient Boosting. Features are calculated only from weekly observations available at the forecast origin: lagged log returns (1, 2, 3, 4, 8, 13 and 26 weeks), rolling return statistics, recent trend slope and distance from rolling price means.

For each stock/lookback/horizon:

1. Construct supervised samples using the requested weekly-aligned future log return.
2. Keep samples chronological; do not shuffle time-series rows.
3. Evaluate candidates on a strictly later validation segment. Validation trade decisions are spaced at least one full forecast horizon apart, so overlapping positions are not counted as separate trades.
4. Simulate long-only entries only when predicted return exceeds estimated round-trip costs (0.10% fee plus 0.05% slippage on each side; 0.30% total) and predicted net return must meet the 0.20% default minimum (at least 0.50% predicted gross return).
5. Reject candidates with fewer than three validation trades, non-positive compounded net return, profit factor at or below 1.0, or maximum drawdown above 20%.
6. Rank the remaining candidates by compounded net return divided by the larger of absolute maximum drawdown or 1%, then by profit factor and validation MAE.
7. Refit the selected candidate at non-overlapping holdout origins using only labels already known at each origin; record holdout prediction and trading metrics.
8. Save the selected model, validation and holdout metrics, fold count and data cutoff with the forecast.

A combination is marked `validated` only when holdout MAE improves on the baseline by at least 5%, at least three non-overlapping holdout folds are available, and the holdout trading simulation also has at least three trades, positive net compounded return, profit factor above 1.0 and maximum drawdown no greater than 20%. If no candidate passes the validation risk filters, no model is selected for that horizon. Otherwise it may be rejected or insufficient. A generated predicted price is not, by itself, evidence that a configuration passed validation. The current prediction worker stores the signal as `wait`; this pipeline does not issue Buy/Sell recommendations.

## Prediction and backtest workflows

`.github/workflows/daily-predictions.yml` is started manually and then maintains an approximately daily cycle using four chained waits of 5 hours 59 minutes 30 seconds before dispatching the next run. GitHub's six-hour job limit, queueing and dispatch latency make this best-effort rather than an exact 24-hour schedule. Each cycle runs the Python unit tests and generates forecasts that are due (a stock is normally eligible for a new model run every 24 hours). Start it from the Actions tab on `main` after configuring the secret.

`.github/workflows/backtesting.yml` runs all stocks, lookback windows and horizons, then re-dispatches another backtest after a two-hour timer. It is a research workflow, not a live prediction source. It stores rolling-origin metrics in `long_term.model_runs`.

## Local development and tests

Python 3.12 is used in GitHub Actions.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Run the workers with the database URL in the environment:

```sh
python -m long_term.ingest_weekly --years 5
python -m long_term.predict_daily
python -m long_term.backtest
```

The commands require `LONG_TERM_DATABASE_URL`. Do not paste credentials into source files or commit a local `.env` file.

## Safety and interpretation

- Forecasts are estimates, not financial advice or a guarantee of returns.
- Signals remain `wait`; no automated trades are placed.
- Validation is historical and may be unavailable for longer horizons due to limited independent weekly folds.
- Do not use future observations in feature construction, scaling, model selection or calibration.
- Preserve the exact data cutoff, model version and validation status for every forecast.
- Keep the weekly database isolated from short-term shards and the main app; do not weaken private-schema grants or RLS to simplify integration.
