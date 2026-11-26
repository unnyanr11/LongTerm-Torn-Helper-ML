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

## Trading thresholds and costs

The long-term default minimum expected net return is **0.2%** after the estimated round-trip cost. The game's 0.10% fee is charged only once, when selling. With the model's assumed 0.05% slippage on each side, the default estimated round-trip cost is 0.20%, making the default gross forecast hurdle 0.40%. There is no maximum expected-profit threshold. Configure the worker/backtest defaults with GitHub Actions repository Variables:

- `LONG_MIN_EXPECTED_NET_RETURN` (fraction; default `0.002` = 0.2%)
- `LONG_ROUND_TRIP_COST` (fraction; default `0.002` = 0.20%)

The same values are used by validation/backtesting and the daily signal gate. These are shared worker settings, not per-user preferences in the main app.

## Multi-model forecasting engine

The daily worker uses the feature-based model competition in `long_term/ml_forecaster.py`. Candidate estimators are Ridge regression, Random Forest, Extra Trees and Histogram Gradient Boosting. Features are calculated only from weekly observations available at the forecast origin: lagged log returns (1, 2, 3, 4, 8, 13 and 26 weeks), rolling return statistics, recent trend slope and distance from rolling price means.

For each stock/lookback/horizon:

1. Construct supervised samples using the requested weekly-aligned future log return.
2. Keep samples chronological; do not shuffle time-series rows.
3. Evaluate candidates on a strictly later chronological validation segment using prediction error only.
4. Rank candidates by validation MAE, then RMSE, then directional accuracy. No trading simulation or profitability criteria participate in model selection.
5. Refit the selected candidate at expanding walk-forward origins using only labels already known at each origin; evaluate on untouched, non-overlapping holdout observations.
8. Save the selected model, validation and holdout metrics, fold count and data cutoff with the forecast.

A combination is marked `validated` only when holdout MAE improves on the flat-price baseline by at least 5% and at least three non-overlapping holdout folds are available. This is an accuracy-only decision; fees, slippage, profit thresholds, trade counts, portfolio returns and drawdown are excluded from backtesting. If no candidate passes the validation risk filters, no model is selected for that horizon. Otherwise it may be rejected or insufficient. A generated predicted price is not, by itself, evidence that a configuration passed validation. The daily worker emits `buy`, `sell` or `wait` only when that exact horizon passes holdout validation and the predicted move clears the configured round-trip cost plus minimum net-return hurdle. It does not place trades.

## Automatic matured-outcome calibration

The daily prediction worker can also refresh matured-outcome calibration automatically, separately
for each stock, analysis lookback and forecast horizon. It joins forecasts to
their completed target-week close; incomplete weeks and forecasts whose target
has not matured are never used. Calibration fits an affine correction to the
predicted log return (scale and bias) on the older 70% of matured observations,
then evaluates it on the later chronological holdout.

A candidate needs at least 20 matured observations (at least 14 for fitting and
6 for holdout), must stay within conservative parameter bounds (scale 0.50–1.50
and absolute bias at most 0.10 log-return units), and must improve holdout MAE
by at least 2% versus the uncalibrated forecast. Only then is it activated.
The prior active version is retained if a candidate fails. A candidate is not
re-evaluated into a new version until additional matured outcomes arrive.
Calibration versions and acceptance metrics are stored in
`long_term.forecast_calibrations`; each forecast's validation JSON records the
active calibration and the latest candidate result. This is automatic; routine
calibration does not require an admin approval.

**One-time database setup:** run
[`migrations/20261126_forecast_calibrations.sql`](migrations/20261126_forecast_calibrations.sql)
against the dedicated long-term Supabase database before the next prediction
workflow run. It creates only the calibration history table and indexes; it
does not modify existing RLS policies or grants. Keep the schema private and
grant access only through the existing server-side database connection.

## Prediction and backtest workflows

`.github/workflows/daily-predictions.yml` is started manually and then maintains an approximately daily cycle using four chained waits of 5 hours 59 minutes 30 seconds before dispatching the next run. GitHub's six-hour job limit, queueing and dispatch latency make this best-effort rather than an exact 24-hour schedule. Each cycle runs the Python unit tests and generates forecasts that are due (a stock is normally eligible for a new model run every 24 hours). Start it from the Actions tab on `main` after configuring the secret.

`.github/workflows/backtesting.yml` runs all stocks, lookback windows and horizons, then re-dispatches another backtest after a two-hour timer. It is a backend research and calibration workflow, not a browser inference path. It stores rolling-origin accuracy metrics in `long_term.model_runs` and recalibrates each stock/lookback/horizon from matured actual target-week closes. Calibration deduplicates repeated daily forecasts for the same target week, fits on older observations, validates on a later holdout, and activates a new version only when holdout MAE improves by at least 2%.

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


## ML parity and portfolio research upgrade

The long-term worker retains its own stock-specific feature pipeline and weekly
forecast horizons while following the short-term worker's engineering pattern:
isolated jobs, deterministic model candidates, chronological model competition,
walk-forward evaluation, calibration, versioned run metrics, concurrency guards,
and automated test gates. It does not import Torn-specific short-term features
or write to the short-term stock shards.

Six long-term candidates (Ridge, ElasticNet, Random Forest, Extra Trees,
HistGradientBoosting and a VotingRegressor ensemble of all five) compete
independently for each stock/lookback/horizon.
Selection is made on earlier chronological validation data; the selected
candidate is refit on each expanding walk-forward origin before untouched
holdout evaluation. Daily prediction runs refit from the latest available
weekly history, and matured forecast calibration is promoted only after a
separate chronological holdout improvement.

### Portfolio-level research

Daily prediction cycles additionally calculate research-only long-only
allocations for each forecast horizon using the validated one-year-lookback
forecasts. The optimizer:
- excludes forecasts that have not passed their own holdout and trading gates;
- deducts the configured round-trip cost and minimum expected net-return hurdle;
- estimates covariance from aligned, complete weekly closes;
- uses capped mean-variance optimization with a 15% per-position cap;
- leaves unused capital in cash when too few eligible assets exist;
- stores allocations and risk metrics in `long_term.portfolio_allocations`.

These allocations are research outputs, not orders or a promise of returns.
They do not assume the user's holdings, available capital or brokerage
execution conditions.

Before enabling portfolio persistence, apply
`migrations/20261126_portfolio_allocations.sql` once to the dedicated
long-term Supabase database. The migration creates one additive private table,
enables RLS on that new table, revokes browser-role access, and does not alter
any existing RLS policy. If the migration has not yet been applied, ordinary
daily forecasts continue and the portfolio research step logs a skipped
message.

The default minimum expected net return remains 0.2%, configurable through
`LONG_MIN_EXPECTED_NET_RETURN`; there is no maximum expected-profit ceiling.


## Technical strategy research

The long-term rolling-origin backtest also evaluates nine weekly close-only
strategy families for every stock, lookback window and forecast horizon:
EMA trend, MACD trend, ROC momentum, dual momentum, Bollinger mean reversion,
RSI mean reversion, Donchian breakout, volatility breakout and Bollinger
squeeze breakout. Signals use only completed weekly closes available at the
decision point and execute at the next completed weekly close. Missing target
weeks or gaps in the execution path are skipped; no weekly prices are
interpolated or fabricated.

The research simulation applies the 0.20% round-trip cost, 0.20% minimum mean
net trade return, 0.5% equity risk budget, 15% position cap, close-only
volatility stop and 2R target. Eligibility requires at least five trades,
positive compounded return, profit factor above 1, maximum drawdown within
20%, and mean net trade return of at least 0.20%. Buy-and-hold and cash are
reported as explicit baselines. Strategy results are saved inside each
existing `long_term.model_runs.metrics` JSON document; no schema, grants or
RLS policies are changed. Strategies remain research-only and do not issue
orders or override validated forecast signals.
