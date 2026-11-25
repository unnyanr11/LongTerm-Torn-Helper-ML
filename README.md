# Long-term Torn stock ML

Dedicated backend repository for long-horizon stock history, weekly aggregation,
research, walk-forward backtesting and forecasts. This project is intentionally
separate from the live/minute prediction worker in `Torn-Helper-ML`.

## Data design

- Target historical coverage: five years, where the upstream source actually
  provides it.
- Canonical interval: UTC calendar weeks (Monday 00:00 through next Monday).
- Expected history: approximately 260-261 weekly records per stock.
- Universe: 35 Torn stocks.
- Never fabricate missing weeks or interpolate observed prices.
- Keep raw source provenance and completeness metadata with every aggregate.
- Recompute the current incomplete week as new observations arrive; finalized
  weeks are idempotently upserted from source observations.

Each weekly row stores open, high, low, close, arithmetic average, weekly
percentage return, realized volatility, observation count, first/last source
timestamps, source name, and completeness status.

## Database ownership

The configured dedicated long-term Supabase project is the source of record
for weekly aggregates, model runs, walk-forward predictions, evaluation metrics,
and long-term forecasts. Do not write these records into the live stock shards
or the main application database.

Credentials must be provided through GitHub Actions secrets. Never commit
connection strings, service-role keys, or passwords. The project reference and
secret name are deployment configuration, not source code.

## Historical backfill

The five-year target is not a claim that five years are already available.
Before bulk ingestion, verify the upstream endpoint's actual earliest timestamp,
pagination behavior, stock coverage, and rate limits. Record coverage per stock.
A backfill must be resumable and idempotent. If the source only provides a
shorter history for a stock, preserve that shorter coverage and report it.

## Long-term prediction configuration

The daily prediction worker evaluates 13 rolling historical windows independently:
1, 2, 3, 6, and 9 months; 1 year; 1 year 6 months; 2 years; 2 years 6 months;
3 years; 3 years 6 months; 4 years; and 5 years.

For each eligible historical window it creates seven projections: 1 week,
15 days, 1 month, 2 months, 3 months, 6 months, and 1 year. This is up to
91 range/horizon combinations per stock (3,185 across 35 stocks). Windows
with fewer than eight complete weekly observations are marked insufficient
and do not receive fabricated forecasts.

Calendar windows are approximated as day counts for filtering weekly records.
Forecast target dates use the requested day horizon. Because the source data
is weekly, 15-day and month-based outputs are model projections from weekly
closes, not observed daily-price forecasts. Signals remain `wait` until
horizon-specific out-of-sample validation and calibration are implemented.
The current validation compares one-week walk-forward errors with a
last-observed-close baseline; it is not proof of profitability.


## Multi-model machine-learning engine

The long-term prediction worker now uses a feature-based Python model competition
instead of relying on a single log-linear trend equation. Candidate estimators
are Ridge regression, Random Forest, Extra Trees, and histogram gradient
boosting. Features are calculated only from weekly observations available at
the forecast origin: lagged log returns, rolling return statistics, recent
trend slope, and price distance from rolling means.

For every stock, lookback window, and horizon:

1. Build supervised samples whose targets are future log returns at the requested
   weekly-aligned horizon.
2. Keep the samples in chronological order; never shuffle time-series rows.
3. Select the candidate with the lowest error on a later validation segment.
4. Refit the selected estimator at each non-overlapping holdout origin using only
   labels that would already have been known at that origin.
5. Compare holdout price-ratio MAE with a zero-return / last-close baseline.
6. Store the selected model, candidate validation metrics, holdout metrics,
   fold count, and forecast cutoff with each forecast.

A model is marked validated only when it beats the baseline MAE by at least 5%
on at least three non-overlapping holdout folds. Insufficient combinations are
reported as insufficient rather than being promoted. A five-year weekly series
cannot support three independent one-year holdout folds after feature warm-up
and model selection; the one-year horizon may therefore remain unvalidated.
This is an honest data-coverage limitation, not a reason to fabricate folds.

The candidate models are deliberately kept within scikit-learn so the GitHub
Actions environment remains reproducible. Model complexity is not evidence of
profitability. Signals remain `wait`; this pipeline is research and forecast
evaluation, not a trading recommendation or a guarantee of returns.

## Planned stages

1. Verify the upstream all-time/weekly endpoint and coverage for all 35 stocks.
2. Provision the private long-term schema in the dedicated Supabase project.
3. Backfill available history with resumable checkpoints.
4. Maintain weekly aggregates independently of the live 180-day retention.
5. Train and evaluate long-horizon models using chronological walk-forward
   splits, horizon purging, and baselines.
6. Publish forecasts with explicit coverage, freshness, and validation status.
7. Add authenticated read-only integration to the main application only after
   the data pipeline has been verified.

## Safety and evaluation

A forecast must not become Buy/Sell solely because a model produced a number.
Require sufficient complete history and out-of-sample improvement over simple
baselines. Keep training, validation and final test periods chronological.
Do not use future data in feature construction, scaling, model selection or
calibration. Store the exact data cutoff and model version for every forecast.

## Local development

Install dependencies with `pip install -r requirements.txt`. Run the history
importer with `python -m long_term.ingest_weekly --years 5` and the daily
forecast worker with `python -m long_term.predict_daily`. Both require
`LONG_TERM_DATABASE_URL` in the environment.
