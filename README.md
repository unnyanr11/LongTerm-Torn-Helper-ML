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

Python requirements and runnable ingestion/backtest commands will be added
after the upstream source and destination project configuration are verified.
