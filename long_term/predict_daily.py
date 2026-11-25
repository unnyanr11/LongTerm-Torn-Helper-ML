"""Daily long-horizon forecasts from stored weekly prices.

This deliberately starts with a transparent log-linear trend baseline. It only
labels a model validated when chronological one-step validation beats a
last-close baseline by at least 5%; otherwise every published signal is wait.
"""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .db import connect
from .stock_universe import STOCKS

MODEL_NAME = "log_linear_trend"
MODEL_VERSION = "1.0.0"
MIN_POINTS = 52
LOOKBACK = 26
HORIZONS_WEEKS = (1, 4)
DUE_HOURS = 24


def _fit_predict(prices, horizon):
    values = np.asarray(prices[-LOOKBACK:], dtype=float)
    if len(values) < 8 or np.any(values <= 0):
        return float(prices[-1])
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, np.log(values), 1)
    prediction = math.exp(intercept + slope * (len(values) - 1 + horizon))
    return max(prediction, 0.000001)


def _validation(prices):
    # Expanding chronological origins; each prediction uses only prior closes.
    errors_model, errors_naive = [], []
    start = max(LOOKBACK, int(len(prices) * 0.70))
    for target_idx in range(start, len(prices)):
        history = prices[:target_idx]
        actual = float(prices[target_idx])
        if actual <= 0 or not history:
            continue
        predicted = _fit_predict(history, 1)
        naive = float(history[-1])
        errors_model.append(abs(predicted - actual))
        errors_naive.append(abs(naive - actual))
    if len(errors_model) < 8:
        return None
    model_mae = float(np.mean(errors_model))
    naive_mae = float(np.mean(errors_naive))
    improvement = (naive_mae - model_mae) / naive_mae if naive_mae > 0 else 0.0
    return {
        "validation_observations": len(errors_model),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "improvement_vs_naive": improvement,
        "minimum_required_improvement": 0.05,
    }


def run():
    now = datetime.now(timezone.utc)
    processed = skipped = 0

    with connect() as conn:
        for stock_id, symbol in STOCKS.items():
            # Serialize per-stock workers and avoid duplicate daily predictions.
            with conn.cursor() as cur:
                cur.execute("select pg_try_advisory_xact_lock(%s)", (910000 + stock_id,))
                if not cur.fetchone()[0]:
                    skipped += 1
                    continue

                cur.execute(
                    """select max(created_at) from long_term.model_runs
                       where stock_id = %s and model_name = %s""",
                    (stock_id, MODEL_NAME),
                )
                last_run = cur.fetchone()[0]
                if last_run and last_run > now - timedelta(hours=DUE_HOURS):
                    skipped += 1
                    continue

                cur.execute(
                    """select week_start, close_price
                       from long_term.weekly_prices
                       where stock_id = %s and is_complete = true
                         and close_price is not null and close_price > 0
                       order by week_start""",
                    (stock_id,),
                )
                rows = cur.fetchall()

                if len(rows) < MIN_POINTS:
                    print(f"{symbol}: skipped; only {len(rows)} complete weekly rows")
                    skipped += 1
                    continue

                weeks = [row[0] for row in rows]
                prices = [float(row[1]) for row in rows]
                data_through = weeks[-1]
                split_idx = max(2, int(len(rows) * 0.70))
                validation_start = weeks[split_idx]
                metrics = _validation(prices)
                validated = bool(
                    metrics
                    and metrics["improvement_vs_naive"] >= 0.05
                )
                status = "validated" if validated else "rejected"

                cur.execute(
                    """insert into long_term.model_runs
                       (stock_id, model_name, model_version, data_through,
                        training_start, validation_start, test_start, metrics, status)
                       values (%s,%s,%s,%s,%s,%s,null,%s::jsonb,%s)
                       returning run_id""",
                    (
                        stock_id, MODEL_NAME, MODEL_VERSION, data_through,
                        weeks[0], validation_start,
                        json.dumps(metrics or {"reason": "insufficient_validation_samples"}),
                        status,
                    ),
                )
                run_id = cur.fetchone()[0]
                origin_price = prices[-1]
                origin_week = data_through

                for horizon in HORIZONS_WEEKS:
                    predicted = _fit_predict(prices, horizon)
                    predicted_return = (predicted / origin_price - 1.0) * 100.0
                    # No directional trade signal until a model has passed validation.
                    signal = "wait"
                    validation = {
                        "model_validated": validated,
                        "validation_metrics": metrics,
                        "horizon_weeks": horizon,
                        "data_points": len(prices),
                        "data_through": data_through.isoformat(),
                    }
                    cur.execute(
                        """insert into long_term.forecasts
                           (stock_id, model_run_id, origin_week, target_week,
                            origin_price, predicted_price, predicted_return_pct,
                            signal, validation)
                           values (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                           on conflict (stock_id, model_run_id, origin_week, target_week)
                           do nothing""",
                        (
                            stock_id, run_id, origin_week,
                            origin_week + timedelta(weeks=horizon),
                            origin_price, predicted, predicted_return,
                            signal, json.dumps(validation),
                        ),
                    )

                processed += 1
                print(
                    f"{symbol}: {status}; {len(prices)} weeks; "
                    f"1w={_fit_predict(prices, 1):.4f}; "
                    f"4w={_fit_predict(prices, 4):.4f}"
                )

        conn.commit()

    print(f"Daily prediction cycle complete: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    run()
