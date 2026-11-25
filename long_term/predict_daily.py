"""Daily multi-window, multi-horizon long-term stock forecasts.

Uses weekly closes and a transparent log-linear trend baseline. Every
lookback/horizon combination is evaluated independently. Sparse windows are
recorded as insufficient rather than padded or fabricated.
"""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np

from .db import connect
from .stock_universe import STOCKS

MODEL_NAME = "multi_window_log_linear"
MODEL_VERSION = "2.0.0"
DUE_HOURS = 24
MIN_TRAIN_POINTS = 8
MIN_VALIDATION_POINTS = 4

# Approximate calendar-month lengths in days; weekly data remains the source.
ANALYSIS_RANGES = (
    ("1_month", 30),
    ("2_months", 60),
    ("3_months", 90),
    ("6_months", 180),
    ("9_months", 270),
    ("1_year", 365),
    ("1_year_6_months", 548),
    ("2_years", 730),
    ("2_years_6_months", 913),
    ("3_years", 1095),
    ("3_years_6_months", 1278),
    ("4_years", 1461),
    ("5_years", 1826),
)

FORECAST_HORIZONS = (
    ("1_week", 7),
    ("15_days", 15),
    ("1_month", 30),
    ("2_months", 60),
    ("3_months", 90),
    ("6_months", 180),
    ("1_year", 365),
)


def _fit_predict(prices, horizon_days):
    values = np.asarray(prices, dtype=float)
    if len(values) < MIN_TRAIN_POINTS or np.any(values <= 0):
        raise ValueError("Insufficient or invalid prices for trend fit")
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, np.log(values), 1)
    weeks_forward = horizon_days / 7.0
    prediction = math.exp(intercept + slope * (len(values) - 1 + weeks_forward))
    return max(prediction, 0.000001)


def _validate(prices):
    """Walk-forward one-week validation, never training on the target week."""
    errors_model, errors_naive = [], []
    first_target = max(MIN_TRAIN_POINTS, int(len(prices) * 0.70))
    for target_idx in range(first_target, len(prices)):
        history = prices[:target_idx]
        actual = float(prices[target_idx])
        if len(history) < MIN_TRAIN_POINTS or actual <= 0:
            continue
        try:
            predicted = _fit_predict(history, 7)
        except (ValueError, OverflowError):
            continue
        errors_model.append(abs(predicted - actual))
        errors_naive.append(abs(float(history[-1]) - actual))

    if len(errors_model) < MIN_VALIDATION_POINTS:
        return {
            "status": "insufficient_validation",
            "validation_observations": len(errors_model),
        }

    model_mae = float(np.mean(errors_model))
    naive_mae = float(np.mean(errors_naive))
    improvement = (naive_mae - model_mae) / naive_mae if naive_mae > 0 else 0.0
    return {
        "status": "validated" if improvement >= 0.05 else "rejected",
        "validation_observations": len(errors_model),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "improvement_vs_naive": improvement,
        "minimum_required_improvement": 0.05,
    }


def _range_rows(rows, cutoff, days):
    start = cutoff - timedelta(days=days)
    return [(week, price) for week, price in rows if week >= start and week <= cutoff]


def run():
    now = datetime.now(timezone.utc)
    processed = skipped = forecasts_written = 0

    with connect() as conn:
        for stock_id, symbol in STOCKS.items():
            with conn.cursor() as cur:
                # Transaction-scoped lock prevents concurrent workers for a stock.
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
                if len(rows) < MIN_TRAIN_POINTS:
                    print(f"{symbol}: skipped; only {len(rows)} complete weekly observations")
                    skipped += 1
                    continue

                origin_week = rows[-1][0]
                origin_price = float(rows[-1][1])
                range_results = {}
                run_validated = False

                for range_name, range_days in ANALYSIS_RANGES:
                    selected = _range_rows(rows, origin_week, range_days)
                    prices = [float(price) for _, price in selected]
                    result = {
                        "analysis_range": range_name,
                        "analysis_range_days": range_days,
                        "observations": len(prices),
                        "first_observation": selected[0][0].isoformat() if selected else None,
                        "status": "insufficient_data",
                    }

                    if len(prices) >= MIN_TRAIN_POINTS:
                        validation = _validate(prices)
                        result["validation"] = validation
                        validated = validation.get("status") == "validated"
                        run_validated = run_validated or validated
                        result["status"] = validation.get("status", "rejected")

                        for horizon_name, horizon_days in FORECAST_HORIZONS:
                            try:
                                predicted = _fit_predict(prices, horizon_days)
                            except (ValueError, OverflowError):
                                continue
                            predicted_return = (predicted / origin_price - 1.0) * 100.0
                            target_date = origin_week + timedelta(days=horizon_days)
                            # Signals remain conservative until horizon-specific
                            # out-of-sample validation and calibration are added.
                            signal = "wait"
                            forecast_validation = {
                                "analysis_range": range_name,
                                "analysis_range_days": range_days,
                                "forecast_horizon": horizon_name,
                                "forecast_horizon_days": horizon_days,
                                "model_validated_one_week_baseline": validated,
                                "range_validation": validation,
                                "observations": len(prices),
                                "data_through": origin_week.isoformat(),
                                "note": "Weekly-close trend projection; not daily observed data.",
                            }
                            result.setdefault("forecasts", []).append(horizon_name)
                            pending_forecasts = result.setdefault("_pending", [])
                            pending_forecasts.append((
                                range_days, horizon_days, target_date,
                                predicted, predicted_return, signal,
                                json.dumps(forecast_validation),
                            ))
                    range_results[range_name] = result

                overall_status = "validated" if run_validated else "rejected"
                all_metrics = {
                    "analysis_ranges": {
                        name: {k: v for k, v in result.items() if k != "_pending"}
                        for name, result in range_results.items()
                    },
                    "forecast_horizons_days": {
                        name: days for name, days in FORECAST_HORIZONS
                    },
                    "minimum_training_observations": MIN_TRAIN_POINTS,
                    "validation_rule": "one-week walk-forward MAE must beat last-close baseline by at least 5%",
                }

                cur.execute(
                    """insert into long_term.model_runs
                       (stock_id, model_name, model_version, data_through,
                        training_start, validation_start, test_start, metrics, status)
                       values (%s,%s,%s,%s,%s,%s,null,%s::jsonb,%s)
                       returning run_id""",
                    (
                        stock_id, MODEL_NAME, MODEL_VERSION, origin_week,
                        rows[0][0], rows[max(1, int(len(rows) * 0.70))][0],
                        json.dumps(all_metrics), overall_status,
                    ),
                )
                run_id = cur.fetchone()[0]

                for result in range_results.values():
                    for item in result.get("_pending", []):
                        range_days, horizon_days, target_date, predicted, predicted_return, signal, validation_json = item
                        cur.execute(
                            """insert into long_term.forecasts
                               (stock_id, model_run_id, origin_week, target_week,
                                origin_price, predicted_price, predicted_return_pct,
                                signal, validation, analysis_range_days, horizon_days)
                               values (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s)
                               on conflict (stock_id, model_run_id, analysis_range_days, horizon_days)
                               do nothing""",
                            (
                                stock_id, run_id, origin_week, target_date,
                                origin_price, predicted, predicted_return,
                                signal, validation_json, range_days, horizon_days,
                            ),
                        )
                        forecasts_written += cur.rowcount

                processed += 1
                supported = sum(
                    len(result.get("_pending", []))
                    for result in range_results.values()
                )
                print(
                    f"{symbol}: {overall_status}; {len(rows)} total weeks; "
                    f"{supported}/91 combinations generated"
                )

        conn.commit()

    print(
        f"Daily cycle complete: processed={processed}, skipped={skipped}, "
        f"forecasts_written={forecasts_written}"
    )


if __name__ == "__main__":
    run()
