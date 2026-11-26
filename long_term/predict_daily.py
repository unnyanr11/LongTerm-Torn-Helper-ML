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
from .calibration import (apply_calibration, calibrate_matured_forecasts,
                          get_active_calibration)
from .evaluation import horizon_weeks
from .ml_forecaster import (evaluate_ml_horizon, ROUND_TRIP_COST, MIN_EXPECTED_NET_RETURN)
from .portfolio import build_portfolio_research
from .stock_universe import STOCKS
from .algorithm_registry import ensure_algorithm_registry, load_active_algorithms

MODEL_NAME = "multi_model_ml_forecast"
MODEL_VERSION = "5.1.0"
DUE_HOURS = 24
MIN_TRAIN_POINTS = 26
MIN_VALIDATION_POINTS = 3

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
    weeks_forward = horizon_weeks(horizon_days)
    prediction = math.exp(intercept + slope * (len(values) - 1 + weeks_forward))
    return max(prediction, 0.000001)


def _range_rows(rows, cutoff, days):
    start = cutoff - timedelta(days=days)
    return [(week, price) for week, price in rows if week >= start and week <= cutoff]



def _forecast_signal(predicted_price, origin_price, validation):
    """Apply live signal costs separately from accuracy-only holdout validation.

    Backtesting establishes whether the forecast model is accurate. Live signal
    eligibility still applies the configured round-trip cost and net-return gate.
    """
    if not isinstance(validation, dict) or validation.get("status") != "validated":
        return "wait"
    try:
        predicted = float(predicted_price)
        origin = float(origin_price)
    except (TypeError, ValueError):
        return "wait"
    if not math.isfinite(predicted) or not math.isfinite(origin) or predicted <= 0 or origin <= 0:
        return "wait"
    expected_return = predicted / origin - 1.0
    required_gross = ROUND_TRIP_COST + MIN_EXPECTED_NET_RETURN
    if expected_return >= required_gross:
        return "buy"
    if expected_return <= -required_gross:
        return "sell"
    return "wait"



def _write_portfolio_allocations(conn, now):
    """Persist research-only, cost-aware allocations for each forecast horizon.

    A missing additive migration is non-fatal to ordinary stock predictions;
    the savepoint keeps the surrounding prediction transaction usable.
    """
    from .evaluation import horizon_weeks

    with conn.cursor() as cur:
        cur.execute("SAVEPOINT portfolio_research")
        try:
            for _horizon_name, horizon_days in FORECAST_HORIZONS:
                cur.execute(
                    """select distinct on (f.stock_id)
                              f.stock_id, f.origin_price, f.predicted_price,
                              f.validation, f.origin_week
                       from long_term.forecasts f
                       where f.analysis_range_days = 365
                         and f.horizon_days = %s
                         and f.issued_at >= now() - interval '2 days'
                       order by f.stock_id, f.issued_at desc, f.forecast_id desc""",
                    (horizon_days,),
                )
                forecasts = cur.fetchall()
                if not forecasts:
                    print(f"Portfolio {horizon_days}d: no recent one-year-range forecasts")
                    continue

                stock_ids = [int(row[0]) for row in forecasts]
                snapshot_week = max(row[4] for row in forecasts)
                cur.execute(
                    """select stock_id, week_start, close_price
                       from long_term.weekly_prices
                       where stock_id = any(%s)
                         and week_start >= %s - interval '730 days'
                         and week_start <= %s
                         and is_complete = true
                         and close_price is not null and close_price > 0
                       order by stock_id, week_start""",
                    (stock_ids, snapshot_week, snapshot_week),
                )
                weekly_rows = cur.fetchall()
                forecast_inputs = [
                    {
                        "stock_id": row[0],
                        "origin_price": row[1],
                        "predicted_price": row[2],
                        "validation": row[3],
                    }
                    for row in forecasts
                ]
                result = build_portfolio_research(
                    forecast_inputs,
                    weekly_rows,
                    horizon_weeks(horizon_days),
                    round_trip_cost=ROUND_TRIP_COST,
                    minimum_net_return=MIN_EXPECTED_NET_RETURN,
                )
                allocations = result.get("allocations", {})
                metrics = {
                    key: value for key, value in result.items()
                    if key not in ("allocations", "weights", "expected_net_returns_by_stock")
                }
                cur.execute(
                    """insert into long_term.portfolio_allocations
                       (snapshot_week, analysis_range_days, horizon_days, status,
                        eligible_assets, allocations, metrics)
                       values (%s,365,%s,%s,%s,%s::jsonb,%s::jsonb)
                       on conflict (snapshot_week, analysis_range_days, horizon_days)
                       do update set status=excluded.status,
                         eligible_assets=excluded.eligible_assets,
                         allocations=excluded.allocations,
                         metrics=excluded.metrics,
                         created_at=now()""",
                    (
                        snapshot_week, horizon_days, result["status"],
                        int(result.get("eligible_assets", 0)),
                        json.dumps(allocations), json.dumps(metrics),
                    ),
                )
                print(
                    f"Portfolio {horizon_days}d: {result['status']}; "
                    f"{result.get('eligible_assets', 0)} eligible assets; "
                    f"{len(allocations)} allocated positions"
                )
            cur.execute("RELEASE SAVEPOINT portfolio_research")
        except Exception as exc:
            cur.execute("ROLLBACK TO SAVEPOINT portfolio_research")
            cur.execute("RELEASE SAVEPOINT portfolio_research")
            print(
                "Portfolio research skipped without interrupting forecasts: "
                f"{type(exc).__name__}: {exc}"
            )

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

                ensure_algorithm_registry(cur)
                active_algorithms = load_active_algorithms(cur, stock_id)
                origin_week = rows[-1][0]
                origin_price = float(rows[-1][1])
                range_results = {}

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
                        result["status"] = "research"

                        for horizon_name, horizon_days in FORECAST_HORIZONS:
                            registry_entry = active_algorithms.get((range_days, horizon_days))
                            horizon_validation = evaluate_ml_horizon(
                                prices, horizon_days,
                                preferred_model=(registry_entry["model_family"] if registry_entry else None),
                            )
                            predicted = horizon_validation.get("latest_predicted_price")
                            if predicted is None:
                                continue

                            # Matured forecast outcomes are recalibrated automatically.
                            # A candidate is promoted only after a chronological holdout win.
                            calibration_candidate = calibrate_matured_forecasts(
                                cur, stock_id, range_days, horizon_days
                            )
                            active_calibration = get_active_calibration(
                                cur, stock_id, range_days, horizon_days
                            )
                            raw_predicted = float(predicted)
                            predicted = apply_calibration(
                                raw_predicted, origin_price, active_calibration
                            )
                            predicted_return = (predicted / origin_price - 1.0) * 100.0
                            effective_days = horizon_weeks(horizon_days) * 7
                            target_date = origin_week + timedelta(days=effective_days)
                            signal = _forecast_signal(predicted, origin_price, horizon_validation)
                            forecast_validation = {
                                "analysis_range": range_name,
                                "analysis_range_days": range_days,
                                "forecast_horizon": horizon_name,
                                "requested_horizon_days": horizon_days,
                                "effective_weekly_horizon_days": effective_days,
                                "horizon_validation": horizon_validation,
                                "selected_model": horizon_validation.get("selected_model"),
                                "selected_algorithm_id": horizon_validation.get("selected_algorithm_id"),
                                "algorithm_selection_source": horizon_validation.get("algorithm_selection_source"),
                                "algorithm_registry_version": (registry_entry.get("version") if registry_entry else None),
                                "holdout_metrics": horizon_validation.get("holdout"),
                                "calibration": {
                                    "active": active_calibration,
                                    "candidate": calibration_candidate,
                                    "raw_predicted_price": raw_predicted,
                                },
                                "observations": len(prices),
                                "data_through": origin_week.isoformat(),
                                "note": "Projection from weekly closes; target aligns to an observed weekly interval.",
                            }
                            result.setdefault("forecasts", []).append(horizon_name)
                            pending_forecasts = result.setdefault("_pending", [])
                            pending_forecasts.append((
                                range_days, horizon_days, target_date,
                                predicted, predicted_return, signal,
                                json.dumps(forecast_validation),
                            ))
                    range_results[range_name] = result

                overall_status = "research"
                all_metrics = {
                    "analysis_ranges": {
                        name: {k: v for k, v in result.items() if k != "_pending"}
                        for name, result in range_results.items()
                    },
                    "forecast_horizons_days": {
                        name: days for name, days in FORECAST_HORIZONS
                    },
                    "minimum_training_observations": MIN_TRAIN_POINTS,
                    "model_candidates": ["ridge", "random_forest", "extra_trees", "hist_gradient_boosting"],
                    "model_selection": "chronological validation followed by non-overlapping holdout evaluation",
                    "validation_rule": "horizon-specific non-overlapping walk-forward MAE must beat last-close baseline by at least 5%, with at least 3 folds",
                    "weekly_horizon_mapping": "ceil(requested_days / 7) weekly observations",
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

        _write_portfolio_allocations(conn, now)
        conn.commit()

    print(
        f"Daily cycle complete: processed={processed}, skipped={skipped}, "
        f"forecasts_written={forecasts_written}"
    )


if __name__ == "__main__":
    run()
