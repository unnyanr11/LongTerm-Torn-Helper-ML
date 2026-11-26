"""Run per-stock, per-lookback, per-horizon rolling-origin backtests."""
import json
from datetime import timedelta

from .db import connect
from .ml_forecaster import evaluate_ml_horizon, MIN_FEATURE_HISTORY
from .calibration import calibrate_matured_forecasts
from .predict_daily import ANALYSIS_RANGES, FORECAST_HORIZONS
from .stock_universe import STOCKS
from .algorithm_registry import ensure_algorithm_registry, publish_algorithm

MODEL_NAME = "multi_model_ml_walk_forward"
MODEL_VERSION = "3.0.0"


def run():
    processed = skipped = 0
    with connect() as conn:
        with conn.cursor() as cur:
            ensure_algorithm_registry(cur)
        for stock_id, symbol in STOCKS.items():
            with conn.cursor() as cur:
                cur.execute("select pg_try_advisory_xact_lock(%s)", (920000 + stock_id,))
                if not cur.fetchone()[0]:
                    skipped += 1
                    continue
                cur.execute(
                    """select week_start, close_price from long_term.weekly_prices
                       where stock_id=%s and is_complete=true and close_price>0
                       order by week_start""",
                    (stock_id,),
                )
                rows = cur.fetchall()
                if len(rows) < MIN_FEATURE_HISTORY + 25:
                    print(f"{symbol}: insufficient weekly history ({len(rows)})")
                    skipped += 1
                    continue
                origin = rows[-1][0]
                results = {}
                any_validated = False
                calibration_results = {}
                for range_name, range_days in ANALYSIS_RANGES:
                    range_start = origin - timedelta(days=range_days)
                    selected_rows = [(week, price) for week, price in rows if week >= range_start and week <= origin]
                    selected = [price for _week, price in selected_rows]
                    range_result = {"analysis_range_days": range_days, "observations": len(selected),
                                    "horizons": {}}
                    for horizon_name, horizon_days in FORECAST_HORIZONS:
                        evaluation = evaluate_ml_horizon(selected, horizon_days)
                        range_result["horizons"][horizon_name] = evaluation
                        any_validated = any_validated or evaluation.get("status") == "validated"

                        # Recalibrate backend forecasts from matured actual closes.
                        # The calibration module uses unique chronological outcomes,
                        # a 70/30 fit/holdout split, and only promotes proven gains.
                        calibration_results[f"{range_name}:{horizon_name}"] = (
                            calibrate_matured_forecasts(
                                cur, stock_id, range_days, horizon_days
                            )
                        )
                    results[range_name] = range_result

                status = "research"
                metrics = {
                    "kind": "rolling_origin_backtest",
                    "data_through": origin.isoformat(),
                    "method": "walk-forward prediction-versus-actual-price evaluation; chronological model selection; untouched holdout folds",
                    "baseline": "zero log-return / last observed close carried forward",
                    "promotion_rule": "at least 5% holdout MAE improvement over flat-price baseline; minimum three untouched holdout folds",
                    "weekly_horizon_mapping": "ceil(requested_days / 7); targets are observed weekly closes",
                    "candidate_models": ["ridge", "elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting", "ensemble"],
                    "evaluation_scope": "prediction accuracy only; no fees, slippage, net-profit gates, portfolio simulation, or trade execution",
                    "calibration_scope": "matured backend forecast outcomes only; no browser model export or browser inference",
                    "calibration_results": calibration_results,
                    "any_pair_validated": any_validated,
                    "analysis_ranges": results,
                }
                cur.execute(
                    """insert into long_term.model_runs
                       (stock_id,model_name,model_version,data_through,training_start,
                        validation_start,test_start,metrics,status)
                       values (%s,%s,%s,%s,%s,%s,null,%s::jsonb,%s)
                       returning run_id""",
                    (stock_id, MODEL_NAME, MODEL_VERSION, origin, rows[0][0],
                     rows[min(26, len(rows)-1)][0], json.dumps(metrics), status),
                )
                run_id = cur.fetchone()[0]

                # Persist a validated winner separately for every stock/range/horizon.
                # Rejected or insufficient candidates leave the previous winner intact.
                for range_name, range_result in results.items():
                    range_days = int(range_result["analysis_range_days"])
                    for horizon_name, evaluation in range_result.get("horizons", {}).items():
                        family = evaluation.get("selected_model")
                        if evaluation.get("status") != "validated" or not family:
                            continue
                        horizon_days = dict(FORECAST_HORIZONS)[horizon_name]
                        published = publish_algorithm(
                            cur, stock_id=stock_id,
                            analysis_range_days=range_days,
                            horizon_days=horizon_days,
                            model_family=family,
                            data_through=origin,
                            source_model_run_id=run_id,
                            metrics={
                                "holdout": evaluation.get("holdout", {}),
                                "holdout_folds": evaluation.get("holdout_folds", 0),
                                "validation_rows": evaluation.get("validation_rows", 0),
                                "selection": evaluation.get("initial_selection", {}),
                            },
                        )
                        evaluation["selected_algorithm_id"] = published["algorithm_id"]
                        evaluation["algorithm_registry_version"] = published["version"]
                        print(
                            f"{symbol} {range_name}/{horizon_name}: "
                            f"published algorithm {published['algorithm_id']} "
                            f"({family}) v{published['version']}"
                        )
                metrics["analysis_ranges"] = results
                cur.execute(
                    "update long_term.model_runs set metrics=%s::jsonb where run_id=%s",
                    (json.dumps(metrics), run_id),
                )
                processed += 1
                print(f"{symbol}: {status}; {len(rows)} complete weeks")
        conn.commit()
    print(f"Backtest complete: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    run()
