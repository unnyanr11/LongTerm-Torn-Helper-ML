"""Run per-stock, per-lookback, per-horizon rolling-origin backtests."""
import json
from datetime import datetime, timedelta, timezone

from .db import connect
from .ml_forecaster import evaluate_ml_horizon, MIN_FEATURE_HISTORY
from .predict_daily import ANALYSIS_RANGES, FORECAST_HORIZONS
from .stock_universe import STOCKS
from .strategy_engine import run_strategy_competition

MODEL_NAME = "multi_model_ml_walk_forward"
MODEL_VERSION = "3.0.0"


def run():
    processed = skipped = 0
    with connect() as conn:
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
                for range_name, range_days in ANALYSIS_RANGES:
                    range_start = origin - timedelta(days=range_days)
                    selected_rows = [(week, price) for week, price in rows if week >= range_start and week <= origin]
                    selected = [price for _week, price in selected_rows]
                    strategy_times = [
                        int(datetime.combine(week, datetime.min.time(), tzinfo=timezone.utc).timestamp())
                        for week, _price in selected_rows
                    ]
                    range_result = {"analysis_range_days": range_days, "observations": len(selected),
                                    "horizons": {}, "technical_strategies": {}}
                    for horizon_name, horizon_days in FORECAST_HORIZONS:
                        evaluation = evaluate_ml_horizon(selected, horizon_days)
                        range_result["horizons"][horizon_name] = evaluation
                        any_validated = any_validated or evaluation.get("status") == "validated"
                        if len(selected) >= 80:
                            # Keep indicator warm-up and model-selection history in the
                            # first 70%; evaluate strategy trades only on the later,
                            # untouched chronological holdout segment.
                            strategy_evaluation_index = max(55, int(len(strategy_times) * 0.70))
                            strategy_evaluation_start = strategy_times[strategy_evaluation_index]
                            range_result["technical_strategy_evaluation"] = {
                                "method": "first 70% causal history for indicator warm-up; final 30% untouched chronological holdout",
                                "training_observations": strategy_evaluation_index,
                                "holdout_observations": len(strategy_times) - strategy_evaluation_index,
                                "holdout_start_week": selected_rows[strategy_evaluation_index][0].isoformat(),
                            }
                            strategy_result = run_strategy_competition(
                                strategy_times, selected,
                                horizon_steps=max(1, (horizon_days + 6) // 7),
                                bar_seconds=7 * 86400,
                                fee_bps=10, slippage_bps=5,
                                round_trip_cost=0.002,
                                minimum_net_return=0.002,
                                max_drawdown=0.20,
                                evaluation_start=strategy_evaluation_start,
                            )
                            range_result["technical_strategies"][horizon_name] = {
                                "evaluation_method": "final_30_percent_chronological_holdout",
                                "evaluation_start_week": selected_rows[strategy_evaluation_index][0].isoformat(),
                                "config": strategy_result["config"],
                                "best_strategy": strategy_result["best_strategy"],
                                "best_strategy_for_current_regime": strategy_result.get("best_strategy_for_current_regime"),
                                "multi_timeframe_confirmation": strategy_result.get("multi_timeframe_confirmation"),
                                "opportunity_ranking": strategy_result.get("opportunity_ranking", []),
                                "opportunity_probability_basis": strategy_result.get("opportunity_probability_basis"),
                                "strategies": {
                                    name: {"status": item["status"], "metrics": item["metrics"],
                                           "trades": item["trades"]}
                                    for name, item in strategy_result["strategies"].items()
                                },
                            }
                        else:
                            range_result["technical_strategies"][horizon_name] = {
                                "status": "insufficient_history",
                                "observations": len(selected),
                                "required_observations": 80,
                            }
                    results[range_name] = range_result

                status = "research"
                metrics = {
                    "kind": "rolling_origin_backtest",
                    "data_through": origin.isoformat(),
                    "method": "feature-based model competition; chronological validation; non-overlapping untouched holdout folds",
                    "baseline": "zero log-return / last observed close carried forward",
                    "promotion_rule": "at least 5% holdout MAE improvement over baseline with at least 3 non-overlapping folds",
                    "weekly_horizon_mapping": "ceil(requested_days / 7); targets are observed weekly closes",
                    "candidate_models": ["ridge", "random_forest", "extra_trees", "hist_gradient_boosting"],
                    "any_pair_validated": any_validated,
                    "analysis_ranges": results,
                }
                cur.execute(
                    """insert into long_term.model_runs
                       (stock_id,model_name,model_version,data_through,training_start,
                        validation_start,test_start,metrics,status)
                       values (%s,%s,%s,%s,%s,%s,null,%s::jsonb,%s)""",
                    (stock_id, MODEL_NAME, MODEL_VERSION, origin, rows[0][0],
                     rows[min(26, len(rows)-1)][0], json.dumps(metrics), status),
                )
                processed += 1
                print(f"{symbol}: {status}; {len(rows)} complete weeks")
        conn.commit()
    print(f"Backtest complete: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    run()
