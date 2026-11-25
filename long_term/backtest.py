"""Run per-stock, per-lookback, per-horizon rolling-origin backtests."""
import json
from datetime import timedelta

from .db import connect
from .ml_forecaster import evaluate_ml_horizon, MIN_FEATURE_HISTORY
from .predict_daily import ANALYSIS_RANGES, FORECAST_HORIZONS
from .stock_universe import STOCKS

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
                    selected = [price for week, price in rows if week >= range_start and week <= origin]
                    range_result = {"analysis_range_days": range_days, "observations": len(selected), "horizons": {}}
                    for horizon_name, horizon_days in FORECAST_HORIZONS:
                        evaluation = evaluate_ml_horizon(selected, horizon_days)
                        range_result["horizons"][horizon_name] = evaluation
                        any_validated = any_validated or evaluation.get("status") == "validated"
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
