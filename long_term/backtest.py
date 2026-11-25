"""Run per-stock, per-lookback, per-horizon rolling-origin backtests."""
import json
from datetime import timedelta, timezone, datetime

from .db import connect
from .evaluation import evaluate_horizon, MIN_BACKTEST_TRAIN_WEEKS
from .predict_daily import ANALYSIS_RANGES, FORECAST_HORIZONS
from .stock_universe import STOCKS

MODEL_NAME = "weekly_log_linear_walk_forward"
MODEL_VERSION = "1.0.0"


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
                if len(rows) < MIN_BACKTEST_TRAIN_WEEKS + 3:
                    print(f"{symbol}: insufficient weekly history ({len(rows)})")
                    skipped += 1
                    continue
                all_prices = [float(r[1]) for r in rows]
                origin = rows[-1][0]
                results = {}
                any_validated = False
                for range_name, range_days in ANALYSIS_RANGES:
                    range_start = origin - timedelta(days=range_days)
                    selected = [price for week, price in rows if week >= range_start and week <= origin]
                    range_result = {"analysis_range_days": range_days, "observations": len(selected), "horizons": {}}
                    for horizon_name, horizon_days in FORECAST_HORIZONS:
                        evaluation = evaluate_horizon(selected, horizon_days)
                        range_result["horizons"][horizon_name] = evaluation
                        any_validated = any_validated or evaluation.get("status") == "validated"
                    results[range_name] = range_result

                status = "validated" if any_validated else "rejected"
                metrics = {
                    "kind": "rolling_origin_backtest",
                    "data_through": origin.isoformat(),
                    "method": "expanding training window; six-month minimum training; non-overlapping target folds",
                    "baseline": "last observed close carried forward",
                    "promotion_rule": "at least 5% MAE improvement over baseline with at least 3 folds",
                    "weekly_horizon_mapping": "ceil(requested_days / 7); targets are observed weekly closes",
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
