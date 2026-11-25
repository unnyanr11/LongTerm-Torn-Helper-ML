"""Leakage-aware, horizon-specific rolling-origin evaluation on weekly closes."""
import math
import numpy as np

MIN_BACKTEST_TRAIN_WEEKS = 26
MIN_BACKTEST_FOLDS = 3


def horizon_weeks(horizon_days: int) -> int:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    return max(1, math.ceil(horizon_days / 7))


def fit_log_linear(prices, steps_ahead: int) -> float:
    values = np.asarray(prices, dtype=float)
    if len(values) < MIN_BACKTEST_TRAIN_WEEKS or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Need at least 26 finite positive weekly closes")
    if steps_ahead < 1:
        raise ValueError("steps_ahead must be positive")
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, np.log(values), 1)
    estimate = float(np.exp(intercept + slope * (len(values) - 1 + steps_ahead)))
    if not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("Model produced an invalid price")
    return estimate


def evaluate_horizon(prices, horizon_days: int, min_folds: int = MIN_BACKTEST_FOLDS):
    """Non-overlapping rolling-origin folds; every target is strictly in the future."""
    values = [float(x) for x in prices]
    step = horizon_weeks(horizon_days)
    model_errors, naive_errors, actual_returns, predicted_returns = [], [], [], []
    fold_rows = []
    # Start with six months of history; step by the forecast horizon so targets do not overlap.
    for origin_idx in range(MIN_BACKTEST_TRAIN_WEEKS, len(values) - step, step):
        history = values[:origin_idx]
        actual = values[origin_idx + step]
        origin = values[origin_idx - 1]
        if min(history) <= 0 or actual <= 0:
            continue
        predicted = fit_log_linear(history, step)
        model_errors.append(abs(predicted - actual))
        naive_errors.append(abs(origin - actual))
        actual_returns.append((actual / origin) - 1.0)
        predicted_returns.append((predicted / origin) - 1.0)
        fold_rows.append({
            "origin_index": origin_idx - 1,
            "target_index": origin_idx + step,
            "origin_price": origin,
            "predicted_price": predicted,
            "actual_price": actual,
        })

    if len(model_errors) < min_folds:
        return {
            "status": "insufficient_backtest",
            "horizon_days": horizon_days,
            "effective_horizon_weeks": step,
            "folds": len(model_errors),
            "minimum_folds": min_folds,
            "minimum_training_weeks": MIN_BACKTEST_TRAIN_WEEKS,
        }

    model_mae = float(np.mean(model_errors))
    naive_mae = float(np.mean(naive_errors))
    model_rmse = float(np.sqrt(np.mean(np.square(model_errors))))
    naive_rmse = float(np.sqrt(np.mean(np.square(naive_errors))))
    model_mape = float(np.mean([
        abs(e) / actual for e, actual in zip(model_errors, [values[f["target_index"]] for f in fold_rows])
    ]) * 100.0)
    improvement = (naive_mae - model_mae) / naive_mae if naive_mae > 0 else 0.0
    direction_hits = [
        (p >= 0 and a >= 0) or (p < 0 and a < 0)
        for p, a in zip(predicted_returns, actual_returns)
    ]
    return {
        "status": "validated" if improvement >= 0.05 else "rejected",
        "horizon_days": horizon_days,
        "effective_horizon_weeks": step,
        "folds": len(model_errors),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "model_rmse": model_rmse,
        "naive_rmse": naive_rmse,
        "model_mape_pct": model_mape,
        "directional_accuracy": float(np.mean(direction_hits)),
        "mae_improvement_vs_last_close": improvement,
        "minimum_required_improvement": 0.05,
        "folds_detail": fold_rows,
    }
