"""Feature-based ML forecasting with chronological model selection and holdout testing."""
import math
import os
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_LAGS = (1, 2, 3, 4, 8, 13, 26)
ROLLING_WINDOWS = (4, 8, 13, 26)
MIN_FEATURE_HISTORY = max(FEATURE_LAGS)
MIN_SUPERVISED_ROWS = 18
MIN_VALIDATION_ROWS = 3
MIN_TEST_FOLDS = 3
MAX_TEST_FOLDS = 8
RANDOM_STATE = 42
ROUND_TRIP_COST = float(os.getenv('LONG_ROUND_TRIP_COST', '0.002'))
MIN_EXPECTED_NET_RETURN = float(os.getenv('LONG_MIN_EXPECTED_NET_RETURN', '0.002'))
MIN_PREDICTED_GROSS_RETURN = ROUND_TRIP_COST + MIN_EXPECTED_NET_RETURN
if (not math.isfinite(ROUND_TRIP_COST) or not math.isfinite(MIN_EXPECTED_NET_RETURN)
        or ROUND_TRIP_COST < 0 or MIN_EXPECTED_NET_RETURN < 0):
    raise ValueError('Long-term trading cost and minimum net return must be non-negative')
MIN_VALIDATION_TRADES = 3
MAX_ACCEPTABLE_DRAWDOWN = 0.20
MIN_ACCEPTABLE_PROFIT_FACTOR = 1.0


def _features(prices, index):
    p = np.asarray(prices, dtype=float)
    if index < MIN_FEATURE_HISTORY or np.any(p[:index + 1] <= 0):
        raise ValueError("Insufficient or invalid history for features")
    logp = np.log(p)
    log_returns = np.diff(logp[:index + 1])
    out = []
    for lag in FEATURE_LAGS:
        out.append(float(logp[index] - logp[index - lag]))
    for window in ROLLING_WINDOWS:
        r = log_returns[-window:]
        if len(r) < window:
            raise ValueError("Insufficient rolling window")
        out.extend([
            float(np.mean(r)),
            float(np.std(r)),
            float(np.min(r)),
            float(np.max(r)),
            float(np.sum(r)),
        ])
    # Trend slope and distance from rolling means, calculated using past data only.
    x = np.arange(MIN_FEATURE_HISTORY + 1, dtype=float)
    y = logp[index - MIN_FEATURE_HISTORY:index + 1]
    out.append(float(np.polyfit(x, y, 1)[0]))
    for window in ROLLING_WINDOWS:
        mean_price = float(np.mean(p[index - window + 1:index + 1]))
        out.append(float(p[index] / mean_price - 1.0))
    return np.asarray(out, dtype=float)


def build_supervised(prices, horizon_days):
    values = np.asarray(prices, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Prices must be finite and positive")
    step = max(1, math.ceil(horizon_days / 7))
    features, targets, origins, target_indices = [], [], [], []
    for origin in range(MIN_FEATURE_HISTORY, len(values) - step):
        try:
            row = _features(values, origin)
        except ValueError:
            continue
        target = math.log(values[origin + step] / values[origin])
        if not math.isfinite(target):
            continue
        features.append(row)
        targets.append(target)
        origins.append(origin)
        target_indices.append(origin + step)
    if not features:
        return (np.empty((0, 0)), np.array([]), np.array([], dtype=int), np.array([], dtype=int))
    return np.vstack(features), np.asarray(targets), np.asarray(origins), np.asarray(target_indices)


def _models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=120, max_depth=5, min_samples_leaf=3,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=160, max_depth=6, min_samples_leaf=3,
            max_features=0.8, random_state=RANDOM_STATE, n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.04, max_leaf_nodes=7,
            l2_regularization=2.0, random_state=RANDOM_STATE,
        ),
    }


def _score(actual_returns, predicted_returns):
    actual = np.asarray(actual_returns, dtype=float)
    predicted = np.asarray(predicted_returns, dtype=float)
    actual_price_ratio = np.exp(actual)
    predicted_price_ratio = np.exp(predicted)
    baseline = np.ones_like(actual_price_ratio)
    mae = float(mean_absolute_error(actual_price_ratio, predicted_price_ratio))
    baseline_mae = float(mean_absolute_error(actual_price_ratio, baseline))
    rmse = float(np.sqrt(mean_squared_error(actual_price_ratio, predicted_price_ratio)))
    improvement = (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else 0.0
    direction = float(np.mean((actual >= 0) == (predicted >= 0)))
    return {
        "mae_price_ratio": mae,
        "baseline_mae_price_ratio": baseline_mae,
        "rmse_price_ratio": rmse,
        "mae_improvement_vs_flat_price": improvement,
        "directional_accuracy": direction,
        "observations": int(len(actual)),
    }



def _strategy_metrics(actual_log_returns, predicted_log_returns, *, round_trip_cost=ROUND_TRIP_COST, minimum_net_return=MIN_EXPECTED_NET_RETURN):
    """Simulate non-overlapping long-only trades after round-trip costs."""
    actual = np.asarray(actual_log_returns, dtype=float)
    predicted = np.asarray(predicted_log_returns, dtype=float)
    actual_simple = np.expm1(np.clip(actual, -20.0, 20.0))
    predicted_simple = np.expm1(np.clip(predicted, -20.0, 20.0))
    if (not math.isfinite(round_trip_cost) or not math.isfinite(minimum_net_return)
            or round_trip_cost < 0 or minimum_net_return < 0):
        raise ValueError('Trading cost and minimum net return must be finite and non-negative')
    required_gross = round_trip_cost + minimum_net_return
    selected = predicted_simple >= required_gross
    trade_returns = actual_simple[selected] - round_trip_cost
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    wins = []
    losses = []
    for net_return in trade_returns:
        equity *= max(0.0, 1.0 + float(net_return))
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        (wins if net_return > 0 else losses).append(float(net_return))
    gross_wins = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = (min(1000.0, gross_wins / gross_losses)
                     if gross_losses > 0 else (1000.0 if gross_wins > 0 else 0.0))
    total_return = equity - 1.0
    trades = int(len(trade_returns))
    eligible = (trades >= MIN_VALIDATION_TRADES
                and total_return > 0
                and profit_factor > MIN_ACCEPTABLE_PROFIT_FACTOR
                and max_drawdown <= MAX_ACCEPTABLE_DRAWDOWN)
    score = total_return / max(max_drawdown, 0.01) if trades else None
    return {
        "trades": trades,
        "wins": int(np.sum(trade_returns > 0)),
        "losses": int(np.sum(trade_returns < 0)),
        "win_rate": float(np.mean(trade_returns > 0)) if trades else None,
        "total_return": float(total_return),
        "max_drawdown": float(-max_drawdown),
        "profit_factor": float(profit_factor),
        "risk_adjusted_score": float(score) if score is not None else None,
        "round_trip_cost": float(round_trip_cost),
        "minimum_expected_net_return": float(minimum_net_return),
        "minimum_predicted_gross_return": float(required_gross),
        "eligible": bool(eligible),
    }


def _non_overlapping_indices(origins, horizon_step):
    """Keep validation/holdout decisions at least one forecast horizon apart."""
    selected = []
    last_origin = None
    for index, origin in enumerate(np.asarray(origins, dtype=int)):
        if last_origin is None or int(origin) - last_origin >= horizon_step:
            selected.append(index)
            last_origin = int(origin)
    return np.asarray(selected, dtype=int)

def _select_model(X_train, y_train, X_valid, y_valid, valid_origins, horizon_step, *, round_trip_cost=ROUND_TRIP_COST, minimum_net_return=MIN_EXPECTED_NET_RETURN):
    """Select on chronological validation using net return, drawdown and prediction error."""
    if len(y_train) < MIN_SUPERVISED_ROWS or len(y_valid) < MIN_VALIDATION_ROWS:
        return None, {"status": "insufficient_validation"}
    non_overlap = _non_overlapping_indices(valid_origins, horizon_step)
    if len(non_overlap) < MIN_VALIDATION_TRADES:
        return None, {"status": "insufficient_non_overlapping_validation_trades",
                      "validation_trades_available": int(len(non_overlap))}
    results = {}
    for name in _models():
        model = _models()[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        errors = np.abs(np.exp(y_valid) - np.exp(pred))
        trading = _strategy_metrics(y_valid[non_overlap], pred[non_overlap], round_trip_cost=round_trip_cost, minimum_net_return=minimum_net_return)
        results[name] = {
            "validation_mae_price_ratio": float(np.mean(errors)),
            "validation_observations": int(len(errors)),
            "validation_trading": trading,
        }
    eligible = [name for name, result in results.items() if result["validation_trading"]["eligible"]]
    if not eligible:
        return None, {"status": "no_candidate_passed_risk_adjusted_validation",
                      "candidates": results}
    winner = min(eligible, key=lambda name: (
        -results[name]["validation_trading"]["risk_adjusted_score"],
        -results[name]["validation_trading"]["profit_factor"],
        results[name]["validation_mae_price_ratio"]))
    return winner, {"status": "selected_risk_adjusted", "selected_model": winner,
                    "ranking_method": "eligible candidates ranked by net return / max(drawdown, 1%), then profit factor, then validation MAE",
                    "minimum_validation_trades": MIN_VALIDATION_TRADES,
                    "maximum_drawdown_pct": MAX_ACCEPTABLE_DRAWDOWN * 100,
                    "minimum_profit_factor": MIN_ACCEPTABLE_PROFIT_FACTOR,
                    "candidates": results}

def evaluate_ml_horizon(prices, horizon_days, *, round_trip_cost=ROUND_TRIP_COST, minimum_net_return=MIN_EXPECTED_NET_RETURN):
    """Select on an earlier chronological segment; report only later holdout folds."""
    values = np.asarray(prices, dtype=float)
    X, y, origins, target_indices = build_supervised(values, horizon_days)
    if len(y) < MIN_SUPERVISED_ROWS + MIN_VALIDATION_ROWS + MIN_TEST_FOLDS:
        return {
            "status": "insufficient_ml_history",
            "horizon_days": int(horizon_days),
            "supervised_rows": int(len(y)),
            "minimum_supervised_rows": MIN_SUPERVISED_ROWS + MIN_VALIDATION_ROWS + MIN_TEST_FOLDS,
        }

    # Reserve an earlier training period, one horizon-sized validation period,
    # and a later untouched test period. Long horizons may correctly be untestable
    # with only five years of weekly history.
    step = max(1, math.ceil(horizon_days / 7))
    train_cut = max(int(len(values) * 0.40), MIN_FEATURE_HISTORY + MIN_SUPERVISED_ROWS + step)
    validation_cut = train_cut + step + max(step * MIN_VALIDATION_TRADES, MIN_VALIDATION_ROWS)
    train_mask = target_indices <= train_cut
    validation_mask = (origins > train_cut) & (target_indices <= validation_cut)
    test_mask = origins > validation_cut
    X_train, y_train = X[train_mask], y[train_mask]
    X_valid, y_valid = X[validation_mask], y[validation_mask]
    validation_origins = origins[validation_mask]
    if len(y_train) < MIN_SUPERVISED_ROWS or len(y_valid) < MIN_VALIDATION_ROWS:
        return {
            "status": "insufficient_ml_history",
            "horizon_days": int(horizon_days),
            "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_valid)),
            "required_training_rows": MIN_SUPERVISED_ROWS,
            "required_validation_rows": MIN_VALIDATION_ROWS,
        }

    winner, selection = _select_model(X_train, y_train, X_valid, y_valid,
                                      validation_origins, step, round_trip_cost=round_trip_cost,
                                      minimum_net_return=minimum_net_return)
    if winner is None:
        return {"status": selection["status"], "horizon_days": int(horizon_days), **selection}

    # Non-overlapping test origins; every training target is known by its origin.
    test_origins = np.unique(origins[test_mask])[::step]
    if len(test_origins) > MAX_TEST_FOLDS:
        positions = np.linspace(0, len(test_origins) - 1, MAX_TEST_FOLDS, dtype=int)
        test_origins = test_origins[positions]
    model_actual, model_pred, fold_detail = [], [], []
    fold_model_selection = []
    for origin in test_origins:
        train_idx = np.flatnonzero(target_indices <= origin)
        test_idx = np.flatnonzero(origins == origin)
        if len(train_idx) < MIN_SUPERVISED_ROWS or len(test_idx) == 0:
            continue

        # Re-run model competition at each expanding origin. The inner
        # chronological validation labels are purged before the first
        # validation origin, so no overlapping future target leaks backwards.
        fold_winner = winner
        fold_selection_status = "retained_initial_selection_insufficient_fold_history"
        if len(train_idx) >= (MIN_SUPERVISED_ROWS * 2 + MIN_VALIDATION_TRADES):
            split_at = max(MIN_SUPERVISED_ROWS, int(len(train_idx) * 0.75))
            fold_valid_idx = train_idx[split_at:]
            if len(fold_valid_idx) >= MIN_VALIDATION_TRADES:
                first_valid_origin = int(origins[fold_valid_idx[0]])
                fold_train_idx = train_idx[
                    target_indices[train_idx] < first_valid_origin
                ]
                if len(fold_train_idx) >= MIN_SUPERVISED_ROWS:
                    candidate, fold_selection = _select_model(
                        X[fold_train_idx], y[fold_train_idx],
                        X[fold_valid_idx], y[fold_valid_idx],
                        origins[fold_valid_idx], step,
                        round_trip_cost=round_trip_cost,
                        minimum_net_return=minimum_net_return,
                    )
                    if candidate is not None:
                        fold_winner = candidate
                        fold_selection_status = fold_selection.get("status", "selected")
                    else:
                        fold_selection_status = fold_selection.get("status", "fallback")
        fold_model_selection.append({
            "origin_index": int(origin),
            "selected_model": fold_winner,
            "selection_status": fold_selection_status,
        })

        model = _models()[fold_winner]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X[train_idx], y[train_idx])
        prediction = float(model.predict(X[test_idx[:1]])[0])
        actual = float(y[test_idx[0]])
        model_pred.append(prediction)
        model_actual.append(actual)
        fold_detail.append({
            "origin_index": int(origin),
            "target_index": int(target_indices[test_idx[0]]),
            "predicted_log_return": prediction,
            "actual_log_return": actual,
            "selected_model": fold_winner,
        })

    if len(model_actual) < MIN_TEST_FOLDS:
        return {
            "status": "insufficient_holdout_folds",
            "horizon_days": int(horizon_days),
            "selected_model": winner,
            "holdout_folds": int(len(model_actual)),
            "selection": selection,
        }

    # Evaluate returns against a zero-return (last-price) baseline.
    metrics = _score(model_actual, model_pred)
    holdout_non_overlap = _strategy_metrics(model_actual, model_pred, round_trip_cost=round_trip_cost,
                                               minimum_net_return=minimum_net_return)
    final_train_idx = np.flatnonzero(target_indices <= len(values) - 1)
    latest_selected_model = (fold_model_selection[-1]["selected_model"]
                             if fold_model_selection else winner)
    latest_model = _models()[latest_selected_model]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        latest_model.fit(X[final_train_idx], y[final_train_idx])
    latest_features = _features(values, len(values) - 1).reshape(1, -1)
    latest_log_return = float(latest_model.predict(latest_features)[0])
    forecast_price = float(values[-1] * np.exp(np.clip(latest_log_return, -2.0, 2.0)))
    status = ("validated" if metrics["mae_improvement_vs_flat_price"] >= 0.05
              and holdout_non_overlap["eligible"] else "rejected")
    return {
        "status": status,
        "horizon_days": int(horizon_days),
        "effective_horizon_weeks": max(1, math.ceil(horizon_days / 7)),
        "selected_model": latest_selected_model,
        "initial_selection": selection,
        "fold_model_selection": fold_model_selection,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "holdout_folds": int(len(model_actual)),
        "holdout": metrics,
        "holdout_trading": holdout_non_overlap,
        "latest_predicted_log_return": latest_log_return,
        "latest_predicted_price": forecast_price,
        "latest_origin_price": float(values[-1]),
        "folds_detail": fold_detail,
        "model_candidates": list(_models().keys()),
    }
