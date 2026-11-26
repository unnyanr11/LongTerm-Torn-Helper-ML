"""Long-only, cost-aware portfolio allocation for long-term forecasts.

This module is deliberately pure: it does not connect to the database or place
orders. It converts validated per-stock expected returns and a weekly-return
covariance matrix into research allocations using projected-gradient
mean-variance optimization.
"""
import math
import numpy as np


def _project_capped_simplex(values, total=1.0, cap=0.15, iterations=80):
    """Project onto {w: sum(w)=total, 0<=w<=cap} by bisection."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    if not math.isfinite(total) or not math.isfinite(cap) or total <= 0 or cap <= 0:
        raise ValueError("total and cap must be finite and positive")
    if total > len(values) * cap + 1e-12:
        raise ValueError("position cap is too small to allocate the requested total")
    low = float(np.min(values - cap))
    high = float(np.max(values))
    for _ in range(iterations):
        mid = (low + high) / 2.0
        projected = np.clip(values - mid, 0.0, cap)
        if projected.sum() > total:
            low = mid
        else:
            high = mid
    result = np.clip(values - (low + high) / 2.0, 0.0, cap)
    # Correct floating point residue without violating bounds.
    residue = total - float(result.sum())
    if abs(residue) > 1e-10:
        room = cap - result if residue > 0 else result
        eligible = np.flatnonzero(room > 1e-12)
        for idx in eligible:
            change = math.copysign(min(abs(residue), float(room[idx])), residue)
            result[idx] += change
            residue -= change
            if abs(residue) <= 1e-10:
                break
    return result


def optimize_portfolio(
    expected_net_returns,
    covariance,
    *,
    risk_aversion=4.0,
    max_weight=0.15,
    max_iterations=2000,
    tolerance=1e-10,
):
    """Maximize mu'w - risk_aversion/2 * w'Σw for long-only weights.

    Inputs are expected net returns and covariance on the same horizon. Assets
    with invalid forecasts or non-finite covariance are rejected. Callers should
    filter out forecasts that have not passed their own out-of-sample gates.
    """
    mu = np.asarray(expected_net_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or not len(mu):
        raise ValueError("expected_net_returns must be a non-empty vector")
    if cov.shape != (len(mu), len(mu)):
        raise ValueError("covariance shape must match the number of assets")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
        raise ValueError("inputs must be finite")
    if not math.isfinite(risk_aversion) or risk_aversion <= 0:
        raise ValueError("risk_aversion must be finite and positive")
    if not math.isfinite(max_weight) or max_weight <= 0:
        raise ValueError("max_weight must be finite and positive")

    # Symmetrize and add a small diagonal ridge to stabilize noisy covariance.
    cov = (cov + cov.T) / 2.0
    cov = cov + np.eye(len(mu)) * max(1e-10, float(np.trace(cov)) / max(1, len(mu)) * 1e-6)
    eigen_max = max(float(np.linalg.eigvalsh(cov).max()), 1e-10)
    step_size = 1.0 / (risk_aversion * eigen_max)
    target_weight = min(1.0, len(mu) * max_weight)
    weights = _project_capped_simplex(np.ones(len(mu)) / len(mu), total=target_weight, cap=max_weight)

    for _ in range(max_iterations):
        gradient = mu - risk_aversion * (cov @ weights)
        updated = _project_capped_simplex(weights + step_size * gradient, total=target_weight, cap=max_weight)
        if float(np.max(np.abs(updated - weights))) <= tolerance:
            weights = updated
            break
        weights = updated

    expected_return = float(mu @ weights)
    variance = max(0.0, float(weights @ cov @ weights))
    volatility = math.sqrt(variance)
    return {
        "weights": weights.tolist(),
        "expected_net_return": expected_return,
        "expected_volatility": volatility,
        "risk_adjusted_return": expected_return / volatility if volatility > 0 else None,
        "invested_weight": float(weights.sum()),
        "cash_weight": max(0.0, 1.0 - float(weights.sum())),
        "max_position_weight": float(weights.max()),
        "assets": int(len(weights)),
        "optimizer": "long_only_capped_mean_variance_projected_gradient",
        "risk_aversion": float(risk_aversion),
        "max_weight": float(max_weight),
    }

def build_portfolio_research(
    forecasts,
    weekly_rows,
    horizon_weeks,
    *,
    round_trip_cost=0.002,
    minimum_net_return=0.002,
    risk_aversion=4.0,
    max_weight=0.15,
    minimum_common_weeks=13,
):
    """Build a portfolio from validated forecasts and aligned weekly closes.

    forecasts: iterable of mappings with stock_id, origin_price, predicted_price,
    and validation (the stored forecast-validation JSON).
    weekly_rows: iterable of (stock_id, week_start, close_price).
    """
    eligible = []
    for row in forecasts:
        validation = row.get("validation") or {}
        if isinstance(validation, str):
            import json
            try:
                validation = json.loads(validation)
            except (TypeError, ValueError):
                continue
        horizon_validation = validation.get("horizon_validation", validation)
        trading = horizon_validation.get("holdout_trading", {})
        if horizon_validation.get("status") != "validated" or trading.get("eligible") is not True:
            continue
        try:
            stock_id = int(row["stock_id"])
            origin = float(row["origin_price"])
            predicted = float(row["predicted_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(origin) and math.isfinite(predicted) and origin > 0 and predicted > 0):
            continue
        expected_net = predicted / origin - 1.0 - round_trip_cost
        if expected_net >= minimum_net_return:
            eligible.append((stock_id, expected_net))

    if not eligible:
        return {"status": "no_eligible_assets", "allocations": {}, "eligible_assets": 0}
    eligible.sort(key=lambda item: item[0])
    stock_ids = [item[0] for item in eligible]
    expected = np.asarray([item[1] for item in eligible], dtype=float)

    history = {stock_id: {} for stock_id in stock_ids}
    for stock_id, week_start, close_price in weekly_rows:
        stock_id = int(stock_id)
        if stock_id not in history:
            continue
        price = float(close_price)
        if math.isfinite(price) and price > 0:
            history[stock_id][week_start] = price

    common_weeks = set.intersection(*(set(history[sid]) for sid in stock_ids))
    common_weeks = sorted(common_weeks)
    if len(common_weeks) < minimum_common_weeks:
        return {
            "status": "insufficient_common_history",
            "allocations": {},
            "eligible_assets": len(stock_ids),
            "common_weeks": len(common_weeks),
            "minimum_common_weeks": minimum_common_weeks,
        }

    price_matrix = np.asarray(
        [[history[sid][week] for week in common_weeks] for sid in stock_ids],
        dtype=float,
    )
    weekly_returns = np.diff(np.log(price_matrix), axis=1)
    covariance = np.atleast_2d(np.cov(weekly_returns, rowvar=True, ddof=1))
    if len(stock_ids) == 1:
        covariance = np.asarray([[float(np.var(weekly_returns[0], ddof=1))]])
    covariance = covariance * max(1, int(horizon_weeks))
    result = optimize_portfolio(
        expected,
        covariance,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
    )
    result.update({
        "status": "optimized",
        "horizon_weeks": int(horizon_weeks),
        "eligible_assets": len(stock_ids),
        "common_weeks": len(common_weeks),
        "allocations": {
            str(stock_id): float(weight)
            for stock_id, weight in zip(stock_ids, result["weights"])
            if weight > 1e-8
        },
        "expected_net_returns_by_stock": {
            str(stock_id): float(value)
            for stock_id, value in zip(stock_ids, expected)
        },
    })
    return result
