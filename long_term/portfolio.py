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
