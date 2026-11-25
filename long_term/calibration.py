"""Matured-outcome calibration for long-term weekly forecasts.

Calibration is isolated by stock, lookback and horizon. Candidates are fitted on
older matured forecasts and accepted only when they improve on a later holdout.
"""
import json
import math

import numpy as np

MIN_SAMPLES = 20
MIN_HOLDOUT = 6
MIN_IMPROVEMENT = 0.02
MIN_SCALE = 0.50
MAX_SCALE = 1.50
MAX_ABS_BIAS = 0.10


def fit_calibration(predicted_log_returns, actual_log_returns):
    predicted = np.asarray(predicted_log_returns, dtype=float)
    actual = np.asarray(actual_log_returns, dtype=float)
    if len(predicted) != len(actual) or len(predicted) < MIN_SAMPLES:
        return {"status": "insufficient_samples", "samples": int(len(predicted))}
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(actual)):
        return {"status": "invalid_samples", "samples": int(len(predicted))}
    split = max(14, int(len(predicted) * 0.70))
    if len(predicted) - split < MIN_HOLDOUT:
        split = len(predicted) - MIN_HOLDOUT
    if split < 14:
        return {"status": "insufficient_holdout", "samples": int(len(predicted))}
    design = np.column_stack([predicted[:split], np.ones(split)])
    scale, bias = np.linalg.lstsq(design, actual[:split], rcond=None)[0]
    scale, bias = float(scale), float(bias)
    if not (MIN_SCALE <= scale <= MAX_SCALE and abs(bias) <= MAX_ABS_BIAS):
        return {"status": "unsafe_parameters", "scale": scale, "bias": bias,
                "samples": int(len(predicted))}
    raw_mae = float(np.mean(np.abs(actual[split:] - predicted[split:])))
    calibrated_mae = float(np.mean(np.abs(actual[split:] -
                                           (scale * predicted[split:] + bias))))
    improvement = ((raw_mae - calibrated_mae) / raw_mae
                   if raw_mae > 0 else 0.0)
    return {
        "status": "accepted" if improvement >= MIN_IMPROVEMENT else "rejected",
        "scale": scale, "bias": bias, "samples": int(len(predicted)),
        "training_samples": int(split), "holdout_samples": int(len(predicted) - split),
        "raw_holdout_mae": raw_mae, "calibrated_holdout_mae": calibrated_mae,
        "holdout_mae_improvement": float(improvement),
        "minimum_improvement": MIN_IMPROVEMENT,
    }


def calibrate_matured_forecasts(cur, stock_id, analysis_range_days, horizon_days):
    """Evaluate matured forecasts and atomically promote a passing calibration."""
    cur.execute(
        """select f.origin_week, f.origin_price, f.predicted_price, w.close_price
           from long_term.forecasts f
           join long_term.weekly_prices w
             on w.stock_id = f.stock_id and w.week_start = f.target_week
           where f.stock_id = %s and f.analysis_range_days = %s
             and f.horizon_days = %s and w.is_complete = true
             and f.origin_price > 0 and f.predicted_price > 0
             and w.close_price > 0
           order by f.target_week, f.origin_week""",
        (stock_id, analysis_range_days, horizon_days),
    )
    rows = cur.fetchall()
    predicted, actual = [], []
    for _origin_week, origin_price, predicted_price, actual_price in rows:
        predicted.append(math.log(float(predicted_price) / float(origin_price)))
        actual.append(math.log(float(actual_price) / float(origin_price)))
    candidate = fit_calibration(predicted, actual)
    if candidate["status"] != "accepted":
        return candidate

    cur.execute(
        """select coalesce(max(version), 0) + 1
           from long_term.forecast_calibrations
           where stock_id=%s and analysis_range_days=%s and horizon_days=%s""",
        (stock_id, analysis_range_days, horizon_days),
    )
    version = int(cur.fetchone()[0])
    cur.execute(
        """update long_term.forecast_calibrations set status='superseded'
           where stock_id=%s and analysis_range_days=%s and horizon_days=%s
             and status='active'""",
        (stock_id, analysis_range_days, horizon_days),
    )
    cur.execute(
        """insert into long_term.forecast_calibrations
           (stock_id, analysis_range_days, horizon_days, version, scale, bias,
            sample_count, training_samples, holdout_samples, raw_holdout_mae,
            calibrated_holdout_mae, holdout_mae_improvement, status, metrics)
           values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'active',%s::jsonb)""",
        (stock_id, analysis_range_days, horizon_days, version,
         candidate["scale"], candidate["bias"], candidate["samples"],
         candidate["training_samples"], candidate["holdout_samples"],
         candidate["raw_holdout_mae"], candidate["calibrated_holdout_mae"],
         candidate["holdout_mae_improvement"], json.dumps(candidate)),
    )
    candidate.update({"status": "activated", "version": version})
    return candidate


def get_active_calibration(cur, stock_id, analysis_range_days, horizon_days):
    cur.execute(
        """select version, scale, bias, sample_count, holdout_mae_improvement
           from long_term.forecast_calibrations
           where stock_id=%s and analysis_range_days=%s and horizon_days=%s
             and status='active'
           order by version desc limit 1""",
        (stock_id, analysis_range_days, horizon_days),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"version": int(row[0]), "scale": float(row[1]), "bias": float(row[2]),
            "samples": int(row[3]), "holdout_mae_improvement": float(row[4])}


def apply_calibration(predicted_price, origin_price, calibration):
    if not calibration or predicted_price <= 0 or origin_price <= 0:
        return float(predicted_price)
    raw_log_return = math.log(float(predicted_price) / float(origin_price))
    adjusted = calibration["scale"] * raw_log_return + calibration["bias"]
    return float(origin_price) * math.exp(max(-2.0, min(2.0, adjusted)))
