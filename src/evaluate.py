"""
evaluate.py
-----------
Computes point-accuracy metrics and interval calibration for the
three-quantile LightGBM forecast.

Metrics reported
────────────────
Point accuracy (median / q50 forecast)
  • MAE   – Mean Absolute Error
  • RMSE  – Root Mean Squared Error
  • MAPE  – Mean Absolute Percentage Error
  • SMAPE – Symmetric MAPE (handles zeros)

Interval calibration (q10/q90 interval)
  • Coverage  – fraction of actuals inside [q10, q90]; target ≈ 0.80
  • Mean Width – average width of the 80% prediction interval
  • Winkler Score – proper scoring rule for interval forecasts
  • Quantile Loss (pinball) per quantile

Overconfidence analysis
  • Identifies promotion days, weekends, and end-of-month spikes
    where the interval is unusually narrow relative to the error
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Metric helpers ─────────────────────────────────────────────────────────────

def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def mape(y, yhat, eps=1.0):
    mask = y > eps
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100

def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat)) / 2 + 1e-8
    return np.mean(np.abs(y - yhat) / denom) * 100

def pinball_loss(y, yhat, q):
    """Quantile / pinball loss."""
    err = y - yhat
    return np.mean(np.where(err >= 0, q * err, (q - 1) * err))

def winkler_score(y, lower, upper, alpha=0.20):
    """
    Winkler (1972) score for (1-alpha) prediction intervals.
    Lower is better.
    """
    width = upper - lower
    penalty_low  = (2 / alpha) * np.clip(lower - y, 0, None)
    penalty_high = (2 / alpha) * np.clip(y - upper, 0, None)
    return np.mean(width + penalty_low + penalty_high)

def coverage(y, lower, upper):
    return np.mean((y >= lower) & (y <= upper))


# ── Main evaluation ────────────────────────────────────────────────────────────

def evaluate(pred_path: str = "outputs/val_predictions.csv") -> dict:
    df = pd.read_csv(pred_path, parse_dates=["date"])

    y      = df["sales"].values
    q10    = df["pred_q10"].values
    q50    = df["pred_q50"].values
    q90    = df["pred_q90"].values

    # Clip predictions to non-negative
    q10 = np.clip(q10, 0, None)
    q50 = np.clip(q50, 0, None)
    q90 = np.clip(q90, 0, None)

    results = {
        "── Point Accuracy (q50 forecast) ──": None,
        "MAE":   round(mae(y, q50), 4),
        "RMSE":  round(rmse(y, q50), 4),
        "MAPE":  round(mape(y, q50), 2),
        "SMAPE": round(smape(y, q50), 2),
        "── Interval Calibration (80% PI) ──": None,
        "Coverage_80":    round(coverage(y, q10, q90), 4),
        "Mean_Width":     round(np.mean(q90 - q10), 4),
        "Winkler_Score":  round(winkler_score(y, q10, q90, alpha=0.20), 4),
        "── Quantile / Pinball Loss ──": None,
        "Pinball_q10":    round(pinball_loss(y, q10, 0.10), 4),
        "Pinball_q50":    round(pinball_loss(y, q50, 0.50), 4),
        "Pinball_q90":    round(pinball_loss(y, q90, 0.90), 4),
    }

    # Per-month coverage (calibration over time)
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df.groupby("month")
        .apply(lambda g: coverage(g["sales"].values,
                                   np.clip(g["pred_q10"].values, 0, None),
                                   np.clip(g["pred_q90"].values, 0, None)),
               include_groups=False)
        .reset_index(name="coverage_80")
    )
    monthly.to_csv("outputs/monthly_coverage.csv", index=False)

    # Overconfidence analysis: flag rows where actual falls outside interval
    df["interval_width"] = q90 - q10
    df["miss"]           = ((y < q10) | (y > q90)).astype(int)
    df["abs_error"]      = np.abs(y - q50)

    # Tag promotional / seasonal contexts
    month = df["date"].dt.month
    day   = df["date"].dt.day
    dow   = df["date"].dt.dayofweek

    df["context"] = "Normal"
    df.loc[(month == 11) & (dow == 3) & (day >= 22), "context"] = "Black Friday"
    df.loc[(month == 7) & (day >= 18),               "context"] = "Summer Sale"
    df.loc[(month == 12) & (day >= 20),              "context"] = "Christmas"
    df.loc[dow >= 5,                                  "context"] = "Weekend"

    overconf = (
        df.groupby("context")
        .agg(
            n_rows        = ("sales", "count"),
            miss_rate     = ("miss",  "mean"),
            mean_width    = ("interval_width", "mean"),
            mean_abs_err  = ("abs_error", "mean"),
        )
        .reset_index()
        .sort_values("miss_rate", ascending=False)
    )
    overconf.to_csv("outputs/overconfidence_analysis.csv", index=False)

    return results, monthly, overconf


def print_results(results: dict):
    print("\n" + "=" * 48)
    print("  EVALUATION RESULTS")
    print("=" * 48)
    for k, v in results.items():
        if v is None:
            print(f"\n{k}")
        else:
            print(f"  {k:<25} {v}")
    print("=" * 48)


if __name__ == "__main__":
    results, monthly, overconf = evaluate()
    print_results(results)
    print("\n[eval] Monthly coverage (first 6 months):")
    print(monthly.head(6).to_string(index=False))
    print("\n[eval] Overconfidence by context:")
    print(overconf.to_string(index=False))
