"""
inventory_decision.py
---------------------
Translates probabilistic forecasts into inventory decisions and computes
the cost tradeoff between a point-forecast policy and a quantile policy.

Framework
─────────────────────────────────────────────────────────────────────────
Newsvendor model (single-period, per item per day):

  Order Quantity Q chosen to minimise expected cost:
      Cost(Q) = c_over × max(Q - D, 0) + c_under × max(D - Q, 0)

  Critical ratio (optimal fractile):
      r* = c_under / (c_over + c_under)

  Two policies compared:
  ┌───────────────────────────┬─────────────────────────────────────────┐
  │ Policy                    │ Order Quantity                          │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Point-forecast (median)   │ Q = q50  (same as median forecast)     │
  │ Quantile policy (q90)     │ Q = q90  (80% PI upper bound)          │
  └───────────────────────────┴─────────────────────────────────────────┘

  Safety stock:
      SS_quantile = q90 - q50  (interval half-width)
      SS_point    = 0          (baseline, no safety stock)

Usage
─────
  python inventory_decision.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ── Cost parameters (easily changed) ──────────────────────────────────────────

COST_OVERSTOCK  = 0.15   # $ per unit per day held (holding + spoilage)
COST_UNDERSTOCK = 1.20   # $ per unit short (lost margin + goodwill)

# Critical ratio
CRITICAL_RATIO  = COST_UNDERSTOCK / (COST_OVERSTOCK + COST_UNDERSTOCK)


def newsvendor_cost(actual: np.ndarray, order: np.ndarray,
                    c_over: float, c_under: float) -> np.ndarray:
    """Per-row newsvendor cost."""
    overstock  = np.clip(order - actual, 0, None)
    understock = np.clip(actual - order, 0, None)
    return c_over * overstock + c_under * understock


def compute_safety_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends columns for:
      • order_point  : q50 (point policy)
      • order_q90    : q90 (quantile policy)
      • safety_stock : q90 - q50
      • cost_point   : newsvendor cost under point policy
      • cost_q90     : newsvendor cost under q90 policy
      • cost_saving  : cost_point - cost_q90 (positive = q90 policy wins)
    """
    df = df.copy()
    df["order_point"] = np.clip(df["pred_q50"], 0, None)
    df["order_q90"]   = np.clip(df["pred_q90"], 0, None)
    df["safety_stock"]= df["order_q90"] - df["order_point"]

    df["cost_point"]  = newsvendor_cost(
        df["sales"].values, df["order_point"].values,
        COST_OVERSTOCK, COST_UNDERSTOCK
    )
    df["cost_q90"]    = newsvendor_cost(
        df["sales"].values, df["order_q90"].values,
        COST_OVERSTOCK, COST_UNDERSTOCK
    )
    df["cost_saving"] = df["cost_point"] - df["cost_q90"]
    return df


def summarise(df: pd.DataFrame) -> dict:
    total_cost_point = df["cost_point"].sum()
    total_cost_q90   = df["cost_q90"].sum()
    total_saving     = df["cost_saving"].sum()
    pct_saving       = total_saving / total_cost_point * 100

    avg_safety_stock = df["safety_stock"].mean()
    holding_cost_ss  = avg_safety_stock * COST_OVERSTOCK * len(df)

    # Stockout rate per policy
    sr_point = (df["sales"] > df["order_point"]).mean() * 100
    sr_q90   = (df["sales"] > df["order_q90"]).mean() * 100

    return {
        "── Cost Parameters ──────────────────": None,
        "c_overstock  ($/unit/day)": COST_OVERSTOCK,
        "c_understock ($/unit/day)": COST_UNDERSTOCK,
        "Critical ratio":           round(CRITICAL_RATIO, 3),
        "── Policy Comparison ────────────────": None,
        "Total cost – point policy ($)":    round(total_cost_point, 2),
        "Total cost – q90 policy   ($)":    round(total_cost_q90, 2),
        "Net saving with q90 policy ($)":   round(total_saving, 2),
        "Saving (%)":                       round(pct_saving, 2),
        "── Safety Stock ─────────────────────": None,
        "Avg daily safety stock (units)":   round(avg_safety_stock, 2),
        "Implied SS holding cost ($)":      round(holding_cost_ss, 2),
        "── Stockout Rate ────────────────────": None,
        "Stockout rate – point policy (%)": round(sr_point, 2),
        "Stockout rate – q90 policy   (%)": round(sr_q90, 2),
    }


def run(pred_path: str = "outputs/val_predictions.csv") -> pd.DataFrame:
    df = pd.read_csv(pred_path, parse_dates=["date"])
    df = compute_safety_stock(df)

    summary = summarise(df)

    print("\n" + "=" * 52)
    print("  INVENTORY DECISION ANALYSIS")
    print("=" * 52)
    for k, v in summary.items():
        if v is None:
            print(f"\n{k}")
        else:
            print(f"  {k:<40} {v}")
    print("=" * 52)

    # Per-month cost tradeoff
    df["month"] = df["date"].dt.to_period("M")
    monthly_cost = (
        df.groupby("month")[["cost_point", "cost_q90", "cost_saving"]]
        .sum()
        .reset_index()
    )
    monthly_cost.to_csv("outputs/monthly_cost_tradeoff.csv", index=False)

    # Per-store summary
    store_cost = (
        df.groupby("store")[["cost_point", "cost_q90", "cost_saving", "safety_stock"]]
        .agg({"cost_point": "sum", "cost_q90": "sum",
              "cost_saving": "sum", "safety_stock": "mean"})
        .reset_index()
    )
    store_cost.to_csv("outputs/store_cost_summary.csv", index=False)

    return df, summary


if __name__ == "__main__":
    run()
