"""
visualize.py
------------
Generates all report figures and saves them to outputs/figures/.

Plots produced
──────────────
1. forecast_sample.png       – 90-day forecast window for 1 store/item
                               with fan chart (q10/q50/q90)
2. calibration_curve.png     – monthly coverage vs. nominal 80%
3. error_distribution.png    – histogram of abs errors + RMSE line
4. feature_importance.png    – top-20 LightGBM feature importances (q50 model)
5. cost_tradeoff.png         – monthly cost: point vs. q90 policy
6. safety_stock_dist.png     – distribution of daily safety stock
7. overconfidence_map.png    – miss rate by promotional context (bar chart)
8. pi_width_over_time.png    – rolling mean PI width vs. rolling mean sales
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

FIG_DIR   = Path("outputs/figures")
MODEL_DIR = Path("outputs/models")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"q10": "#4575b4", "q50": "#d73027", "q90": "#1a9641"}
PROMO_COLOUR = "#f4a261"


def _save(name: str, dpi: int = 150):
    path = FIG_DIR / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[viz] Saved → {path}")


# ── 1. Forecast fan chart ──────────────────────────────────────────────────────

def plot_forecast_sample(
    pred_df: pd.DataFrame,
    store: int = 1,
    item: int = 5,
    last_n_days: int = 120,
):
    sub = pred_df[(pred_df["store"] == store) & (pred_df["item"] == item)].copy()
    sub = sub.sort_values("date").tail(last_n_days)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Shaded PI
    ax.fill_between(sub["date"], sub["pred_q10"], sub["pred_q90"],
                    alpha=0.25, color=PALETTE["q10"], label="80% PI (q10–q90)")

    ax.plot(sub["date"], sub["pred_q50"],
            color=PALETTE["q50"], lw=1.8, label="Median (q50)")
    ax.scatter(sub["date"], sub["sales"],
               s=10, color="black", alpha=0.5, label="Actuals")

    # Mark promotional days
    promo_flag = (
        ((sub["date"].dt.month == 11) & (sub["date"].dt.dayofweek == 3) &
         (sub["date"].dt.day >= 22)) |
        ((sub["date"].dt.month == 7) & (sub["date"].dt.day >= 18)) |
        ((sub["date"].dt.month == 12) & (sub["date"].dt.day >= 20))
    )
    ax.scatter(sub.loc[promo_flag, "date"], sub.loc[promo_flag, "sales"],
               s=30, color=PROMO_COLOUR, zorder=5, label="Promo day")

    ax.set_title(f"Demand Forecast – Store {store}, Item {item}\n"
                 f"(last {last_n_days} days)", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Sales (units)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save("forecast_sample.png")


# ── 2. Calibration curve ───────────────────────────────────────────────────────

def plot_calibration(monthly_cov: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 4))
    months = [str(m) for m in monthly_cov["month"]]
    cov    = monthly_cov["coverage_80"].values

    ax.bar(months, cov, color="#4575b4", alpha=0.7, label="Actual coverage")
    ax.axhline(0.80, color="red", lw=2, ls="--", label="Nominal 80%")
    ax.set_ylim(0, 1.05)
    ax.set_title("Monthly Interval Coverage (80% PI)", fontsize=13)
    ax.set_xlabel("Month"); ax.set_ylabel("Coverage fraction")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save("calibration_curve.png")


# ── 3. Error distribution ──────────────────────────────────────────────────────

def plot_error_distribution(pred_df: pd.DataFrame):
    errors = (pred_df["sales"] - pred_df["pred_q50"]).values
    rmse   = np.sqrt(np.mean(errors ** 2))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(errors, bins=80, color="#2c7bb6", edgecolor="white",
            alpha=0.8, density=True)
    ax.axvline(0,    color="black", lw=1.5, ls="-",  label="Zero error")
    ax.axvline(rmse, color="red",   lw=1.5, ls="--", label=f"RMSE = {rmse:.2f}")
    ax.axvline(-rmse,color="red",   lw=1.5, ls="--")
    ax.set_title("Residual Distribution (Actual − q50 Forecast)", fontsize=13)
    ax.set_xlabel("Error (units)"); ax.set_ylabel("Density")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save("error_distribution.png")


# ── 4. Feature importance ──────────────────────────────────────────────────────

def plot_feature_importance(model, top_n: int = 20):
    imp = pd.Series(
        model.feature_importances_,
        index=model.feature_name_
    ).nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot.barh(ax=ax, color="#4575b4", edgecolor="white")
    ax.set_title(f"Top-{top_n} Feature Importances (q50 model)", fontsize=13)
    ax.set_xlabel("Gain importance"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save("feature_importance.png")


# ── 5. Cost tradeoff ───────────────────────────────────────────────────────────

def plot_cost_tradeoff(monthly_cost: pd.DataFrame):
    months = [str(m) for m in monthly_cost["month"]]
    x = np.arange(len(months))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    # Panel 1: absolute costs
    ax1.bar(x - w/2, monthly_cost["cost_point"], w, label="Point policy",
            color="#d73027", alpha=0.8)
    ax1.bar(x + w/2, monthly_cost["cost_q90"],   w, label="q90 policy",
            color="#1a9641", alpha=0.8)
    ax1.set_title("Monthly Inventory Cost: Point vs q90 Policy", fontsize=13)
    ax1.set_ylabel("Total cost ($)"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: savings
    colors = ["#1a9641" if v >= 0 else "#d73027"
              for v in monthly_cost["cost_saving"]]
    ax2.bar(x, monthly_cost["cost_saving"], color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    ax2.set_title("Monthly Cost Saving (Point − q90)", fontsize=12)
    ax2.set_ylabel("Saving ($)"); ax2.set_xlabel("Month")
    ax2.grid(axis="y", alpha=0.3)
    plt.xticks(x, months, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    _save("cost_tradeoff.png")


# ── 6. Safety stock distribution ──────────────────────────────────────────────

def plot_safety_stock(inv_df: pd.DataFrame):
    ss = inv_df["safety_stock"].clip(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(ss, bins=60, color="#74add1", edgecolor="white", alpha=0.85)
    ax.axvline(ss.mean(), color="red", lw=2, ls="--",
               label=f"Mean = {ss.mean():.1f} units")
    ax.set_title("Distribution of Daily Safety Stock (q90 − q50)", fontsize=13)
    ax.set_xlabel("Safety stock (units)"); ax.set_ylabel("Frequency")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save("safety_stock_dist.png")


# ── 7. Overconfidence map ──────────────────────────────────────────────────────

def plot_overconfidence(overconf_df: pd.DataFrame):
    df = overconf_df.sort_values("miss_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#d73027" if r > 0.20 else "#4575b4" for r in df["miss_rate"]]
    ax.barh(df["context"], df["miss_rate"] * 100, color=colors, alpha=0.85)
    ax.axvline(20, color="black", lw=1.5, ls="--", label="Nominal 20% miss")
    ax.set_title("Miss Rate by Demand Context\n"
                 "(Miss rate > 20% → interval too narrow)", fontsize=13)
    ax.set_xlabel("Miss rate (%)"); ax.legend(); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save("overconfidence_map.png")


# ── 8. PI width vs. sales over time ───────────────────────────────────────────

def plot_pi_width_over_time(pred_df: pd.DataFrame):
    daily = (
        pred_df.groupby("date")
        .agg(
            mean_sales=("sales",    "mean"),
            mean_q10  =("pred_q10", "mean"),
            mean_q90  =("pred_q90", "mean"),
        )
        .reset_index()
    )
    daily["mean_width"] = daily["mean_q90"] - daily["mean_q10"]

    roll = daily.set_index("date").rolling("28D")
    daily["roll_sales"] = roll["mean_sales"].mean().values
    daily["roll_width"] = roll["mean_width"].mean().values

    fig, ax = plt.subplots(figsize=(13, 4))
    ax2 = ax.twinx()
    ax.plot(daily["date"], daily["roll_sales"], color="#d73027", lw=1.5,
            label="28-day avg sales")
    ax2.plot(daily["date"], daily["roll_width"], color="#4575b4", lw=1.5,
             ls="--", label="28-day avg PI width")
    ax.set_title("Rolling PI Width vs. Demand Level (28-day window)", fontsize=13)
    ax.set_xlabel("Date"); ax.set_ylabel("Avg daily sales (units)")
    ax2.set_ylabel("Avg PI width (units)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save("pi_width_over_time.png")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def generate_all_plots():
    pred_df  = pd.read_csv("outputs/val_predictions.csv", parse_dates=["date"])
    pred_df["pred_q10"] = pred_df["pred_q10"].clip(0)
    pred_df["pred_q50"] = pred_df["pred_q50"].clip(0)
    pred_df["pred_q90"] = pred_df["pred_q90"].clip(0)

    monthly_cov  = pd.read_csv("outputs/monthly_coverage.csv")
    monthly_cost = pd.read_csv("outputs/monthly_cost_tradeoff.csv")
    overconf     = pd.read_csv("outputs/overconfidence_analysis.csv")
    inv_df       = pd.read_csv("outputs/val_predictions.csv", parse_dates=["date"])

    from inventory_decision import compute_safety_stock
    inv_df = compute_safety_stock(inv_df)

    # Load q50 model for feature importance
    model_q50 = None
    model_path = MODEL_DIR / "lgbm_q50.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            model_q50 = pickle.load(f)

    print("[viz] Generating plots …")
    plot_forecast_sample(pred_df, store=1, item=5)
    plot_calibration(monthly_cov)
    plot_error_distribution(pred_df)
    if model_q50:
        plot_feature_importance(model_q50)
    plot_cost_tradeoff(monthly_cost)
    plot_safety_stock(inv_df)
    plot_overconfidence(overconf)
    plot_pi_width_over_time(pred_df)
    print("[viz] All plots saved to outputs/figures/")


if __name__ == "__main__":
    generate_all_plots()
