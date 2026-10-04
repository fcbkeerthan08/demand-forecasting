"""
run_pipeline.py
---------------
End-to-end pipeline runner.

Steps
─────
  1. Generate synthetic data  (data/train.csv)
  2. Engineer features
  3. Train quantile models (q10, q50, q90)
  4. Evaluate accuracy + calibration
  5. Run inventory decision analysis
  6. Generate all plots

Run from the project root:
  python src/run_pipeline.py
"""

import os
import sys
import time

# Allow importing siblings
sys.path.insert(0, os.path.dirname(__file__))

from data_generator   import generate_dataset
from train            import load_and_engineer, temporal_split, train_quantile_models, save_models
from evaluate         import evaluate, print_results
from inventory_decision import run as run_inventory
from visualize        import generate_all_plots

import numpy as np
import pandas as pd
from pathlib import Path
from feature_engineering import FEATURE_COLS

TARGET    = "sales"
SPLIT_DATE = "2017-01-01"
QUANTILES  = [0.10, 0.50, 0.90]


def main():
    t0 = time.time()
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  DEMAND FORECASTING PIPELINE WITH PREDICTION INTERVALS  ║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Step 1: Data ──────────────────────────────────────────────────
    print("▶ Step 1/6  Generate synthetic dataset")
    data_path = Path("data/train.csv")
    if data_path.exists():
        print(f"  [skip] {data_path} already exists – delete to regenerate")
    else:
        generate_dataset(output_path=str(data_path))

    # ── Step 2: Features ──────────────────────────────────────────────
    print("\n▶ Step 2/6  Feature engineering")
    df = load_and_engineer(data_path)

    # ── Step 3: Train ─────────────────────────────────────────────────
    print("\n▶ Step 3/6  Train LightGBM quantile models")
    train_df, val_df = temporal_split(df, SPLIT_DATE)
    models = train_quantile_models(train_df, val_df, quantiles=QUANTILES)
    save_models(models)

    # Save validation predictions
    val_df = val_df.copy()
    for q, model in models.items():
        col = f"pred_q{int(q*100):02d}"
        val_df[col] = np.clip(model.predict(val_df[FEATURE_COLS]), 0, None)
    out_path = Path("outputs/val_predictions.csv")
    val_df[["date", "store", "item", TARGET,
            "pred_q10", "pred_q50", "pred_q90"]].to_csv(out_path, index=False)
    print(f"  Saved {len(val_df):,} predictions → {out_path}")

    # ── Step 4: Evaluate ──────────────────────────────────────────────
    print("\n▶ Step 4/6  Evaluate accuracy & calibration")
    results, monthly_cov, overconf = evaluate(str(out_path))
    print_results(results)
    print("\n  Overconfidence by context:")
    print(overconf.to_string(index=False))

    # ── Step 5: Inventory decision ────────────────────────────────────
    print("\n▶ Step 5/6  Inventory decision analysis")
    inv_df, inv_summary = run_inventory(str(out_path))

    # ── Step 6: Visualize ─────────────────────────────────────────────
    print("\n▶ Step 6/6  Generate plots")
    generate_all_plots()

    elapsed = time.time() - t0
    print(f"\n✓ Pipeline complete in {elapsed/60:.1f} min")
    print("  Outputs → outputs/")


if __name__ == "__main__":
    main()
