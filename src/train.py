"""
train.py
--------
Trains three LightGBM quantile models (q=0.10, 0.50, 0.90) on the
engineered feature matrix.

Training strategy
─────────────────
• Temporal split  : train on 2013–2016, validate on 2017
• LightGBM with   : objective='quantile', alpha=q
• Early stopping  : patience=50 rounds on validation set
• Models saved    : outputs/models/lgbm_q{q}.pkl
"""

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_engineering import build_features, FEATURE_COLS

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

QUANTILES = [0.10, 0.50, 0.90]
TARGET    = "sales"
SPLIT_DATE = "2017-01-01"          # everything before this → train
MODEL_DIR  = Path("outputs/models")
DATA_PATH  = Path("data/train.csv")

LGBM_PARAMS_BASE = dict(
    boosting_type    = "gbdt",
    n_estimators     = 2000,
    learning_rate    = 0.05,
    num_leaves       = 64,
    min_child_samples= 50,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq     = 5,
    lambda_l1        = 0.1,
    lambda_l2        = 0.1,
    verbose          = -1,
    n_jobs           = -1,
    random_state     = 42,
)


def load_and_engineer(data_path: Path) -> pd.DataFrame:
    print(f"[train] Loading data from {data_path} …")
    df = pd.read_csv(data_path, parse_dates=["date"])
    print(f"[train] Raw rows: {len(df):,}")
    df = build_features(df, target_col=TARGET)
    print(f"[train] After feature engineering: {len(df):,} rows")
    return df


def temporal_split(df: pd.DataFrame, split_date: str):
    train = df[df["date"] < split_date].copy()
    val   = df[df["date"] >= split_date].copy()
    print(f"[train] Train: {len(train):,} rows  |  Val: {len(val):,} rows")
    return train, val


def train_quantile_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    quantiles=QUANTILES,
) -> dict:
    X_tr = train[FEATURE_COLS]
    y_tr = train[TARGET].values
    X_val = val[FEATURE_COLS]
    y_val = val[TARGET].values

    models = {}
    for q in quantiles:
        print(f"\n[train] ── Training q={q:.2f} model ──")
        params = {**LGBM_PARAMS_BASE, "objective": "quantile", "alpha": q}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
        models[q] = model
        print(f"[train] Best iteration: {model.best_iteration_}")

    return models


def save_models(models: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for q, model in models.items():
        path = MODEL_DIR / f"lgbm_q{int(q*100):02d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[train] Saved → {path}")


def main():
    df = load_and_engineer(DATA_PATH)
    train, val = temporal_split(df, SPLIT_DATE)
    models = train_quantile_models(train, val)
    save_models(models)

    # Save validation predictions for evaluation
    val = val.copy()
    for q, model in models.items():
        col = f"pred_q{int(q*100):02d}"
        val[col] = np.clip(model.predict(val[FEATURE_COLS]), 0, None)

    out_path = Path("outputs/val_predictions.csv")
    val[["date", "store", "item", TARGET, "pred_q10", "pred_q50", "pred_q90"]].to_csv(
        out_path, index=False
    )
    print(f"\n[train] Validation predictions saved → {out_path}")
    return models, val


if __name__ == "__main__":
    main()
