"""
create_notebook.py
------------------
Generates the Jupyter notebook programmatically.
Run from the demand-forecasting root:
    python src/create_notebook.py
"""
import json
from pathlib import Path

def cell(source, cell_type="code"):
    if cell_type == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }

cells = [

cell("""# 📦 Demand Forecasting with Prediction Intervals
## LightGBM Quantile Regression for Probabilistic Inventory Planning

This notebook walks through the complete pipeline:
1. Dataset overview & EDA
2. Feature engineering
3. Quantile model training
4. Accuracy + calibration evaluation
5. Inventory decision analysis
6. Overconfidence diagnosis
""", "markdown"),

cell("""import sys, os
sys.path.insert(0, 'src')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
print("Environment ready ✓")"""),

cell("""## 1. Dataset Overview""", "markdown"),

cell("""from data_generator import generate_dataset
from pathlib import Path

DATA_PATH = Path('data/train.csv')
if not DATA_PATH.exists():
    df_raw = generate_dataset(output_path=str(DATA_PATH))
else:
    df_raw = pd.read_csv(DATA_PATH, parse_dates=['date'])

print(f"Shape: {df_raw.shape}")
print(f"Date range: {df_raw.date.min().date()} → {df_raw.date.max().date()}")
df_raw.head()"""),

cell("""# Sales distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(df_raw['sales'], bins=100, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('Sales Distribution (all stores/items)')
axes[0].set_xlabel('Daily sales (units)')
axes[0].set_ylabel('Frequency')

# Aggregate daily total sales
daily = df_raw.groupby('date')['sales'].sum()
axes[1].plot(daily.index, daily.rolling(28).mean(), color='#d73027', lw=1.5)
axes[1].fill_between(daily.index, daily.rolling(28).min(), daily.rolling(28).max(),
                      alpha=0.2, color='#d73027')
axes[1].set_title('Total Daily Sales (28-day rolling band)')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Total units sold')

plt.tight_layout()
plt.show()"""),

cell("""## 2. Feature Engineering""", "markdown"),

cell("""from feature_engineering import build_features, FEATURE_COLS

# Use a subset for speed in the notebook (store 1, all items)
df_sub = df_raw[df_raw['store'] == 1].copy()
df_feat = build_features(df_sub)

print(f"Feature columns ({len(FEATURE_COLS)}): {FEATURE_COLS[:10]} ...")
print(f"\\nDataset after feature engineering: {df_feat.shape}")
df_feat[FEATURE_COLS[:8]].describe().round(2)"""),

cell("""# Correlation heatmap for a sample of features
sample_feats = ['lag_7', 'lag_14', 'lag_28', 'roll_mean_7', 'roll_mean_28',
                'ewm_7', 'month', 'day_of_week', 'is_weekend', 'sales']

corr = df_feat[sample_feats].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()"""),

cell("""## 3. Train Quantile Models""", "markdown"),

cell("""from train import load_and_engineer, temporal_split, train_quantile_models, save_models

SPLIT_DATE = '2017-01-01'
QUANTILES  = [0.10, 0.50, 0.90]

# Load full dataset with features
df_full = load_and_engineer(DATA_PATH)
train_df, val_df = temporal_split(df_full, SPLIT_DATE)
print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")"""),

cell("""# Train (takes ~5-15 min depending on hardware)
models = train_quantile_models(train_df, val_df, quantiles=QUANTILES)
save_models(models)
print("\\nModels trained ✓")"""),

cell("""# Generate predictions on validation set
pred_df = val_df.copy()
for q, model in models.items():
    col = f'pred_q{int(q*100):02d}'
    pred_df[col] = np.clip(model.predict(val_df[FEATURE_COLS]), 0, None)

pred_df[['date', 'store', 'item', 'sales', 'pred_q10', 'pred_q50', 'pred_q90']].to_csv(
    'outputs/val_predictions.csv', index=False
)
pred_df.head(3)"""),

cell("""## 4. Forecast Fan Chart""", "markdown"),

cell("""# Pick store=1, item=5 for illustration
s, i = 1, 5
sub = pred_df[(pred_df.store==s) & (pred_df.item==i)].sort_values('date').tail(120)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(sub['date'], sub['pred_q10'], sub['pred_q90'],
                alpha=0.25, color='#4575b4', label='80% PI (q10–q90)')
ax.plot(sub['date'], sub['pred_q50'], color='#d73027', lw=1.8, label='Median (q50)')
ax.scatter(sub['date'], sub['sales'], s=10, color='black', alpha=0.5, label='Actuals')

# Flag promotions
promo = ((sub.date.dt.month==11) & (sub.date.dt.dayofweek==3) & (sub.date.dt.day>=22)) | \\
        ((sub.date.dt.month==7)  & (sub.date.dt.day>=18)) | \\
        ((sub.date.dt.month==12) & (sub.date.dt.day>=20))
ax.scatter(sub.loc[promo,'date'], sub.loc[promo,'sales'],
           s=40, color='#f4a261', zorder=5, label='Promo day')

ax.set_title(f'Demand Forecast – Store {s}, Item {i} (last 120 days)')
ax.set_xlabel('Date'); ax.set_ylabel('Daily Sales (units)')
ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()"""),

cell("""## 5. Evaluation: Point Accuracy + Interval Calibration""", "markdown"),

cell("""from evaluate import evaluate, print_results

results, monthly_cov, overconf = evaluate('outputs/val_predictions.csv')
print_results(results)"""),

cell("""# Monthly calibration plot
months = [str(m) for m in monthly_cov['month']]
fig, ax = plt.subplots(figsize=(13, 4))
ax.bar(months, monthly_cov['coverage_80'], color='#4575b4', alpha=0.7)
ax.axhline(0.80, color='red', lw=2, ls='--', label='Nominal 80%')
ax.set_ylim(0, 1.05)
ax.set_title('Monthly 80% Interval Coverage')
ax.set_ylabel('Coverage fraction')
plt.xticks(rotation=45, ha='right', fontsize=7)
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()"""),

cell("""## 6. Feature Importance""", "markdown"),

cell("""import pickle

model_q50 = models[0.50]

imp = pd.Series(model_q50.feature_importances_, index=model_q50.feature_name_) \\
        .nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
imp.plot.barh(ax=ax, color='#4575b4', edgecolor='white')
ax.set_title('Top-20 Feature Importances (q50 model, gain)')
ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()"""),

cell("""## 7. Inventory Decision Analysis""", "markdown"),

cell("""from inventory_decision import run as run_inventory

inv_df, inv_summary = run_inventory('outputs/val_predictions.csv')"""),

cell("""# Load monthly cost CSV and plot
monthly_cost = pd.read_csv('outputs/monthly_cost_tradeoff.csv')
months = [str(m) for m in monthly_cost['month']]
x = np.arange(len(months)); w = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

ax1.bar(x - w/2, monthly_cost['cost_point'], w, label='Point policy', color='#d73027', alpha=0.8)
ax1.bar(x + w/2, monthly_cost['cost_q90'],   w, label='q90 policy',   color='#1a9641', alpha=0.8)
ax1.set_title('Monthly Inventory Cost: Point vs q90 Policy')
ax1.set_ylabel('Total cost ($)'); ax1.legend(); ax1.grid(axis='y', alpha=0.3)

savings = monthly_cost['cost_saving']
colors  = ['#1a9641' if v >= 0 else '#d73027' for v in savings]
ax2.bar(x, savings, color=colors, alpha=0.85)
ax2.axhline(0, color='black', lw=1)
ax2.set_title('Monthly Cost Saving (Point − q90)')
ax2.set_ylabel('Saving ($)')
plt.xticks(x, months, rotation=45, ha='right', fontsize=7)
plt.tight_layout(); plt.show()"""),

cell("""## 8. Overconfidence Analysis""", "markdown"),

cell("""print("\\nMiss rate by demand context (miss = actual outside [q10, q90]):")
print(overconf.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
df_oc = overconf.sort_values('miss_rate')
colors = ['#d73027' if r > 0.20 else '#4575b4' for r in df_oc['miss_rate']]
ax.barh(df_oc['context'], df_oc['miss_rate'] * 100, color=colors, alpha=0.85)
ax.axvline(20, color='black', lw=1.5, ls='--', label='Nominal 20% miss rate')
ax.set_title('Miss Rate by Context\\n(>20% → model is overconfident)')
ax.set_xlabel('Miss rate (%)')
ax.legend(); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.show()"""),

cell("""## 9. Key Takeaways""", "markdown"),

cell("""summary = {
    'What': [
        'Forecasting model',
        'Quantiles produced',
        'Training data',
        'Validation data',
    ],
    'Detail': [
        'LightGBM with Quantile Loss (pinball)',
        'q10 (lower), q50 (median), q90 (upper)',
        '2013-01-01 to 2016-12-31',
        '2017-01-01 to 2017-12-31',
    ]
}
print(pd.DataFrame(summary).to_string(index=False))
print()
print('Key insight: The q90-based safety stock policy reduces total inventory')
print('cost by targeting the critical ratio = c_under/(c_over+c_under) ≈ 0.89')
print('Overconfidence is highest during promotional spikes (Black Friday, Christmas)')
print('→ production fix: add explicit promo-lift features or ensemble with promo model')"""),

]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("notebooks/demand_forecasting_analysis.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook created: {out}")
