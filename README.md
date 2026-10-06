# 📦 Demand Forecasting with Prediction Intervals

> **Probabilistic retail demand forecasting using LightGBM Quantile Regression.**  
> Point forecasts hide risk. This project produces calibrated 80% prediction intervals
> and uses them to make better inventory decisions.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Project Structure](#project-structure)
5. [Setup & Installation](#setup--installation)
6. [Reproducing Results](#reproducing-results)
7. [Key Results](#key-results)
8. [Methodology Write-up](#methodology-write-up)
9. [License](#license)

---

## Problem Statement

A retailer needs daily demand forecasts at the **store × item** level to plan inventory.
A single point forecast is insufficient because:
- **Overstocking** costs money in holding, spoilage, and capital lock-up.
- **Stockouts** cost more in lost sales and customer goodwill.

The solution: produce **calibrated prediction intervals** (10th/50th/90th percentile),
translate the upper quantile into a safety-stock policy, and measure the cost benefit
over a naive point-forecast policy.

---

## Dataset

**Synthetic store-item demand data** inspired by the
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) (Kaggle).

- **Scale**: 10 stores × 50 items × 5 years daily (2013–2017) → **~9.1 M rows**
- **Generative model**: multiplicative seasonality + trend + Negative-Binomial noise
- **Promotions embedded**: Black Friday, summer sale, Christmas week

> No Kaggle account required — the dataset is generated on-the-fly by `src/data_generator.py`.

---

## Approach

### Feature Engineering
| Category | Features |
|---|---|
| Calendar | day_of_week, month, quarter, week_of_year, is_weekend |
| Fourier | month_sin/cos, dow_sin/cos, woy_sin/cos |
| Lag sales | lag_7, lag_14, lag_28, lag_90, lag_365 |
| Rolling stats | roll_mean_{7,14,28,91}, roll_std_{14,28}, roll_min/max_28 |
| EWM | ewm_7, ewm_14 |
| Holidays | is_us_holiday (via `holidays` library) |
| Promotions | is_blackfriday, is_summer_sale, is_xmas_week |
| IDs | store_id, item_id (label-encoded) |

### Model
**LightGBM with Quantile Loss** — three separate models trained simultaneously:
- `lgbm_q10.pkl` → 10th percentile (lower bound)
- `lgbm_q50.pkl` → 50th percentile (median / point forecast)
- `lgbm_q90.pkl` → 90th percentile (upper bound)

Training: 2013–2016 | Validation: 2017 (temporal, no leakage)

### Interval Calibration
The 80% prediction interval [q10, q90] is _calibrated_ when ~80% of actuals
fall inside it. We measure **monthly coverage** and report the **Winkler score**
(a proper scoring rule that simultaneously rewards accuracy and width).

### Inventory Decision (Newsvendor Framework)
```
Critical ratio:  r* = c_under / (c_over + c_under)
Safety stock:    SS = q90 - q50  (upper PI half-width)
```

Two policies compared on the 2017 validation year:
| Policy | Order quantity |
|---|---|
| Point-forecast | Q = q50 |
| Quantile (q90) | Q = q90 |

---

## Project Structure

```
demand-forecasting/
├── src/
│   ├── data_generator.py        # Synthetic dataset generation
│   ├── feature_engineering.py   # Feature pipeline (lags, rolling, Fourier)
│   ├── train.py                 # LightGBM quantile training
│   ├── evaluate.py              # Point accuracy + interval calibration
│   ├── inventory_decision.py    # Newsvendor cost analysis
│   ├── visualize.py             # All 8 report plots
│   └── run_pipeline.py          # End-to-end orchestrator
├── notebooks/
│   └── demand_forecasting_analysis.ipynb  # Narrative walkthrough
├── outputs/
│   ├── figures/                 # All plots (auto-generated)
│   ├── models/                  # Trained .pkl models (auto-generated)
│   ├── val_predictions.csv      # Forecast + actuals (auto-generated)
│   ├── monthly_coverage.csv     # Calibration by month
│   └── ...
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start Guide (How to Run)

Follow these step-by-step instructions to set up the project and run the complete forecasting pipeline on your local machine.

### Step 1: Prerequisites
Ensure you have the following installed on your computer:
* **Python 3.8+**: [Download here](https://www.python.org/downloads/) (Make sure to check "Add Python to PATH" during installation)
* **Git**: [Download here](https://git-scm.com/)

### Step 2: Clone the Repository
Open your terminal (Command Prompt, PowerShell, or Terminal) and run:
```bash
git clone https://github.com/<your-username>/demand-forecasting.git
cd demand-forecasting
```

### Step 3: Create a Virtual Environment (Highly Recommended)
Creating a virtual environment keeps this project's dependencies separate from your main system.
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
*(You will know it worked when you see `(venv)` at the beginning of your terminal prompt).*

### Step 4: Install Dependencies
With your virtual environment activated, install all the required Python packages (LightGBM, Pandas, Scikit-learn, etc.):
```bash
pip install -r requirements.txt
```

### Step 5: Run the Full Pipeline
You can run the entire end-to-end pipeline with a single command! This script automatically generates the synthetic data, engineers the features, trains all 3 quantile models, evaluates them, and creates the visual plots.

```bash
python src/run_pipeline.py
```
> ⏳ **Note:** The pipeline processes over 900,000 rows of data. It usually takes between **5 to 15 minutes** to finish depending on your computer's speed. Grab a coffee!

### Step 6: View the Outputs
Once the pipeline says `[DONE]`, all results will be saved in your folder automatically. Navigate to the `outputs/` directory to see what was created:
* 🖼️ **`outputs/figures/`**: Open the `.png` files to see beautifully generated charts (Forecast Fan Chart, Cost Tradeoffs, Feature Importance, etc.)
* 🧠 **`outputs/models/`**: Contains the saved LightGBM models (`.pkl` files).
* 📊 **`outputs/val_predictions.csv`**: A CSV file containing all 182,500 predictions compared against actual sales.

---

## 🔬 Alternative: Explore the Jupyter Notebook

If you prefer an interactive, narrative walkthrough of the code instead of running a single script, we have a Jupyter Notebook prepared for you.

With your virtual environment still activated, run:
```bash
jupyter lab notebooks/demand_forecasting_analysis.ipynb
```
This will open the code in your web browser where you can run it cell by cell and see the charts appear inline.

---

## Key Results

Results from the 2017 validation set (182,500 store×item×day rows).

### Point Accuracy (q50 / median forecast)
| Metric | Value |
|---|---|
| MAE | **19.03 units** |
| RMSE | **29.02 units** |
| MAPE | 38.9 % |
| SMAPE | 32.2 % |

> High MAPE is expected: the dataset spans items with very low base sales where even small absolute errors create large percentage errors.

### Interval Calibration (80% PI = q10 → q90)
| Metric | Value | Target / Interpretation |
|---|---|---|
| **Coverage** | **79.4 %** | ✅ Target = 80 % — nearly perfect calibration |
| Mean PI width | 59.5 units | Adaptive: wider on promo days, narrower on calm days |
| Winkler score | 87.1 | Proper scoring rule (lower = better sharpness + accuracy) |
| Pinball loss q10 | 3.66 | |
| Pinball loss q50 | 9.52 | |
| Pinball loss q90 | 5.05 | |

### Inventory Decision (Newsvendor, c_over=\$0.15, c_under=\$1.20)
| Policy | Total cost (2017) | Stockout rate |
|---|---|---|
| Point forecast (q50) | \$2,642,694 | 50.1 % |
| **Quantile policy (q90)** | **\$1,328,283** | **10.4 %** |
| **Net saving** | **\$1,314,411 (49.7 %)** | |

Safety stock = q90 − q50 = **~34 units/day** on average.

### Overconfidence Findings
| Context | Miss rate | vs. Nominal 20% |
|---|---|---|
| Black Friday | 22.5 % | ⚠️ Overconfident |
| Christmas week | 21.0 % | ⚠️ Slightly overconfident |
| Weekend | 20.8 % | ⚠️ Marginal |
| Normal days | 20.5 % | ✅ Well-calibrated |
| Summer sale | 20.3 % | ✅ Well-calibrated |

The model is marginally overconfident during **promotional demand spikes** (Black Friday, Christmas). Production fix: add explicit promo-lift multiplier features or ensemble with a separate promotion model.

---

## Methodology Write-up

### 1. Why Quantile Regression?

Quantile loss (pinball loss) directly targets a specific quantile of the
conditional distribution without distributional assumptions. Unlike Gaussian
prediction intervals, it handles the **right-skewed, discrete** nature of
retail demand naturally. Three separate models are trained — one per quantile
— which is computationally redundant but maximally flexible: each model can
learn a different optimal tree structure.

### 2. Feature Engineering Decisions

**Lag features** (7, 14, 28, 90, 365 days) capture periodic autocorrelation.
The 365-day lag is particularly powerful: it tells the model "what happened
this day last year", capturing both annual seasonality and item-level trend.

**Rolling statistics** (mean, std, min, max over multiple windows) give the
model a local context of recent demand levels and volatility. The rolling
standard deviation is critical for uncertainty: high-volatility items should
have wider intervals.

**Fourier features** (sin/cos of month, day-of-week, week-of-year) let the
model represent smooth seasonality without overfitting to individual months.

**Promotion flags** are rule-encoded from calendar knowledge. A production
system would ingest actual promotion calendars from the retailer's systems.

### 3. Temporal Validation

Data from 2013–2016 trains the models; 2017 is held out entirely.
This mirrors real-world deployment: you always forecast the future,
never the past. Cross-validation with time-series split (or rolling-origin)
would give tighter variance estimates but was omitted here for clarity.

### 4. Calibration vs. Sharpness

A well-calibrated 80% PI should contain 80% of actuals.
**Overconfident** intervals (coverage < 80%) expose the business to unexpected
stockouts. **Underconfident** intervals (coverage > 80%) lead to excessive
safety stock. The Winkler score balances both: it rewards narrow intervals
but penalises misses, making it a proper scoring rule.

Monthly calibration plots reveal drift: coverage tends to dip in November–December
when promotional demand spikes are hardest to predict precisely.

### 5. Inventory Decision Framework

Using the **newsvendor model** with asymmetric costs:
- Holding cost (`c_over = $0.15/unit/day`) penalises overstock lightly.
- Stockout cost (`c_under = $1.20/unit/day`) penalises stockouts heavily.

The critical ratio `r* ≈ 0.89` means the optimal order quantity is near the
89th percentile of demand — closely aligning with using `q90` as the order
quantity. The quantile policy therefore has a strong theoretical grounding.

The safety stock `SS = q90 − q50` is **adaptive**: it is automatically larger
on high-uncertainty days (promotions, weekends) and smaller on low-uncertainty
days, without any manual tuning. This is the key advantage over fixed
safety-stock rules.

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built with LightGBM, pandas, and matplotlib.*
