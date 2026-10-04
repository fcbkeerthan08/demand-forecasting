"""
feature_engineering.py
-----------------------
Transforms raw (date, store, item, sales) rows into a rich feature matrix.

Features produced
─────────────────
• Calendar  : day_of_week, day_of_month, month, quarter, week_of_year,
              is_weekend, year
• Lag sales : lag_7, lag_14, lag_28, lag_90, lag_365
• Rolling   : roll_mean_7/14/28/91, roll_std_14/28, roll_min_28, roll_max_28
• EWM       : ewm_7, ewm_14
• Seasonality : month_sin/cos, dow_sin/cos (Fourier)
• Holidays  : is_us_holiday (public holidays via the `holidays` library)
• Promotions: is_blackfriday, is_summer_sale, is_xmas_week (rule-based)
• Store/Item: store_id, item_id (label-encoded for tree models)
"""

import warnings
import numpy as np
import pandas as pd

# Optional: US holiday support
try:
    import holidays as hol_lib
    _HAS_HOLIDAYS = True
except ImportError:
    _HAS_HOLIDAYS = False
    warnings.warn("'holidays' package not found; is_us_holiday set to 0.")


# ── Holiday / promotion helpers ────────────────────────────────────────────────

def _make_us_holidays(years):
    if not _HAS_HOLIDAYS:
        return set()
    us = hol_lib.US(years=list(years))
    return set(us.keys())


def _is_promo(date_series: pd.Series) -> pd.DataFrame:
    """Encode promotional windows as binary flags."""
    dt = pd.to_datetime(date_series)
    month  = dt.dt.month
    day    = dt.dt.day
    dow    = dt.dt.dayofweek   # 3 = Thursday

    # Black Friday: 4th Thursday of November + 3 trailing days
    nov_thursdays = (month == 11) & (dow == 3) & (day >= 22) & (day <= 28)

    # Summer sale: last two weeks of July
    summer = (month == 7) & (day >= 18)

    # Christmas week
    xmas = (month == 12) & (day >= 20) & (day <= 26)

    return pd.DataFrame(
        {
            "is_blackfriday": nov_thursdays.astype(int),
            "is_summer_sale": summer.astype(int),
            "is_xmas_week":   xmas.astype(int),
        },
        index=date_series.index,
    )


# ── Core feature builder ───────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns [date, store, item, sales]
         Must be sorted by (store, item, date) for lag computation.

    Returns
    -------
    DataFrame with all engineered features; rows with NaN lags are dropped.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # ── Calendar features ──────────────────────────────────────────────
    dt = df["date"].dt
    df["year"]         = dt.year
    df["month"]        = dt.month
    df["day_of_month"] = dt.day
    df["day_of_week"]  = dt.dayofweek      # 0=Mon … 6=Sun
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["quarter"]      = dt.quarter
    df["is_weekend"]   = (dt.dayofweek >= 5).astype(int)

    # Fourier seasonality features
    df["month_sin"]    = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]    = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]      = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]      = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["woy_sin"]      = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["woy_cos"]      = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # ── Holidays & promotions ──────────────────────────────────────────
    us_holidays = _make_us_holidays(df["year"].unique())
    df["is_us_holiday"] = df["date"].isin(us_holidays).astype(int)
    promo = _is_promo(df["date"])
    df = pd.concat([df, promo], axis=1)

    # ── Lag and rolling features (per store × item group) ─────────────
    grp = df.groupby(["store", "item"])[target_col]

    lag_days = [7, 14, 28, 90, 365]
    for lag in lag_days:
        df[f"lag_{lag}"] = grp.shift(lag)

    rolling_windows = [7, 14, 28, 91]
    for w in rolling_windows:
        rolled = grp.shift(1).rolling(w, min_periods=max(1, w // 2))
        df[f"roll_mean_{w}"] = rolled.mean()
        if w >= 14:
            df[f"roll_std_{w}"]  = rolled.std().fillna(0)
    df["roll_min_28"] = grp.shift(1).rolling(28, min_periods=14).min()
    df["roll_max_28"] = grp.shift(1).rolling(28, min_periods=14).max()

    # Exponentially weighted means
    for span in [7, 14]:
        df[f"ewm_{span}"] = (
            grp.shift(1)
            .transform(lambda s: s.ewm(span=span, min_periods=3).mean())
        )

    # ── Store / item encodings ─────────────────────────────────────────
    df["store_id"] = df["store"].astype("category").cat.codes
    df["item_id"]  = df["item"].astype("category").cat.codes

    # ── Drop rows with missing lags (warm-up period) ───────────────────
    df = df.dropna(subset=[f"lag_{max(lag_days)}"]).reset_index(drop=True)

    return df


FEATURE_COLS = [
    # calendar
    "year", "month", "day_of_month", "day_of_week",
    "week_of_year", "quarter", "is_weekend",
    # Fourier
    "month_sin", "month_cos", "dow_sin", "dow_cos", "woy_sin", "woy_cos",
    # holidays / promotions
    "is_us_holiday", "is_blackfriday", "is_summer_sale", "is_xmas_week",
    # lags
    "lag_7", "lag_14", "lag_28", "lag_90", "lag_365",
    # rolling
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_91",
    "roll_std_14", "roll_std_28",
    "roll_min_28", "roll_max_28",
    # EWM
    "ewm_7", "ewm_14",
    # IDs
    "store_id", "item_id",
]
