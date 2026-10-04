"""
data_generator.py
-----------------
Generates a realistic synthetic retail demand dataset inspired by the
Store Item Demand Forecasting Challenge (Kaggle).

10 stores × 50 items × 5 years of daily sales (2013-01-01 to 2017-12-31)
with multiplicative seasonality, trend, store/item fixed effects,
promotional spikes, and Negative-Binomial noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)


def _promotion_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    """Return 1 where a date is inside a promotional window, else 0."""
    mask = np.zeros(len(dates), dtype=float)
    # Black Friday (4th Thursday of November + 3 days)
    for year in dates.year.unique():
        nov = dates[(dates.year == year) & (dates.month == 11)]
        thursdays = nov[nov.dayofweek == 3]
        if len(thursdays) >= 4:
            bf = thursdays[3]
            bf_range = pd.date_range(bf, periods=4, freq="D")
            mask[dates.isin(bf_range)] = 1.5  # 150 % lift

        # Summer sale (last two weeks of July)
        summer_start = pd.Timestamp(year=year, month=7, day=18)
        summer_end = pd.Timestamp(year=year, month=7, day=31)
        mask[
            (dates >= summer_start) & (dates <= summer_end)
        ] = 0.6  # 60 % lift

        # Holiday week Christmas
        xmas_start = pd.Timestamp(year=year, month=12, day=20)
        xmas_end = pd.Timestamp(year=year, month=12, day=26)
        mask[
            (dates >= xmas_start) & (dates <= xmas_end)
        ] = 1.0  # 100 % lift
    return mask


def generate_dataset(
    output_path: str = "data/train.csv",
    n_stores: int = 10,
    n_items: int = 50,
    start: str = "2013-01-01",
    end: str = "2017-12-31",
) -> pd.DataFrame:
    """Generate and save the synthetic dataset; return the DataFrame."""
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)

    # ── Shared time signals ────────────────────────────────────────────
    t = np.arange(n_days) / 365.25          # fractional years
    trend = 1 + 0.05 * t                    # 5 % YoY growth
    weekly = 1 + 0.15 * np.sin(2 * np.pi * t * 52)
    annual = 1 + 0.25 * np.sin(2 * np.pi * t - np.pi / 2)
    promo  = 1 + _promotion_mask(dates)

    rows = []
    for store in range(1, n_stores + 1):
        store_eff = RNG.uniform(0.6, 1.4)          # store size multiplier
        for item in range(1, n_items + 1):
            item_base = RNG.uniform(5, 80)          # base daily sales
            item_seasonal_amp = RNG.uniform(0.8, 1.2)

            # Demand signal
            mu = (
                item_base
                * store_eff
                * trend
                * (1 + (item_seasonal_amp - 1) * (weekly - 1))
                * annual
                * promo
            )
            mu = np.clip(mu, 1, None)

            # Negative-Binomial noise (overdispersed counts)
            r = RNG.uniform(3, 15, size=n_days)    # dispersion per day
            p = r / (r + mu)
            sales = RNG.negative_binomial(r, p).astype(int)

            for i, d in enumerate(dates):
                rows.append(
                    {
                        "date": d,
                        "store": store,
                        "item": item,
                        "sales": sales[i],
                    }
                )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[data_generator] Saved {len(df):,} rows → {output_path}")
    return df


if __name__ == "__main__":
    generate_dataset()
