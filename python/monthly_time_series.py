"""
Generates the monthly NO2 residual time series and seasonal decomposition
(Figure 7) for Doura and Pernis.

Uses a classical additive decomposition (centered 12-month moving-average
trend; seasonal component = mean deviation from trend per calendar month,
averaged across all available years; residual = observed - trend - seasonal).
No external decomposition library required.

Run: python monthly_time_series.py
Output: dissertation_final/outputs/monthly_time_series_decomposition.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.environ.get("IESUT_DATA_DIR", "./data"), "outputs")
DOURA_XLS = os.path.join(OUTPUT_DIR, "IESUT_doura_no2_table.xls")
PERNIS_XLS = os.path.join(OUTPUT_DIR, "IESUT_pernis_no2_table.xls")


def monthly_series(path):
    df = pd.read_excel(path)
    monthly = df.groupby(["year", "month"])["residual"].mean().reset_index()
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    monthly = monthly.sort_values("date").reset_index(drop=True)
    return monthly


def break_at_gaps(df, date_col="date", gap_days=60):
    """Insert a NaN row wherever consecutive dates are more than gap_days apart,
    so matplotlib draws a visual break instead of connecting across missing periods."""
    df = df.sort_values(date_col).reset_index(drop=True)
    rows = [df.iloc[0:1]]
    for i in range(1, len(df)):
        gap = (df[date_col].iloc[i] - df[date_col].iloc[i - 1]).days
        if gap > gap_days:
            blank = df.iloc[i:i + 1].copy()
            for col in blank.columns:
                if col != date_col:
                    blank[col] = np.nan
            blank[date_col] = df[date_col].iloc[i - 1] + pd.Timedelta(days=gap // 2)
            rows.append(blank)
        rows.append(df.iloc[i:i + 1])
    return pd.concat(rows, ignore_index=True)


def classical_decompose(monthly, period=12):
    y = monthly["residual"].values.astype(float)
    n = len(y)

    # Centered moving-average trend (window = period, then a 2-point MA if period is even)
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        window = y[i - half:i + half + 1] if period % 2 == 1 else y[i - half:i + half]
        trend[i] = np.nanmean(window)

    detrended = y - trend
    monthly = monthly.copy()
    monthly["trend"] = trend
    monthly["detrended"] = detrended

    seasonal_by_month = monthly.groupby("month")["detrended"].mean()
    seasonal_by_month = seasonal_by_month - seasonal_by_month.mean()  # centre around 0
    monthly["seasonal"] = monthly["month"].map(seasonal_by_month)
    monthly["resid"] = y - monthly["trend"] - monthly["seasonal"]

    return monthly


doura = classical_decompose(monthly_series(DOURA_XLS))
pernis = classical_decompose(monthly_series(PERNIS_XLS))

doura_plot = break_at_gaps(doura)
pernis_plot = break_at_gaps(pernis)

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=False)

ax_obs, ax_trend, ax_seas, ax_resid = axes

ax_obs.plot(doura_plot["date"], doura_plot["residual"], color="#d73027", linewidth=1.3, label="Doura NO2")
ax_obs.plot(pernis_plot["date"], pernis_plot["residual"], color="#4575b4", linewidth=1.3, label="Pernis NO2")
ax_obs.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax_obs.set_ylabel("Observed\nresidual (mol/m2)")
ax_obs.legend(loc="upper left", fontsize=9)
ax_obs.grid(alpha=0.3)

ax_trend.plot(doura_plot["date"], doura_plot["trend"], color="#d73027", linewidth=1.6)
ax_trend.plot(pernis_plot["date"], pernis_plot["trend"], color="#4575b4", linewidth=1.6)
ax_trend.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax_trend.set_ylabel("Trend\n(12-month centred MA)")
ax_trend.grid(alpha=0.3)

month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
doura_seas = doura.drop_duplicates("month").sort_values("month")
pernis_seas = pernis.drop_duplicates("month").sort_values("month")
width = 0.35
x = np.arange(1, 13)
ax_seas.bar(x - width/2, doura_seas.set_index("month")["seasonal"].reindex(x), width=width,
            color="#d73027", alpha=0.8, label="Doura NO2")
ax_seas.bar(x + width/2, pernis_seas.set_index("month")["seasonal"].reindex(x), width=width,
            color="#4575b4", alpha=0.8, label="Pernis NO2")
ax_seas.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax_seas.set_xticks(x)
ax_seas.set_xticklabels(month_labels)
ax_seas.set_ylabel("Seasonal\ncomponent (mol/m2)")
ax_seas.grid(alpha=0.3)

ax_resid.scatter(doura["date"], doura["resid"], color="#d73027", s=10, alpha=0.7)
ax_resid.scatter(pernis["date"], pernis["resid"], color="#4575b4", s=10, alpha=0.7)
ax_resid.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax_resid.set_ylabel("Residual\n(unexplained)")
ax_resid.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "monthly_time_series_decomposition.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
