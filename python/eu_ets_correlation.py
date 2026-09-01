# -*- coding: utf-8 -*-
"""
eu_ets_correlation.py
Correlate EU ETS carbon price with Pernis NO2/SO2 IESUT residuals (Fig 5.13)

Data sources:
  - EU ETS EUA price: ICAP Allowance Price Explorer (ICAP, 2026)
  - Pernis residuals: IESUT tool output (Albazrkan, 2026)

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.environ.get("IESUT_DATA_DIR", "./data")
EUA_FILE = os.path.join(BASE, "data", "EU_ETS", "eu_ets_eua_price_daily_2018_2026.csv")
NO2_FILE = os.path.join(BASE, "outputs", "IESUT_pernis_no2_table.xls")
SO2_FILE = os.path.join(BASE, "outputs", "IESUT_pernis_so2_table.xls")
OUT_DIR  = os.path.join(BASE, "outputs")

# ── 1. Load and clean EU ETS price data ────────────────────────────────────
print("Loading EU ETS carbon price data...")
raw = pd.read_csv(EUA_FILE, header=None)

# Row 0 = system names, Row 1 = column headers, Row 2+ = data
# "until 2018" system: col 4 = Primary Market price
# "from 2019, download" system: col 9 = Primary Market, col 10 = Secondary Market
data_rows = raw.iloc[2:].copy()
data_rows.columns = range(len(data_rows.columns))

eua_rows = []
for _, row in data_rows.iterrows():
    date_str = str(row[0]).strip()
    try:
        date = pd.to_datetime(date_str)
    except:
        continue

    price = None
    # Try "until 2018" primary market (col 4)
    try:
        v = float(row[4])
        if not np.isnan(v):
            price = v
    except:
        pass

    # Try "from 2019" primary market (col 9)
    if price is None:
        try:
            v = float(row[9])
            if not np.isnan(v):
                price = v
        except:
            pass

    # Try "from 2019" secondary market (col 10)
    if price is None:
        try:
            v = float(row[10])
            if not np.isnan(v):
                price = v
        except:
            pass

    if price is not None:
        eua_rows.append({"date": date, "eua_price_eur": price})

df_eua = pd.DataFrame(eua_rows)
df_eua["year"]  = df_eua["date"].dt.year
df_eua["month"] = df_eua["date"].dt.month

# Monthly mean EUA price
df_eua_monthly = (df_eua.groupby(["year", "month"])["eua_price_eur"]
                  .mean().reset_index()
                  .rename(columns={"eua_price_eur": "eua_price_eur_mean"}))

print(f"  EUA data: {len(df_eua_monthly)} months, "
      f"{df_eua_monthly.year.min()}–{df_eua_monthly.year.max()}")

# ── 2. Load Pernis residuals and aggregate to monthly means ────────────────
print("Loading Pernis IESUT residuals...")

df_no2 = pd.read_excel(NO2_FILE)
df_so2 = pd.read_excel(SO2_FILE)

COVID_YEARS = {2020, 2021}
df_no2 = df_no2[~df_no2["year"].isin(COVID_YEARS)]
df_so2 = df_so2[~df_so2["year"].isin(COVID_YEARS)]

no2_monthly = (df_no2.groupby(["year", "month"])["residual"]
               .mean().reset_index()
               .rename(columns={"residual": "no2_residual_mean"}))

so2_monthly = (df_so2.groupby(["year", "month"])["residual"]
               .mean().reset_index()
               .rename(columns={"residual": "so2_residual_mean"}))

print(f"  NO2 residuals: {len(no2_monthly)} months")
print(f"  SO2 residuals: {len(so2_monthly)} months")

# ── 3. Merge all on year/month ─────────────────────────────────────────────
df = (df_eua_monthly
      .merge(no2_monthly, on=["year", "month"], how="inner")
      .merge(so2_monthly, on=["year", "month"], how="inner"))

# Exclude COVID period (2020-2021) and post-study period (2025+)
df = df[~df["year"].isin([2020, 2021]) & (df["year"] <= 2024)].copy()
df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
df = df.sort_values("date").reset_index(drop=True)

print(f"\nMerged dataset (excl. COVID): {len(df)} months")
print(df[["year","month","eua_price_eur_mean","no2_residual_mean","so2_residual_mean"]].to_string(index=False))

# ── 4. Correlation statistics ──────────────────────────────────────────────
print("\n--- Correlation Statistics ---")

# Pearson
r_no2, p_no2 = stats.pearsonr(df["eua_price_eur_mean"], df["no2_residual_mean"])
r_so2, p_so2 = stats.pearsonr(df["eua_price_eur_mean"], df["so2_residual_mean"])

# Spearman
rho_no2, ps_no2 = stats.spearmanr(df["eua_price_eur_mean"], df["no2_residual_mean"])
rho_so2, ps_so2 = stats.spearmanr(df["eua_price_eur_mean"], df["so2_residual_mean"])

print(f"EUA price vs Pernis NO2 residual:")
print(f"  Pearson  r = {r_no2:.4f}, p = {p_no2:.4f}")
print(f"  Spearman ρ = {rho_no2:.4f}, p = {ps_no2:.4f}")
print(f"EUA price vs Pernis SO2 residual:")
print(f"  Pearson  r = {r_so2:.4f}, p = {p_so2:.4f}")
print(f"  Spearman ρ = {rho_so2:.4f}, p = {ps_so2:.4f}")

# ── 5. Save CSV ────────────────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, "eu_ets_correlation.csv")
df.to_csv(csv_out, index=False)
print(f"\nCSV saved: {csv_out}")
print("\nDone.")
