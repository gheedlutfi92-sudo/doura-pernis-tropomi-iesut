# -*- coding: utf-8 -*-
"""
rivm_pernis_comparison.py
Convert TROPOMI Pernis NO2/SO2 residuals to surface concentrations using
ERA5 PBLH, then compare against RIVM ground station NL01484 (Zwartewaalstraat,
Pernis) — the official monitoring station adjacent to Shell Pernis refinery.

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = r"C:\Users\HP\Desktop\proposal& dissertation\dissertation_final"
PBLH_NC  = os.path.join(BASE, "data", "ERA5", "era5_pblh_monthly.nc")
RIVM_DIR = os.path.join(BASE, "data", "Ground_Stations", "Rotterdam_RIVM")
GEE_CSV  = os.path.join(BASE, "data", "TROPOMI_GEE_Exports", "pernis_monthly_samples_v33.csv")
OUT_DIR  = os.path.join(BASE, "outputs")

# ── Constants ──────────────────────────────────────────────────────────────
MW_NO2 = 46.0   # g/mol
MW_SO2 = 64.0   # g/mol

# Shell Pernis refinery centroid (verified from Google Maps)
PERNIS_LAT = 51.888
PERNIS_LON = 4.334

# Primary RIVM ground stations near Shell Pernis
RIVM_NO2_STATION = "NL01485"  # Rotterdam-Hoogvliet (~2km from Pernis) — NO2
RIVM_SO2_STATION = "NL01484"  # Zwartewaalstraat, Pernis — SO2 dedicated station

# Study years (COVID 2020-2021 excluded)
STUDY_YEARS = [2018, 2019, 2022, 2023, 2024]

# ── 1. Extract Pernis PBLH from ERA5 ──────────────────────────────────────
print("Loading ERA5 PBLH for Pernis...")
ds = xr.open_dataset(PBLH_NC)
pblh_pernis = ds["blh"].sel(
    latitude=PERNIS_LAT, longitude=PERNIS_LON, method="nearest"
).to_dataframe().reset_index()

pblh_pernis["year"]  = pblh_pernis["valid_time"].dt.year
pblh_pernis["month"] = pblh_pernis["valid_time"].dt.month
pblh_pernis = pblh_pernis[["year", "month", "blh"]].rename(columns={"blh": "pblh_m"})
pblh_pernis = pblh_pernis[pblh_pernis["year"].isin(STUDY_YEARS)]
print(f"  PBLH months available: {len(pblh_pernis)}")
print(f"  Mean PBLH Pernis: {pblh_pernis['pblh_m'].mean():.0f} m")

# ── 2. Load TROPOMI Pernis data and aggregate to monthly means ────────────
print("\nLoading Pernis data from final GEE export...")

gee = pd.read_csv(GEE_CSV)
gee = gee[gee["year"].isin(STUDY_YEARS)]

def monthly_mean(gee_df, col):
    monthly = gee_df.groupby(["year", "month"])[col].mean().reset_index()
    monthly.columns = ["year", "month", f"{col}_col"]
    return monthly

no2_monthly = monthly_mean(gee, "no2")
so2_monthly = monthly_mean(gee, "so2")
print(f"  NO2 months: {len(no2_monthly)}, SO2 months: {len(so2_monthly)}")

# Merge with PBLH
no2_merged = no2_monthly.merge(pblh_pernis, on=["year", "month"], how="inner")
so2_merged = so2_monthly.merge(pblh_pernis, on=["year", "month"], how="inner")

# ── 3. Convert mol/m2 → µg/m3 ─────────────────────────────────────────────
# surface_conc (µg/m3) = VCD (mol/m2) / PBLH (m) * MW (g/mol) * 1e6
no2_merged["no2_surface_ugm3"] = (no2_merged["no2_col"] / no2_merged["pblh_m"]) * MW_NO2 * 1e6
so2_merged["so2_surface_ugm3"] = (so2_merged["so2_col"] / so2_merged["pblh_m"]) * MW_SO2 * 1e6

print(f"\nTROPOMI-derived Pernis NO2 (µg/m3): mean={no2_merged['no2_surface_ugm3'].mean():.2f}")
print(f"TROPOMI-derived Pernis SO2 (µg/m3): mean={so2_merged['so2_surface_ugm3'].mean():.2f}")

# ── 4. Load RIVM ground station data ──────────────────────────────────────
print(f"\nLoading RIVM data (NO2: {RIVM_NO2_STATION}, SO2: {RIVM_SO2_STATION})...")

rivm_rows = []
for year in STUDY_YEARS:
    for pollutant, col_name, station in [
        ("NO2", "no2_ground_ugm3", RIVM_NO2_STATION),
        ("SO2", "so2_ground_ugm3", RIVM_SO2_STATION),
    ]:
        fpath = os.path.join(RIVM_DIR, f"{year}_{pollutant}.csv")
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}")
            continue
        df = pd.read_csv(fpath, sep=";", comment="#", low_memory=False)
        df = df[df["meetlocatie_id"] == station].copy()
        if df.empty:
            continue
        df["waarde"] = pd.to_numeric(df["waarde"], errors="coerce")
        df["dt"] = pd.to_datetime(df["begindatumtijd"], utc=True, errors="coerce")
        df = df.dropna(subset=["dt", "waarde"])
        df = df[df["waarde"] >= 0]
        df["year"]  = df["dt"].dt.year
        df["month"] = df["dt"].dt.month
        # Filter to the intended year only (UTC+1 can cause Dec to appear in next year)
        df = df[df["year"] == year]
        monthly = df.groupby(["year", "month"])["waarde"].mean().reset_index()
        monthly.rename(columns={"waarde": col_name}, inplace=True)
        rivm_rows.append(monthly)
        print(f"  {year} {pollutant}: {len(monthly)} months")

# Merge NO2 and SO2 ground data
rivm_no2 = pd.concat([r for r in rivm_rows if "no2_ground_ugm3" in r.columns], ignore_index=True)
rivm_so2 = pd.concat([r for r in rivm_rows if "so2_ground_ugm3" in r.columns], ignore_index=True)

# ── 5. Merge TROPOMI converted + RIVM ground ───────────────────────────────
compare_no2 = no2_merged.merge(rivm_no2, on=["year", "month"], how="inner")
compare_so2 = so2_merged.merge(rivm_so2, on=["year", "month"], how="inner")

compare_no2["date"] = pd.to_datetime(compare_no2[["year","month"]].assign(day=1))
compare_so2["date"] = pd.to_datetime(compare_so2[["year","month"]].assign(day=1))
compare_no2 = compare_no2.sort_values("date")
compare_so2 = compare_so2.sort_values("date")

print(f"\nNO2 overlap months: {len(compare_no2)}")
print(f"SO2 overlap months: {len(compare_so2)}")

# ── 6. Print summary table ─────────────────────────────────────────────────
print("\nNO2 Comparison (TROPOMI vs RIVM):")
print(compare_no2[["year","month","no2_surface_ugm3","no2_ground_ugm3"]].to_string(index=False))
print("\nSO2 Comparison (TROPOMI vs RIVM):")
print(compare_so2[["year","month","so2_surface_ugm3","so2_ground_ugm3"]].to_string(index=False))

# ── 7. Correlation stats ───────────────────────────────────────────────────
from scipy import stats

def corr_stats(x, y, label):
    if len(x) < 3:
        print(f"{label}: not enough data (n={len(x)})")
        return
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    print(f"{label}: Pearson r={r:.3f} (p={p:.3f}), Spearman ρ={rho:.3f} (p={ps:.3f}), n={len(x)}")

print("\n--- Correlation Statistics ---")
if len(compare_no2) >= 3:
    corr_stats(compare_no2["no2_surface_ugm3"], compare_no2["no2_ground_ugm3"], "NO2 TROPOMI vs RIVM")
if len(compare_so2) >= 3:
    corr_stats(compare_so2["so2_surface_ugm3"], compare_so2["so2_ground_ugm3"], "SO2 TROPOMI vs RIVM")

# ── 8. Save comparison CSV ─────────────────────────────────────────────────
combined = compare_no2[["year","month","no2_surface_ugm3","no2_ground_ugm3","pblh_m"]].merge(
    compare_so2[["year","month","so2_surface_ugm3","so2_ground_ugm3"]],
    on=["year","month"], how="outer"
)
csv_out = os.path.join(OUT_DIR, "rivm_pernis_comparison.csv")
combined.to_csv(csv_out, index=False)
print(f"Comparison CSV saved: {csv_out}")
print("\nDone.")
