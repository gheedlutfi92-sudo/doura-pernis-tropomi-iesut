# -*- coding: utf-8 -*-
"""
tropomi_surface_conversion.py
Convert TROPOMI NO2/SO2 column density (mol/m2) to surface concentration
(ug/m3) using ERA5 Planetary Boundary Layer Height, then compare against
Ministry of Environment ground station data (Al-Saidia, Baghdad).

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import pandas as pd
import numpy as np
import xarray as xr
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.environ.get("IESUT_DATA_DIR", "./data")
GEE_CSV   = os.path.join(BASE, "data", "TROPOMI_GEE_Exports", "doura_monthly_samples_v33.csv")
PBLH_NC   = os.path.join(BASE, "data", "ERA5", "era5_pblh_monthly.nc")
MOE_DIR   = os.path.join(BASE, "data", "Ground_Stations", "Baghdad_Ministry_of_Environment")
OUT_DIR   = os.path.join(BASE, "outputs")

# ── Constants ─────────────────────────────────────────────────────────────────
MW_NO2 = 46.0   # g/mol
MW_SO2 = 64.0   # g/mol
R_GAS  = 24.045 # molar volume at 25 degC, 1 atm (L/mol)

# Al-Daura Refinery centroid (verified from Google Maps)
DOURA_LAT = 33.264
DOURA_LON = 44.427

# ── Step 1: Load ERA5 PBLH and extract Doura monthly values ───────────────────
print("Loading ERA5 PBLH...")
ds = xr.open_dataset(PBLH_NC)

# Extract nearest grid point to Doura
pblh_doura = ds["blh"].sel(
    latitude=DOURA_LAT, longitude=DOURA_LON, method="nearest"
).to_dataframe().reset_index()

pblh_doura["year"]  = pblh_doura["valid_time"].dt.year
pblh_doura["month"] = pblh_doura["valid_time"].dt.month
pblh_doura = pblh_doura[["year", "month", "blh"]].rename(columns={"blh": "pblh_m"})
print(f"  PBLH extracted: {len(pblh_doura)} months")
print(f"  Mean PBLH Doura: {pblh_doura['pblh_m'].mean():.0f} m")

# ── Step 2: Load TROPOMI monthly means for Doura (spatial mean per month) ─────
print("\nLoading TROPOMI data from final GEE export...")

gee = pd.read_csv(GEE_CSV)
gee = gee[~gee["year"].isin([2020, 2021])]  # exclude COVID years

def monthly_mean(gee_df, pollutant):
    monthly = gee_df.groupby(["year", "month"])[pollutant].mean().reset_index()
    monthly.columns = ["year", "month", f"{pollutant}_col"]
    return monthly

no2_monthly = monthly_mean(gee, "no2")
so2_monthly = monthly_mean(gee, "so2")

# Merge with PBLH
no2_merged = no2_monthly.merge(pblh_doura, on=["year", "month"], how="inner")
so2_merged = so2_monthly.merge(pblh_doura, on=["year", "month"], how="inner")

# ── Step 3: Convert mol/m2 -> ug/m3 ──────────────────────────────────────────
# surface_conc (ug/m3) = VCD (mol/m2) / PBLH (m) * MW (g/mol) * 1e6 (ug/g)
no2_merged["no2_surface_ugm3"] = (no2_merged["no2_col"] / no2_merged["pblh_m"]) * MW_NO2 * 1e6
so2_merged["so2_surface_ugm3"] = (so2_merged["so2_col"] / so2_merged["pblh_m"]) * MW_SO2 * 1e6
print("\nTROPOMI converted NO2 surface concentration (ug/m3):")
print(no2_merged[["year","month","no2_surface_ugm3"]].describe())

# ── Step 4: Load Ministry of Environment monthly station data ─────────────────
print("\nLoading Ministry of Environment data...")

MOE_MONTHS_AR = {
    "كانون الثاني": 1, "كانون 2": 1,
    "شباط": 2,
    "آذار": 3, "اذار": 3,
    "نيسان": 4,
    "ايار": 5, "آيار": 5,
    "حزيران": 6,
    "تموز": 7,
    "اب": 8, "آب": 8,
    "ايلول": 9, "آيلول": 9,
    "تشرين الاول": 10, "تشرين 1": 10,
    "تشرين الثاني": 11, "تشرين 2": 11,
    "كانون الاول": 12, "كانون 1": 12,
}

moe_rows = []
for year in [2018, 2019, 2022, 2023, 2024]:
    fpath = os.path.join(MOE_DIR, f"بيانات {year}.xlsx")
    df = pd.read_excel(fpath, header=None)
    for _, row in df.iterrows():
        month_ar = str(row.iloc[0]).strip()
        if month_ar in MOE_MONTHS_AR:
            try:
                no2_ppm = float(row.iloc[3])
                so2_ppm = float(row.iloc[5])
                # Convert ppm -> ug/m3
                no2_ugm3 = no2_ppm * (MW_NO2 / R_GAS) * 1000
                so2_ugm3 = so2_ppm * (MW_SO2 / R_GAS) * 1000
                moe_rows.append({
                    "year": year, "month": MOE_MONTHS_AR[month_ar],
                    "no2_ground_ugm3": no2_ugm3,
                    "so2_ground_ugm3": so2_ugm3,
                })
            except (ValueError, TypeError):
                pass

moe_df = pd.DataFrame(moe_rows)
print(f"  Ministry data loaded: {len(moe_df)} months")

# ── Step 5: Merge TROPOMI converted + ground station ──────────────────────────
compare_no2 = no2_merged.merge(moe_df[["year","month","no2_ground_ugm3"]], on=["year","month"], how="inner")
compare_so2 = so2_merged.merge(moe_df[["year","month","so2_ground_ugm3"]], on=["year","month"], how="inner")

print("\nNO2 Comparison (TROPOMI converted vs Ground):")
print(compare_no2[["year","month","no2_surface_ugm3","no2_ground_ugm3"]].to_string(index=False))

print("\nSO2 Comparison (TROPOMI converted vs Ground):")
print(compare_so2[["year","month","so2_surface_ugm3","so2_ground_ugm3"]].to_string(index=False))

# ── Step 6: Save comparison table ─────────────────────────────────────────────
combined = compare_no2.merge(
    compare_so2[["year","month","so2_surface_ugm3","so2_ground_ugm3"]],
    on=["year","month"], how="outer"
)
csv_out = os.path.join(OUT_DIR, "tropomi_vs_ground_comparison.csv")
combined.to_csv(csv_out, index=False)
print(f"Comparison table saved: {csv_out}")
print("\nDone.")
