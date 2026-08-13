# -*- coding: utf-8 -*-
"""
sen_slope_projection.py
Compute Mann-Kendall trend test + Sen's Slope on IESUT NO2 residuals
from the final IESUT run, then project 10 years forward from Dec 2025.

Inputs:
  dissertation_final/outputs/IESUT_doura_no2_table.xls
  dissertation_final/outputs/IESUT_pernis_no2_table.xls

Outputs:
  dissertation_final/outputs/mann_kendall_results.csv
  dissertation_final/outputs/sen_slope_projection.csv

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import pymannkendall as mk
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
BASE    = r"C:\Users\HP\Desktop\proposal& dissertation\dissertation_final\outputs"
DOURA   = os.path.join(BASE, "IESUT_doura_no2_table.xls")
PERNIS  = os.path.join(BASE, "IESUT_pernis_no2_table.xls")
OUT_MK  = os.path.join(BASE, "mann_kendall_results.csv")
OUT_PRJ = os.path.join(BASE, "sen_slope_projection.csv")

COVID_YEARS = {2020, 2021}

# ── Load and aggregate to monthly means ───────────────────────────────────
def monthly_means(fpath):
    df = pd.read_excel(fpath, engine="xlrd")
    df = df[~df["year"].isin(COVID_YEARS)]
    monthly = (
        df.groupby(["year", "month"])["residual"]
        .mean()
        .reset_index()
        .sort_values(["year", "month"])
        .reset_index(drop=True)
    )
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    return monthly

doura_m  = monthly_means(DOURA)
pernis_m = monthly_means(PERNIS)

# ── Step 1: Mann-Kendall + Sen's slope ───────────────────────────────────
print("Running Mann-Kendall trend test...\n")

mk_rows   = []
mk_slopes = {}

for site_name, monthly in [("Doura (Baghdad)", doura_m),
                            ("Shell Pernis (Rotterdam)", pernis_m)]:
    series = monthly["residual"].values
    n      = len(series)
    result = mk.original_test(series)

    mk_rows.append({
        "Site"                       : site_name,
        "Series"                     : "NO2 residual (all pixels)",
        "n_months"                   : n,
        "Trend"                      : result.trend,
        "p_value"                    : result.p,
        "Tau"                        : result.Tau,
        "Sen_slope_per_month_mol_m2" : result.slope,
    })
    mk_slopes[site_name] = {
        "monthly": monthly,
        "slope"  : result.slope,
        "p"      : result.p,
        "trend"  : result.trend,
    }

pd.DataFrame(mk_rows).to_csv(OUT_MK, index=False)

# ── Step 2: 10-year projection ─────────────────────────────────────────────
proj_start = pd.Timestamp("2025-12-01")
proj_dates = pd.date_range(proj_start, periods=121, freq="MS")

def make_projection(monthly, slope, proj_dates):
    anchor_date = monthly["date"].iloc[len(monthly) // 2]
    anchor_val  = float(np.median(monthly["residual"]))
    months_off  = ((proj_dates.year  - anchor_date.year)  * 12 +
                   (proj_dates.month - anchor_date.month))
    return anchor_val + slope * months_off, anchor_date, anchor_val

doura_proj,  d_anc_date, d_anc_val  = make_projection(
    doura_m,  mk_slopes["Doura (Baghdad)"]["slope"],         proj_dates)
pernis_proj, p_anc_date, p_anc_val = make_projection(
    pernis_m, mk_slopes["Shell Pernis (Rotterdam)"]["slope"], proj_dates)

doura_std  = doura_m["residual"].std()
pernis_std = pernis_m["residual"].std()
tf = np.linspace(1.0, 2.2, len(proj_dates))

pd.DataFrame({
    "date"              : proj_dates,
    "doura_projected"   : doura_proj,
    "doura_proj_lower"  : doura_proj  - doura_std  * tf,
    "doura_proj_upper"  : doura_proj  + doura_std  * tf,
    "pernis_projected"  : pernis_proj,
    "pernis_proj_lower" : pernis_proj - pernis_std * tf,
    "pernis_proj_upper" : pernis_proj + pernis_std * tf,
}).to_csv(OUT_PRJ, index=False)

# ── Key projection numbers ─────────────────────────────────────────────────
d_slope = mk_slopes["Doura (Baghdad)"]["slope"]

def proj_to(anchor_date, anchor_val, slope, target):
    t = pd.Timestamp(target)
    return anchor_val + slope * ((t.year - anchor_date.year) * 12 +
                                  (t.month - anchor_date.month))

d_now  = doura_proj[0]
d_2030 = proj_to(d_anc_date, d_anc_val, d_slope, "2030-01-01")
d_2035 = proj_to(d_anc_date, d_anc_val, d_slope, "2035-01-01")
pct_2030 = (d_2030 - d_now) / abs(d_now) * 100
pct_2035 = (d_2035 - d_now) / abs(d_now) * 100

# ── Final summary ──────────────────────────────────────────────────────────
print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Doura (Baghdad) NO\u2082 residual  (n={len(doura_m)} months)")
print(f"    Trend      : {mk_slopes['Doura (Baghdad)']['trend'].upper()}")
print(f"    p-value    : {mk_slopes['Doura (Baghdad)']['p']:.6f}")
print(f"    Sen slope  : {d_slope:+.4e} mol/m\u00b2/month")
print(f"    Dec 2025   : {d_now:.4e} mol/m\u00b2")
print(f"    Jan 2030   : {d_2030:.4e} mol/m\u00b2  ({pct_2030:+.1f}% from Dec 2025)")
print(f"    Jan 2035   : {d_2035:.4e} mol/m\u00b2  ({pct_2035:+.1f}% from Dec 2025)")

p_slope = mk_slopes["Shell Pernis (Rotterdam)"]["slope"]
print(f"\n  Shell Pernis NO\u2082 residual  (n={len(pernis_m)} months)")
print(f"    Trend      : {mk_slopes['Shell Pernis (Rotterdam)']['trend'].upper()}")
print(f"    p-value    : {mk_slopes['Shell Pernis (Rotterdam)']['p']:.6f}")
print(f"    Sen slope  : {p_slope:+.4e} mol/m\u00b2/month")

print(f"\n  Saved: {OUT_MK}")
print(f"  Saved: {OUT_PRJ}")
print("=" * 60)
print("\nDone.")
