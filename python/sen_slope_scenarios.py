# -*- coding: utf-8 -*-
"""
sen_slope_scenarios.py
(renamed from sen_slope_projection.py -- now also produces both scenario charts)

Compute Mann-Kendall trend test + Sen's Slope on IESUT NO2 residuals from the
final IESUT run, project 10 years forward from Dec 2025, and plot two
scenario charts:

  Chart A -- If nothing changes: observed + projected NO2 industrial excess
             for both sites, Sen's slope, no regulatory intervention.
             (matches Dissertation Section 5.3 -- "+115.2% by 2030,
             +256.3% by 2035")

  Chart B -- Regulatory adoption counterfactual: what if Doura's excess
             declined at the same (statistically verified) rate Shell
             Pernis's has, instead of continuing its own trend? Both
             trajectories reuse the Sen's slopes computed for Chart A --
             no separate, weaker estimate is introduced. No WHO guideline
             line is plotted here: the guideline is a TOTAL ambient
             concentration standard, not comparable to the industrial
             excess residual plotted in this chart, so including it was
             misleading (confirmed visually -- it sat disconnected near
             the top of the chart, well above the plotted range).

Charts are plotted in micromol/m^2 (not mol/m^2) purely for axis
readability -- avoids matplotlib's "1e-5" scientific-notation offset.
The CSV outputs and console summary remain in mol/m^2, matching the
dissertation text exactly.

Inputs:
  dissertation_final/outputs/IESUT_doura_no2_table.xls
  dissertation_final/outputs/IESUT_pernis_no2_table.xls

Outputs:
  dissertation_final/outputs/mann_kendall_results.csv
  dissertation_final/outputs/sen_slope_projection.csv
  dissertation_final/outputs/chart_A_business_as_usual_no2.png
  dissertation_final/outputs/chart_B_regulatory_adoption_no2.png

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)

-- Reuse --------------------------------------------------------------
To run this for SO2 instead of NO2, change POLLUTANT below to "so2" --
nothing else needs editing. Output filenames auto-adjust; the original
NO2 outputs (mann_kendall_results.csv, sen_slope_projection.csv) are
left untouched when POLLUTANT = "no2" so the already-verified dissertation
numbers are never overwritten by an SO2 run.
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# -- Configuration -- change this one line to reuse for SO2 -----------------
POLLUTANT = "no2"   # "no2" or "so2"

POLLUTANT_LABEL = "NO\u2082" if POLLUTANT == "no2" else "SO\u2082"

# -- Paths --------------------------------------------------------------
BASE    = r"C:\Users\HP\Desktop\proposal& dissertation\dissertation_final\outputs"
DOURA   = os.path.join(BASE, f"IESUT_doura_{POLLUTANT}_table.xls")
PERNIS  = os.path.join(BASE, f"IESUT_pernis_{POLLUTANT}_table.xls")

# Keep original filenames for the already-verified NO2 run; suffix otherwise
OUT_MK  = os.path.join(BASE, "mann_kendall_results.csv"
                        if POLLUTANT == "no2" else f"mann_kendall_results_{POLLUTANT}.csv")
OUT_PRJ = os.path.join(BASE, "sen_slope_projection.csv"
                        if POLLUTANT == "no2" else f"sen_slope_projection_{POLLUTANT}.csv")
CHART_A = os.path.join(BASE, f"chart_A_business_as_usual_{POLLUTANT}.png")
CHART_B = os.path.join(BASE, f"chart_B_regulatory_adoption_{POLLUTANT}.png")

# Chart-only unit conversion: mol/m^2 -> micromol/m^2, for axis readability.
UNIT_SCALE = 1e6
UNIT_LABEL = "\u00b5mol/m\u00b2"

COVID_YEARS = {2020, 2021}

# -- Load and aggregate to monthly means ---------------------------------
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

# -- Step 1: Mann-Kendall + Sen's slope ----------------------------------
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
        "Series"                     : f"{POLLUTANT_LABEL} residual (all pixels)",
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

# -- Step 2: 10-year projection ------------------------------------------
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

# -- Key projection numbers ------------------------------------------------
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

# -- Final summary -----------------------------------------------------
print("=" * 60)
print("  RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Doura (Baghdad) {POLLUTANT_LABEL} residual  (n={len(doura_m)} months)")
print(f"    Trend      : {mk_slopes['Doura (Baghdad)']['trend'].upper()}")
print(f"    p-value    : {mk_slopes['Doura (Baghdad)']['p']:.6f}")
print(f"    Sen slope  : {d_slope:+.4e} mol/m\u00b2/month")
print(f"    Dec 2025   : {d_now:.4e} mol/m\u00b2")
print(f"    Jan 2030   : {d_2030:.4e} mol/m\u00b2  ({pct_2030:+.1f}% from Dec 2025)")
print(f"    Jan 2035   : {d_2035:.4e} mol/m\u00b2  ({pct_2035:+.1f}% from Dec 2025)")

p_slope = mk_slopes["Shell Pernis (Rotterdam)"]["slope"]
print(f"\n  Shell Pernis {POLLUTANT_LABEL} residual  (n={len(pernis_m)} months)")
print(f"    Trend      : {mk_slopes['Shell Pernis (Rotterdam)']['trend'].upper()}")
print(f"    p-value    : {mk_slopes['Shell Pernis (Rotterdam)']['p']:.6f}")
print(f"    Sen slope  : {p_slope:+.4e} mol/m\u00b2/month")

print(f"\n  Saved: {OUT_MK}")
print(f"  Saved: {OUT_PRJ}")
print("=" * 60)

# ============================================================================
# CHART A -- If nothing changes (plot of the Mann-Kendall / Sen's slope
#            results computed above)
# ============================================================================
print("\n[Chart A] Plotting 'if nothing changes' scenario...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(doura_m["date"], doura_m["residual"] * UNIT_SCALE, "o-", color="#F44336",
        markersize=3, linewidth=1, alpha=0.6, label="Doura - observed")
ax.plot(pernis_m["date"], pernis_m["residual"] * UNIT_SCALE, "o-", color="#2196F3",
        markersize=3, linewidth=1, alpha=0.6, label="Shell Pernis - observed")

ax.plot(proj_dates, doura_proj * UNIT_SCALE, "--", color="#F44336", linewidth=2,
        label="Doura - projected (Sen's slope, no intervention)")
ax.fill_between(proj_dates,
                 (doura_proj - doura_std * tf) * UNIT_SCALE,
                 (doura_proj + doura_std * tf) * UNIT_SCALE,
                 color="#F44336", alpha=0.15)

ax.plot(proj_dates, pernis_proj * UNIT_SCALE, "--", color="#2196F3", linewidth=2,
        label="Shell Pernis - projected (Sen's slope)")
ax.fill_between(proj_dates,
                 (pernis_proj - pernis_std * tf) * UNIT_SCALE,
                 (pernis_proj + pernis_std * tf) * UNIT_SCALE,
                 color="#2196F3", alpha=0.15)

ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
ax.set_xlabel("Date")
ax.set_ylabel(f"{POLLUTANT_LABEL} industrial residual ({UNIT_LABEL})")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(CHART_A, dpi=300)
plt.close(fig)
print(f"  Saved: {CHART_A}")

# ============================================================================
# CHART B -- Regulatory adoption counterfactual
# ============================================================================
print("\n[Chart B] Building regulatory adoption counterfactual...")

doura_annual  = doura_m.groupby("year")["residual"].mean()
pernis_annual = pernis_m.groupby("year")["residual"].mean()

# Reuse the same Mann-Kendall / Sen's slopes already computed and verified
# for Chart A (converted from mol/m^2/month to mol/m^2/year) instead of
# fitting a separate annual OLS estimate. A fresh 6-7-point annual OLS fit
# for Pernis previously gave a wrong-signed, non-significant result (p=0.19)
# that disagreed with the robust 67-month Mann-Kendall test (p=0.045) --
# reusing the verified Sen's slope avoids introducing that weaker estimate.
d_slope_annual = mk_slopes["Doura (Baghdad)"]["slope"] * 12
p_slope_annual = mk_slopes["Shell Pernis (Rotterdam)"]["slope"] * 12

last_year  = int(doura_annual.index.max())
last_value = doura_annual.iloc[-1]

# "No intervention" counterfactual: Doura continues its own (verified) trend
bau_years  = list(range(last_year, last_year + 16))
bau_values = np.array([last_value + d_slope_annual * (yr - last_year) for yr in bau_years])

# "Regulated" counterfactual: Doura's trajectory shifted by Pernis's own
# (verified) rate of change instead -- i.e. what if Doura's excess declined
# at the same statistically significant rate Pernis's has.
reg_years  = list(range(last_year, last_year + 16))
reg_values = np.array([last_value + p_slope_annual * (yr - last_year) for yr in reg_years])

print(f"  Doura Sen's slope (annualised)  : {d_slope_annual:.4e} mol/m\u00b2 per year")
print(f"  Pernis Sen's slope (annualised) : {p_slope_annual:.4e} mol/m\u00b2 per year")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(doura_annual.index, doura_annual.values * UNIT_SCALE, "o-", color="#F44336",
        linewidth=2, markersize=6, label="Doura - observed (annual mean)")

ax.plot(bau_years, bau_values * UNIT_SCALE, "--", color="#F44336", linewidth=2, alpha=0.6,
        label="Doura - projected, no intervention (Sen's slope)")

ax.plot(reg_years, reg_values * UNIT_SCALE, "--", color="#4CAF50", linewidth=2,
        label="Doura - projected, if regulated like Shell Pernis (Sen's slope)")

ax.set_xlabel("Year")
ax.set_ylabel(f"{POLLUTANT_LABEL} industrial residual ({UNIT_LABEL}, annual mean)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(CHART_B, dpi=300)
plt.close(fig)
print(f"  Saved: {CHART_B}")

print("\n" + "=" * 60)
print("DONE. All outputs saved to:")
print(f"  {OUT_MK}")
print(f"  {OUT_PRJ}")
print(f"  {CHART_A}")
print(f"  {CHART_B}")
print("=" * 60)
