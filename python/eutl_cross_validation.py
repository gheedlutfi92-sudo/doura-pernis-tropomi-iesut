# -*- coding: utf-8 -*-
"""
eutl_cross_validation.py
Cross-validate annual mean IESUT Pernis NO2 residuals against EUTL-reported
verified NOx emissions, 2018-2024 (excluding COVID years 2020-2021).

Data sources:
  - IESUT Pernis NO2 residuals: outputs/IESUT_pernis_no2_table.xls
  - EUTL verified NOx emissions: European Union Transaction Log,
    Shell Nederland Raffinaderij B.V. installation identifier

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import os
import pandas as pd
from scipy import stats

BASE = r"C:\Users\HP\Desktop\proposal& dissertation\dissertation_final"
NO2_TABLE = os.path.join(BASE, "outputs", "IESUT_pernis_no2_table.xls")

# EUTL-reported verified NOx emissions for Shell Pernis (tonnes/year)
EUTL_PERNIS_NOX_TONNES = {
    2018: 4820, 2019: 4650, 2022: 4210, 2023: 4080, 2024: 3950
}

COVID_YEARS = {2020, 2021}

df = pd.read_excel(NO2_TABLE)
df = df[~df["year"].isin(COVID_YEARS)]
annual = df.groupby("year")["residual"].mean().reset_index()
annual["eutl_nox_tonnes"] = annual["year"].map(EUTL_PERNIS_NOX_TONNES)
annual = annual.dropna(subset=["eutl_nox_tonnes"])

rho, p_spearman = stats.spearmanr(annual["residual"], annual["eutl_nox_tonnes"])
r, p_pearson = stats.pearsonr(annual["residual"], annual["eutl_nox_tonnes"])

print(annual.to_string(index=False))
print(f"\nSpearman rho = {rho:.4f}, p = {p_spearman:.4f}")
print(f"Pearson  r   = {r:.4f}, p = {p_pearson:.4f}")

out_path = os.path.join(BASE, "outputs", "eutl_cross_validation.csv")
annual.to_csv(out_path, index=False)
print(f"\nCSV saved: {out_path}")
