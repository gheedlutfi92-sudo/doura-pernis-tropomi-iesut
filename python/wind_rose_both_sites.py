# -*- coding: utf-8 -*-
"""
wind_rose_both_sites.py
Generate traditional wind roses for both study sites in one run:
  - Doura, Baghdad  (Abu Ghraib Agricultural Met. Station, daily)
  - Pernis, Rotterdam (KNMI Station 343 Rotterdam-Geulhaven, hourly)

Study period: 2018-2019 & 2022-2024  (COVID 2020-2021 excluded)

Author : Ghid Albazrkan
Degree : MSc GIS, University of Aberdeen (GG5910/GG5912)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_FINAL = os.environ.get("IESUT_DATA_DIR", "./data")
BASE_OLD   = os.environ.get("IESUT_LEGACY_DATA_DIR", BASE_FINAL)

DOURA_MET  = os.path.join(BASE_FINAL, "data", "Ground_Stations",
                           "Baghdad_Meteorological", "meteorological_dataset_baghdad.xlsx")
KNMI_DIR   = os.path.join(BASE_OLD, "data", "Ground_Stations", "Rotterdam_Meteorological")
KNMI_FILE1 = os.path.join(KNMI_DIR, "uurgeg_343_2011-2020", "uurgeg_343_2011-2020.txt")
KNMI_FILE2 = os.path.join(KNMI_DIR, "uurgeg_343_2021-2030", "uurgeg_343_2021-2030.txt")
OUT_DIR    = os.path.join(BASE_FINAL, "outputs")

COVID_YEARS  = {2020, 2021}
STUDY_YEARS  = [2018, 2019, 2022, 2023, 2024]

# ── Shared styling ─────────────────────────────────────────────────────────
SPEED_BINS   = [0, 1, 2, 3, 4, 20]
SPEED_LABELS = ["< 1 m/s", "1–2 m/s", "2–3 m/s", "3–4 m/s", "≥ 4 m/s"]
COLOURS      = ["#91bfdb", "#4575b4", "#fee090", "#d73027", "#a50026"]
N_SECTORS    = 16
COMPASS16    = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                "S","SSW","SW","WSW","W","WNW","NW","NNW"]
COMPASS8_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
COMPASS8_LBL = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SEASONS      = {
    "Winter (Dec–Feb)": [12, 1, 2],
    "Spring (Mar–May)": [3, 4, 5],
    "Summer (Jun–Aug)": [6, 7, 8],
    "Autumn (Sep–Nov)": [9, 10, 11],
}

# ── Core computation ────────────────────────────────────────────────────────
def compute_wind_rose(ws, wd):
    sector_width = 360.0 / N_SECTORS
    sectors = np.arange(N_SECTORS) * sector_width
    freq = np.zeros((N_SECTORS, len(SPEED_BINS) - 1))
    for i in range(N_SECTORS):
        lo = (sectors[i] - sector_width / 2) % 360
        hi = (sectors[i] + sector_width / 2) % 360
        mask_dir = (wd >= lo) & (wd < hi) if lo < hi else (wd >= lo) | (wd < hi)
        for j in range(len(SPEED_BINS) - 1):
            mask_spd = (ws >= SPEED_BINS[j]) & (ws < SPEED_BINS[j + 1])
            freq[i, j] = (mask_dir & mask_spd).sum()
    return sectors, freq / len(ws) * 100.0

def plot_wind_rose(ax, sectors, freq_pct):
    angles_rad       = np.deg2rad(sectors)
    sector_width_rad = 2 * np.pi / N_SECTORS
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottom = np.zeros(N_SECTORS)
    for j in range(len(SPEED_BINS) - 1):
        ax.bar(angles_rad, freq_pct[:, j], width=sector_width_rad * 0.9,
               bottom=bottom, color=COLOURS[j], alpha=0.85,
               label=SPEED_LABELS[j], edgecolor="white", linewidth=0.4)
        bottom += freq_pct[:, j]
    ax.set_thetagrids(COMPASS8_DEG, COMPASS8_LBL, fontsize=11, fontweight="bold")
    r_max = max(np.ceil(bottom.max() / 5) * 5, 5)
    ax.set_ylim(0, r_max)
    r_ticks = np.arange(5, r_max + 1, 5)
    ax.set_yticks(r_ticks)
    ax.set_yticklabels([f"{r:.0f}%" for r in r_ticks], fontsize=7, color="gray")
    ax.set_rlabel_position(45)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax.spines["polar"].set_visible(False)
    return bottom

def print_summary(label, sectors, freq_pct):
    total_pct = freq_pct.sum(axis=1)
    print(f"\n--- {label}: wind direction frequency (16 sectors) ---")
    for name, pct in zip(COMPASS16, total_pct):
        print(f"  {name:>5}: {pct:.2f}%")
    top3 = total_pct.argsort()[::-1][:3]
    print(f"  Top 3 prevailing: " + ", ".join(f"{COMPASS16[i]} ({total_pct[i]:.1f}%)" for i in top3))

# ══════════════════════════════════════════════════════════════════════════════
# SITE 1 — DOURA, BAGHDAD
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SITE 1 — Doura, Baghdad (Abu Ghraib Station)")
print("=" * 60)

print("Loading Baghdad meteorological data...")
df_raw = pd.read_excel(DOURA_MET, sheet_name="Sheet1", header=4)
df_d   = df_raw.iloc[1:].copy()
df_d.columns = ["date","rain_mm","at_max","at_min","at_avg",
                 "rh_max","rh_min","rh_avg","slr_total",
                 "ws_avg","ws_max","wd","et_mm"]
df_d["date"]   = pd.to_datetime(df_d["date"], errors="coerce")
df_d["wd"]     = pd.to_numeric(df_d["wd"],    errors="coerce")
df_d["ws_avg"] = pd.to_numeric(df_d["ws_avg"], errors="coerce")
df_d = df_d.dropna(subset=["date","wd","ws_avg"]).copy()
df_d["year"]  = df_d["date"].dt.year
df_d["month"] = df_d["date"].dt.month
df_d = df_d[~df_d["year"].isin(COVID_YEARS)].copy()

print(f"  Records after COVID exclusion: {len(df_d):,} days")
print(f"  Date range: {df_d.date.min().date()} to {df_d.date.max().date()}")

ws_d, wd_d = df_d["ws_avg"].values, df_d["wd"].values
sectors_d, freq_d = compute_wind_rose(ws_d, wd_d)
print_summary("Doura", sectors_d, freq_d)

# Annual wind rose — Doura
fig, ax = plt.subplots(1, 1, figsize=(9, 9), subplot_kw={"polar": True}, facecolor="white")
bottom_d = plot_wind_rose(ax, sectors_d, freq_d)
calm_d = (df_d["ws_avg"] < 0.5).mean() * 100
ax.legend(title="Wind Speed (daily avg)", title_fontsize=10,
          loc="lower right", bbox_to_anchor=(1.30, 0.05), fontsize=10, framealpha=0.9)
ax.annotate(f"Calm: {calm_d:.1f}%", xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
fig.text(0.01, 0.98, "Abu Ghraib Agricultural Met. Station\n44.23°E, 33.32°N | Iraq",
         fontsize=8, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.9))
plt.tight_layout()
out_d1 = os.path.join(OUT_DIR, "wind_rose_doura.png")
plt.savefig(out_d1, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_d1}")

# Seasonal wind roses — Doura
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"polar": True}, facecolor="white")
angles_rad = np.deg2rad(sectors_d)
sector_width_rad = 2 * np.pi / N_SECTORS
for ax2, (season_name, months) in zip(axes2.flat, SEASONS.items()):
    df_s = df_d[df_d["month"].isin(months)]
    if df_s.empty:
        continue
    _, freq_s = compute_wind_rose(df_s["ws_avg"].values, df_s["wd"].values)
    ax2.set_theta_zero_location("N"); ax2.set_theta_direction(-1)
    bot = np.zeros(N_SECTORS)
    for j in range(len(SPEED_BINS) - 1):
        ax2.bar(angles_rad, freq_s[:, j], width=sector_width_rad * 0.9,
                bottom=bot, color=COLOURS[j], alpha=0.85, edgecolor="white", linewidth=0.3)
        bot += freq_s[:, j]
    ax2.set_thetagrids(COMPASS8_DEG, COMPASS8_LBL, fontsize=9)
    r_max_s = max(np.ceil(bot.max() / 5) * 5, 5)
    ax2.set_ylim(0, r_max_s)
    ax2.set_yticks(np.arange(5, r_max_s + 1, 5))
    ax2.set_yticklabels([f"{r:.0f}%" for r in np.arange(5, r_max_s + 1, 5)], fontsize=7, color="gray")
    ax2.set_rlabel_position(45)
    ax2.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax2.spines["polar"].set_visible(False)
    ax2.set_title(f"{season_name}\nN={len(df_s):,} days", fontsize=11, fontweight="bold", pad=12)
handles = [plt.Rectangle((0,0),1,1,facecolor=c,alpha=0.85) for c in COLOURS]
fig2.legend(handles, SPEED_LABELS, title="Wind Speed", title_fontsize=10,
            loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5,-0.02), framealpha=0.9)
plt.tight_layout()
out_d2 = os.path.join(OUT_DIR, "wind_rose_seasonal_doura.png")
fig2.savefig(out_d2, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_d2}")

# ══════════════════════════════════════════════════════════════════════════════
# SITE 2 — PERNIS, ROTTERDAM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SITE 2 — Pernis, Rotterdam (KNMI Station 343)")
print("=" * 60)

def load_knmi(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line[0].isalpha():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                date = str(parts[1]); hh = int(parts[2])
                dd = parts[3]; fh = parts[4]
                if not dd or not fh:
                    continue
                dd = int(dd); fh = float(fh) / 10.0
                if dd in (0, 990):
                    continue
                year = int(date[:4]); month = int(date[4:6])
                rows.append({"year": year, "month": month, "wd": dd, "ws": fh})
            except:
                continue
    return pd.DataFrame(rows)

print("Loading KNMI Rotterdam-Geulhaven data...")
df_r = pd.concat([load_knmi(KNMI_FILE1), load_knmi(KNMI_FILE2)], ignore_index=True)
df_r = df_r[(df_r["year"] >= 2018) & (df_r["year"] <= 2024) & (~df_r["year"].isin(COVID_YEARS))].copy()
print(f"  Records after COVID exclusion: {len(df_r):,} hourly observations")

ws_r, wd_r = df_r["ws"].values, df_r["wd"].values
sectors_r, freq_r = compute_wind_rose(ws_r, wd_r)
print_summary("Rotterdam", sectors_r, freq_r)

# Annual wind rose — Rotterdam
fig3, ax3 = plt.subplots(1, 1, figsize=(9, 9), subplot_kw={"polar": True}, facecolor="white")
bottom_r = plot_wind_rose(ax3, sectors_r, freq_r)
calm_r = (df_r["ws"] < 0.5).mean() * 100
ax3.legend(title="Wind Speed (hourly mean)", title_fontsize=10,
           loc="lower right", bbox_to_anchor=(1.30, 0.05), fontsize=10, framealpha=0.9)
ax3.annotate(f"Calm: {calm_r:.1f}%", xy=(0.01, 0.01), xycoords="axes fraction",
             fontsize=9, color="gray",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
fig3.text(0.01, 0.98, "KNMI Station 343 — Rotterdam-Geulhaven\n4.32°E, 51.89°N | Netherlands",
          fontsize=8, verticalalignment="top",
          bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.9))
plt.tight_layout()
out_r1 = os.path.join(OUT_DIR, "wind_rose_rotterdam.png")
plt.savefig(out_r1, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_r1}")

# Seasonal wind roses — Rotterdam
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"polar": True}, facecolor="white")
angles_rad_r = np.deg2rad(sectors_r)
for ax4, (season_name, months) in zip(axes4.flat, SEASONS.items()):
    df_rs = df_r[df_r["month"].isin(months)]
    if df_rs.empty:
        continue
    _, freq_rs = compute_wind_rose(df_rs["ws"].values, df_rs["wd"].values)
    ax4.set_theta_zero_location("N"); ax4.set_theta_direction(-1)
    bot = np.zeros(N_SECTORS)
    for j in range(len(SPEED_BINS) - 1):
        ax4.bar(angles_rad_r, freq_rs[:, j], width=sector_width_rad * 0.9,
                bottom=bot, color=COLOURS[j], alpha=0.85, edgecolor="white", linewidth=0.3)
        bot += freq_rs[:, j]
    ax4.set_thetagrids(COMPASS8_DEG, COMPASS8_LBL, fontsize=9)
    r_max_rs = max(np.ceil(bot.max() / 5) * 5, 5)
    ax4.set_ylim(0, r_max_rs)
    ax4.set_yticks(np.arange(5, r_max_rs + 1, 5))
    ax4.set_yticklabels([f"{r:.0f}%" for r in np.arange(5, r_max_rs + 1, 5)], fontsize=7, color="gray")
    ax4.set_rlabel_position(45)
    ax4.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax4.spines["polar"].set_visible(False)
    ax4.set_title(f"{season_name}\nN={len(df_rs):,} hours", fontsize=11, fontweight="bold", pad=12)
handles = [plt.Rectangle((0,0),1,1,facecolor=c,alpha=0.85) for c in COLOURS]
fig4.legend(handles, SPEED_LABELS, title="Wind Speed", title_fontsize=10,
            loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5,-0.02), framealpha=0.9)
plt.tight_layout()
out_r2 = os.path.join(OUT_DIR, "wind_rose_seasonal_rotterdam.png")
fig4.savefig(out_r2, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_r2}")

# ══════════════════════════════════════════════════════════════════════════════
# POLLUTION ROSE — DOURA (NO₂ IESUT residuals by wind sector)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("POLLUTION ROSE — Doura NO₂ (IESUT residuals by wind sector)")
print("=" * 60)

IESUT_NO2 = os.path.join(BASE_FINAL, "outputs", "IESUT_doura_no2_table.xls")
df_iesut = pd.read_excel(IESUT_NO2)
df_iesut = df_iesut[~df_iesut["year"].isin(COVID_YEARS)].copy()

# Monthly mean residual across all pixels
iesut_monthly = (df_iesut.groupby(["year", "month"])["residual"]
                 .mean().reset_index()
                 .rename(columns={"residual": "no2_residual"}))

# Each day of wind data gets that month's mean IESUT residual
df_poll = df_d[["year", "month", "wd"]].merge(iesut_monthly, on=["year", "month"], how="inner")
print(f"  Matched: {len(df_poll):,} daily wind observations with IESUT residuals")

# Bin into 8 compass sectors
def wd_to_sector8(wd_val):
    return int(round(wd_val / 45)) % 8

df_poll["sector"] = df_poll["wd"].apply(wd_to_sector8)
SECTOR8_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ANGLES8       = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
SECTOR_WIDTH8 = 2 * np.pi / 8

sector_stats = df_poll.groupby("sector").agg(
    mean_res=("no2_residual", "mean"),
    count=("no2_residual", "count")
).reindex(range(8), fill_value=0).reset_index()

print("\n--- Mean NO₂ Residual by Wind Sector ---")
for _, row in sector_stats.iterrows():
    name = SECTOR8_NAMES[int(row["sector"])]
    print(f"  {name:>3}: {row['mean_res']:+.3e} mol/m²  (n={int(row['count'])} days)")

residuals8 = sector_stats["mean_res"].values
counts8    = sector_stats["count"].values

# Plot — positive sectors red, negative sectors blue, bars scaled to absolute value
colours_poll = ["#d73027" if r >= 0 else "#4575b4" for r in residuals8]
bar_heights  = np.abs(residuals8)

fig5, ax5 = plt.subplots(1, 1, figsize=(9, 9), subplot_kw={"polar": True}, facecolor="white")
ax5.set_theta_zero_location("N")
ax5.set_theta_direction(-1)

for i in range(8):
    ax5.bar(ANGLES8[i], bar_heights[i], width=SECTOR_WIDTH8 * 0.85,
            color=colours_poll[i], alpha=0.85, edgecolor="white", linewidth=0.5)

ax5.set_thetagrids(np.rad2deg(ANGLES8), SECTOR8_NAMES, fontsize=13, fontweight="bold")
r_max5 = max(bar_heights.max() * 1.25, 1e-8)
ax5.set_ylim(0, r_max5)
ax5.set_rlabel_position(45)

# Format radial tick labels in scientific notation
r_ticks5 = np.linspace(0, r_max5, 4)[1:]
ax5.set_yticks(r_ticks5)
ax5.set_yticklabels([f"{v:.1e}" for v in r_ticks5], fontsize=7, color="gray")

ax5.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
ax5.spines["polar"].set_visible(False)

# Annotate each sector with sign and count
for i in range(8):
    sign = "+" if residuals8[i] >= 0 else "−"
    ax5.annotate(
        f"{sign}\nn={int(counts8[i])}",
        xy=(ANGLES8[i], bar_heights[i] + r_max5 * 0.06),
        ha="center", va="bottom", fontsize=8,
        color="#d73027" if residuals8[i] >= 0 else "#4575b4", fontweight="bold"
    )

red_patch  = plt.Rectangle((0,0),1,1, facecolor="#d73027", alpha=0.85)
blue_patch = plt.Rectangle((0,0),1,1, facecolor="#4575b4", alpha=0.85)
ax5.legend([red_patch, blue_patch],
           ["Positive residual (industrial excess)", "Negative residual (below background)"],
           loc="lower right", bbox_to_anchor=(1.40, 0.0), fontsize=10, framealpha=0.9)

fig5.text(0.01, 0.98, "Bar height = |mean residual| (mol/m²)\nRed = excess above background; Blue = below background",
          fontsize=8, verticalalignment="top",
          bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="goldenrod", alpha=0.9))

plt.tight_layout()
out_poll = os.path.join(OUT_DIR, "pollution_rose_doura.png")
fig5.savefig(out_poll, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nSaved: {out_poll}")

# ── Final summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("All outputs saved to dissertation_final/outputs/:")
for f in [out_d1, out_d2, out_r1, out_r2, out_poll]:
    print(f"  {os.path.basename(f)}")
print("=" * 60)
