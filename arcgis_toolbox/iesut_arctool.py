"""
IESUT — Industrial Emission Separator & Uncertainty Tool
=========================================================
ArcGIS Pro Script Tool — ANALYSIS ONLY
Outputs: GDB feature classes + statistics in Geoprocessing window
Charts: run iesut_charts.py separately after this tool completes

PARAMETERS (add in this order in ArcGIS toolbox):
  0  Input Data File         File      Input  Required
                             Browse to the site CSV file.
  1  Site Name               String    Input  Required
                             Short name used for output layer naming, e.g. doura, kuwait, abadan.
                             No spaces — use underscores if needed.
  2  Pollutant               String    Input  Required
                             Value List: NO2; SO2; Both
  3  Output Geodatabase      Workspace Input  Required
  4  Background Threshold km Long      Input  Optional  Default: 12
  5  Number of RF Trees      Long      Input  Optional  Default: 300
  6  Bootstrap Iterations    Long      Input  Optional  Default: 100
  7  Confidence Interval %   Long      Input  Optional
                             Value List: 80; 90; 95  Default: 90
  8  Run Sensitivity Analysis Boolean  Input  Optional  Default: true
  9  Output Feature Class    Feature Class  Output  Derived
"""

import arcpy
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def msg(text):
    arcpy.AddMessage(text)

def warn(text):
    arcpy.AddWarning(text)

# Use threading backend so sklearn can parallelise inside ArcGIS Pro
# without spawning separate processes (which ArcGIS kills)
joblib.parallel_config(backend="threading", n_jobs=-1)


# ---------------------------------------------------------------------------
# READ ARCGIS TOOL PARAMETERS
# ---------------------------------------------------------------------------

_data_file   = arcpy.GetParameterAsText(0)
_site_key    = arcpy.GetParameterAsText(1).strip().lower().replace(" ", "_")
_poll_input  = arcpy.GetParameterAsText(2)
_output_gdb  = arcpy.GetParameterAsText(3)
_threshold   = arcpy.GetParameterAsText(4)
_n_trees     = arcpy.GetParameterAsText(5)
_n_bootstrap = arcpy.GetParameterAsText(6)
_ci_pct      = arcpy.GetParameterAsText(7)
_run_sens    = arcpy.GetParameterAsText(8)

if not _site_key:
    arcpy.AddError("Site Name cannot be empty. Enter a short name, e.g. doura, kuwait, abadan.")
    raise SystemExit(1)

_POLL_MAP = {"NO2": "no2", "SO2": "so2", "Both": "both"}
_poll_key     = _POLL_MAP.get(_poll_input, "no2")
_threshold_km = int(_threshold)   if _threshold.strip()   else 12
_n_trees_val  = int(_n_trees)     if _n_trees.strip()     else 300
_n_boot_val   = int(_n_bootstrap) if _n_bootstrap.strip() else 100
_ci_val       = int(_ci_pct)      if _ci_pct.strip()      else 90
_run_sens_val = _run_sens.lower() != "false" if _run_sens.strip() else True

_lower_pct = (100 - _ci_val) / 2
_upper_pct = 100 - (100 - _ci_val) / 2

POLLUTANTS_TO_RUN = ["no2", "so2"] if _poll_key == "both" else [_poll_key]


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

class Config:
    GDB_PATH   = _output_gdb
    DATA_FILE  = _data_file

    BACKGROUND_THRESHOLD_KM = _threshold_km
    PREDICTORS = [
        "viirs_ntl", "population", "elevation",
        "lst_celsius", "wind_speed", "temperature_c",
        "dist_from_refinery",
        "month_sin", "month_cos",
    ]

    N_TREES         = _n_trees_val
    N_BOOTSTRAP     = _n_boot_val
    CI_LOWER        = _lower_pct
    CI_UPPER        = _upper_pct
    CV_FOLDS        = 5
    RANDOM_STATE    = 42
    RUN_SENSITIVITY = _run_sens_val
    SENSITIVITY_DISTANCES = [8, 10, 12, 14, 16, 18, 20, 25]


# ---------------------------------------------------------------------------
# MODULE 1 — DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    msg(f"[DATA] Site detected : {_site_key}")
    msg(f"[DATA] Loading       : {os.path.basename(Config.DATA_FILE)}")
    df = pd.read_csv(Config.DATA_FILE)
    msg(f"[DATA] Rows: {len(df):,}   Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# MODULE 2 — TRAINING DATA PREPARATION
# ---------------------------------------------------------------------------

def prepare_training_data(df, pollutant, threshold_km=None):
    threshold   = threshold_km or Config.BACKGROUND_THRESHOLD_KM
    threshold_m = threshold * 1000

    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["so2_no2_ratio"] = np.where(df["no2"] > 0, df["so2"] / df["no2"], 0.0)

    leaky = {pollutant}
    if pollutant == "so2":
        leaky.add("so2_no2_ratio")

    available_predictors = [
        p for p in Config.PREDICTORS if p in df.columns and p not in leaky
    ]

    mask_bg    = df["dist_from_refinery"] > threshold_m
    train_df   = df[mask_bg].copy()
    predict_df = df.copy()

    train_df   = train_df[train_df[pollutant] > 0].dropna(
        subset=[pollutant] + available_predictors).reset_index(drop=True)
    predict_df = predict_df.dropna(
        subset=available_predictors).reset_index(drop=True)

    X_train   = train_df[available_predictors].values
    y_train   = train_df[pollutant].values
    X_predict = predict_df[available_predictors].values

    msg(f"[DATA] Background training pixels : {len(X_train)}")
    msg(f"[DATA] Full prediction pixels     : {len(X_predict)}")
    msg(f"[DATA] Predictors used ({len(available_predictors)})       : {available_predictors}")

    return X_train, y_train, X_predict, predict_df, available_predictors, train_df


# ---------------------------------------------------------------------------
# MODULE 3 — RANDOM FOREST + CROSS-VALIDATION
# ---------------------------------------------------------------------------

def train_rf(X_train, y_train):
    msg(f"[MODEL] Training {Config.N_TREES}-tree Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=Config.N_TREES,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        oob_score=True,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    msg(f"[MODEL] OOB R² = {rf.oob_score_:.4f}  "
        f"(in-sample fit — how well model explains background variation)")
    return rf


def cross_validate(rf, X_train, y_train, train_df=None):
    k = Config.CV_FOLDS
    n = len(X_train)
    r2_scores, rmse_scores, mae_scores = [], [], []

    if train_df is not None and "year" in train_df.columns:
        sort_idx = np.argsort((train_df["year"] * 100 + train_df["month"]).values)
        msg(f"[CV] Temporally blocked cross-validation "
            f"({k} folds, sorted chronologically — no temporal leakage)")
    else:
        sort_idx = np.arange(n)

    X_s, y_s = X_train[sort_idx], y_train[sort_idx]
    fold_size = n // k

    for fold in range(k):
        val_start = fold * fold_size
        val_end   = val_start + fold_size if fold < k - 1 else n
        mask = np.zeros(n, dtype=bool)
        mask[val_start:val_end] = True

        fold_rf = RandomForestRegressor(
            n_estimators=Config.N_TREES, max_features="sqrt",
            min_samples_split=5, min_samples_leaf=2,
            random_state=Config.RANDOM_STATE, n_jobs=-1,
        )
        fold_rf.fit(X_s[~mask], y_s[~mask])
        y_pred = fold_rf.predict(X_s[mask])

        r2_scores.append(r2_score(y_s[mask], y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_s[mask], y_pred)))
        mae_scores.append(mean_absolute_error(y_s[mask], y_pred))
        msg(f"  Fold {fold+1} (rows {val_start}–{val_end-1}): "
            f"R²={r2_scores[-1]:.4f}  RMSE={rmse_scores[-1]:.6f}  "
            f"MAE={mae_scores[-1]:.6f}")

    results = {
        "r2_mean":   np.mean(r2_scores),  "r2_std":   np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores), "rmse_std": np.std(rmse_scores),
        "mae_mean":  np.mean(mae_scores),
    }
    msg(f"[CV] Mean R²   = {results['r2_mean']:.4f} ± {results['r2_std']:.4f}  "
        f"(out-of-sample temporal generalisation)")
    msg(f"[CV] Mean RMSE = {results['rmse_mean']:.6f} ± {results['rmse_std']:.6f}")
    msg(f"[CV] Mean MAE  = {results['mae_mean']:.6f}")
    return results


def get_feature_importance(rf, X_train, y_train, feature_names):
    msg("[IMPORTANCE] Calculating permutation importance (30 repeats)...")
    baseline = r2_score(y_train, rf.predict(X_train))
    rng = np.random.RandomState(Config.RANDOM_STATE)
    n_features = len(feature_names)
    importances = np.zeros((n_features, 30))

    for rep in range(30):
        for i in range(n_features):
            X_perm = X_train.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            importances[i, rep] = baseline - r2_score(y_train, rf.predict(X_perm))

    imp_df = pd.DataFrame({
        "feature":                feature_names,
        "impurity_importance":    rf.feature_importances_,
        "permutation_importance": importances.mean(axis=1),
        "permutation_std":        importances.std(axis=1),
    }).sort_values("permutation_importance", ascending=False)

    msg("[IMPORTANCE] Ranking (permutation-based — most reliable):")
    for _, row in imp_df.iterrows():
        msg(f"  {row['feature']:25s}  {row['permutation_importance']:.6f} "
            f"± {row['permutation_std']:.6f}")
    return imp_df


# ---------------------------------------------------------------------------
# MODULE 4 — UNCERTAINTY QUANTIFICATION
# ---------------------------------------------------------------------------

def predict_with_uncertainty(rf, X_predict):
    msg(f"[UNCERTAINTY] Quantile RF — extracting {_lower_pct:.0f}th–{_upper_pct:.0f}th "
        f"percentile across {Config.N_TREES} individual tree predictions...")
    tree_preds = np.array([t.predict(X_predict) for t in rf.estimators_])
    pred_lower = np.percentile(tree_preds, Config.CI_LOWER, axis=0)
    pred_upper = np.percentile(tree_preds, Config.CI_UPPER, axis=0)
    q_unc = pred_upper - pred_lower
    msg(f"[UNCERTAINTY] Mean quantile uncertainty ({int(_ci_val)}% CI): "
        f"{np.mean(q_unc):.8f} mol/m²")
    return {
        "predicted_mean":   np.mean(tree_preds, axis=0),
        "predicted_median": np.percentile(tree_preds, 50, axis=0),
        "predicted_lower":  pred_lower,
        "predicted_upper":  pred_upper,
        "uncertainty":      q_unc,
    }


def bootstrap_uncertainty(X_train, y_train, X_predict):
    n      = len(X_train)
    n_boot = Config.N_BOOTSTRAP
    preds  = np.zeros((n_boot, len(X_predict)))

    msg(f"[BOOTSTRAP] Running {n_boot} iterations "
        f"(resampling {n} training pixels with replacement each time)...")
    for i in range(n_boot):
        idx  = np.random.choice(n, size=n, replace=True)
        rf_b = RandomForestRegressor(
            n_estimators=100, max_features="sqrt",
            min_samples_split=5, random_state=i, n_jobs=-1,
        )
        rf_b.fit(X_train[idx], y_train[idx])
        preds[i] = rf_b.predict(X_predict)
        if (i + 1) % 25 == 0:
            msg(f"  Completed {i+1}/{n_boot}")
            arcpy.SetProgressorPosition()

    boot_unc = np.percentile(preds, 97.5, axis=0) - np.percentile(preds, 2.5, axis=0)
    msg(f"[BOOTSTRAP] Mean bootstrap uncertainty (95% spread): "
        f"{np.mean(boot_unc):.8f} mol/m²")
    return {
        "boot_lower":       np.percentile(preds, 2.5,  axis=0),
        "boot_upper":       np.percentile(preds, 97.5, axis=0),
        "boot_std":         np.std(preds, axis=0),
        "boot_uncertainty": boot_unc,
    }


def calculate_residuals(actual, predictions, bootstrap):
    residual     = actual - predictions["predicted_median"]
    resid_lower  = actual - predictions["predicted_upper"]
    resid_upper  = actual - predictions["predicted_lower"]
    combined_unc = np.sqrt(
        predictions["uncertainty"] ** 2 + bootstrap["boot_uncertainty"] ** 2
    )

    valid = actual > 0
    msg(f"[RESIDUALS] Mean residual            : "
        f"{np.nanmean(residual[valid]):+.8f} mol/m²")
    msg(f"[RESIDUALS] Std residual             : "
        f"{np.nanstd(residual[valid]):.8f} mol/m²")
    msg(f"[RESIDUALS] Median residual          : "
        f"{np.nanmedian(residual[valid]):+.8f} mol/m²")
    msg(f"[RESIDUALS] Mean quantile unc        : "
        f"{np.nanmean(predictions['uncertainty'][valid]):.8f} mol/m²")
    msg(f"[RESIDUALS] Mean bootstrap unc       : "
        f"{np.nanmean(bootstrap['boot_uncertainty'][valid]):.8f} mol/m²")
    msg(f"[RESIDUALS] Mean combined unc        : "
        f"{np.nanmean(combined_unc[valid]):.8f} mol/m²")
    msg(f"[RESIDUALS] Positive pixels (excess) : "
        f"{np.sum(residual[valid] > 0):,} / {np.sum(valid):,}")

    return {
        "residual":            residual,
        "residual_lower":      resid_lower,
        "residual_upper":      resid_upper,
        "model_uncertainty":   predictions["uncertainty"],
        "combined_uncertainty": combined_unc,
    }


# ---------------------------------------------------------------------------
# MODULE 5 — SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------

def buffer_sensitivity_analysis(df, pollutant):
    results = []
    msg(f"[SENSITIVITY] Testing {len(Config.SENSITIVITY_DISTANCES)} "
        f"background threshold distances: {Config.SENSITIVITY_DISTANCES} km")
    msg(f"[SENSITIVITY] Confirms that chosen {Config.BACKGROUND_THRESHOLD_KM} km "
        f"threshold does not drive results...")

    for dist_km in Config.SENSITIVITY_DISTANCES:
        dist_m = dist_km * 1000
        df_e   = df.copy()
        df_e["month_sin"] = np.sin(2 * np.pi * df_e["month"] / 12)
        df_e["month_cos"] = np.cos(2 * np.pi * df_e["month"] / 12)
        leaky = {pollutant}
        av    = [p for p in Config.PREDICTORS if p in df_e.columns and p not in leaky]

        train = df_e[df_e["dist_from_refinery"] > dist_m].dropna(subset=[pollutant] + av)
        train = train[train[pollutant] > 0]
        if len(train) < 50:
            msg(f"  {dist_km}km: insufficient samples ({len(train)}) — skipped")
            continue

        X, y = train[av].values, train[pollutant].values
        rf = RandomForestRegressor(
            n_estimators=150, max_features="sqrt",
            min_samples_split=5, oob_score=True,
            random_state=Config.RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X, y)

        all_d   = df_e.dropna(subset=av)
        resid   = all_d[pollutant].values - rf.predict(all_d[av].values)
        near    = all_d["dist_from_refinery"] < 5000
        mean_nr = np.mean(resid[near.values])

        results.append({
            "distance_km": dist_km,
            "n_training":  len(train),
            "oob_r2":      rf.oob_score_,
            "mean_residual_near_refinery": mean_nr,
        })
        msg(f"  {dist_km:2d} km: n={len(train):5d}  "
            f"OOB R²={rf.oob_score_:.4f}  "
            f"Near-refinery residual={mean_nr:+.8f}")

    return pd.DataFrame(results) if results else None


# ---------------------------------------------------------------------------
# MODULE 6 — REGULATORY COMPARISON (both sites only)
# ---------------------------------------------------------------------------

def regulatory_comparison(res1, res2, name1, name2):
    r1 = res1[~np.isnan(res1) & (res1 != 0)]
    r2 = res2[~np.isnan(res2) & (res2 != 0)]
    u_stat, p_val = stats.mannwhitneyu(r1, r2, alternative="two-sided")
    n1, n2  = len(r1), len(r2)
    r_eff   = 1 - (2 * u_stat) / (n1 * n2)
    mean1, mean2 = np.mean(r1), np.mean(r2)

    msg(f"\n{'='*60}")
    msg(f"REGULATORY COMPARISON — {name1} vs {name2}")
    msg(f"{'='*60}")
    msg(f"  {name1:10s} mean residual : {mean1:+.8f} mol/m²")
    msg(f"  {name2:10s} mean residual : {mean2:+.8f} mol/m²")
    msg(f"  Direction : {'Doura HIGHER — industrial excess confirmed' if mean1 > mean2 else 'No clear directional excess'}")
    msg(f"  Mann-Whitney U  = {u_stat:.0f}")
    msg(f"  p-value         = {p_val:.4e}  "
        f"({'SIGNIFICANT at α=0.05' if p_val < 0.05 else 'NOT significant'})")
    msg(f"  Effect size (r) = {r_eff:.4f}  "
        f"({'small but genuine — large N' if abs(r_eff) < 0.1 else 'moderate–large'})")
    return {"mean1": mean1, "mean2": mean2, "u": u_stat, "p": p_val, "r": r_eff}


# ---------------------------------------------------------------------------
# MODULE 7 — GDB EXPORT
# ---------------------------------------------------------------------------

def _remove_map_layer(fc_name):
    """Remove any map layer matching fc_name BEFORE deleting it from the GDB.
    Prevents broken/dangling layer references in the Contents pane."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap
        if active_map:
            for lyr in active_map.listLayers():
                if lyr.name == fc_name:
                    active_map.removeLayer(lyr)
    except Exception:
        pass


def _add_map_layer(fc_path):
    """Add a feature class to the active map by path."""
    try:
        aprx = arcpy.mp.ArcGISProject("CURRENT")
        active_map = aprx.activeMap
        if active_map:
            active_map.addDataFromPath(fc_path)
    except Exception:
        pass


def export_to_gdb(df, fc_name, gdb_path):
    fc_path = os.path.join(gdb_path, fc_name)
    arcpy.SetProgressorLabel(f"Writing {fc_name} to GDB…")

    # Remove existing map layer FIRST — before deleting from GDB —
    # so the Contents pane never shows a broken reference
    _remove_map_layer(fc_name)

    if arcpy.Exists(fc_path):
        arcpy.management.Delete(fc_path)

    sr = arcpy.SpatialReference(4326)
    arcpy.management.CreateFeatureclass(gdb_path, fc_name, "POINT", spatial_reference=sr)

    field_defs = [
        ("year",           "LONG"),
        ("month",          "LONG"),
        ("residual",       "DOUBLE"),
        ("residual_lower", "DOUBLE"),
        ("residual_upper", "DOUBLE"),
        ("uncertainty",    "DOUBLE"),
    ]
    for fname, ftype in field_defs:
        arcpy.management.AddField(fc_path, fname, ftype)

    with arcpy.da.InsertCursor(fc_path, ["SHAPE@XY"] + [f[0] for f in field_defs]) as cur:
        for _, row in df.iterrows():
            cur.insertRow([
                (float(row["lon"]), float(row["lat"])),
                int(row.get("year",  0)),
                int(row.get("month", 0)),
                float(row.get("residual",       np.nan)),
                float(row.get("residual_lower", np.nan)),
                float(row.get("residual_upper", np.nan)),
                float(row.get("uncertainty",    np.nan)),
            ])

    # Add the freshly written FC back to the map immediately
    _add_map_layer(fc_path)
    msg(f"[EXPORT] {len(df)} features written → {fc_name}")
    return fc_path


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def run_single_site(pollutant):
    msg(f"\n{'='*60}")
    msg(f"IESUT  |  {_site_key.upper()}  |  {pollutant.upper()}")
    msg(f"{'='*60}")

    arcpy.SetProgressorLabel(f"[1/8] Loading data…")
    df = load_data()
    if df is None or len(df) == 0:
        arcpy.AddError(f"No data loaded from {os.path.basename(Config.DATA_FILE)}")
        return None

    arcpy.SetProgressorLabel(f"[2/8] Preparing training data…")
    X_train, y_train, X_predict, predict_df, features, train_df = \
        prepare_training_data(df, pollutant)

    arcpy.SetProgressorLabel(f"[3/8] Training Random Forest…")
    rf = train_rf(X_train, y_train)

    arcpy.SetProgressorLabel(f"[4/8] Cross-validation…")
    cv = cross_validate(rf, X_train, y_train, train_df=train_df)

    arcpy.SetProgressorLabel(f"[5/8] Feature importance…")
    imp = get_feature_importance(rf, X_train, y_train, features)

    arcpy.SetProgressorLabel(f"[6/8] Quantile uncertainty…")
    predictions = predict_with_uncertainty(rf, X_predict)

    arcpy.SetProgressorLabel(f"[7/8] Bootstrap uncertainty…")
    bootstrap = bootstrap_uncertainty(X_train, y_train, X_predict)

    arcpy.SetProgressorLabel(f"[8/8] Residuals + GDB export…")
    res = calculate_residuals(predict_df[pollutant].values, predictions, bootstrap)

    predict_df["residual"]        = res["residual"]
    predict_df["residual_lower"]  = res["residual_lower"]
    predict_df["residual_upper"]  = res["residual_upper"]
    predict_df["uncertainty"]     = res["combined_uncertainty"]

    if Config.RUN_SENSITIVITY:
        arcpy.SetProgressorLabel(f"[+] Sensitivity analysis…")
        buffer_sensitivity_analysis(df, pollutant)
    else:
        msg("[SENSITIVITY] Skipped")

    export_cols = ["lon", "lat", "year", "month",
                   "residual", "residual_lower", "residual_upper", "uncertainty"]
    export_cols = [c for c in export_cols if c in predict_df.columns]
    fc_path = export_to_gdb(
        predict_df[export_cols], f"IESUT_{_site_key}_{pollutant}", Config.GDB_PATH
    )

    valid = predict_df[pollutant].values > 0
    return {
        "rf": rf, "cv": cv, "oob_r2": rf.oob_score_,
        "importance": imp,
        "predictions": predictions,
        "bootstrap": bootstrap,
        "predict_df": predict_df,
        "df": df,
        "fc_path": fc_path,
        "residual_stats": {
            "mean": float(np.nanmean(res["residual"][valid])),
            "std":  float(np.nanstd(res["residual"][valid])),
        },
    }


def run_analysis():
    msg("=" * 60)
    msg("IESUT — Industrial Emission Separator & Uncertainty Tool")
    msg("=" * 60)
    msg(f"Site                : {_site_key}")
    msg(f"Pollutant(s)        : {POLLUTANTS_TO_RUN}")
    msg(f"Background threshold: {Config.BACKGROUND_THRESHOLD_KM} km")
    msg(f"RF trees            : {Config.N_TREES}")
    msg(f"Bootstrap iterations: {Config.N_BOOTSTRAP}")
    msg(f"Confidence interval : {int(_ci_val)}% "
        f"({Config.CI_LOWER:.1f}th–{Config.CI_UPPER:.1f}th percentile)")
    msg(f"Sensitivity analysis: {Config.RUN_SENSITIVITY}")
    msg("")

    arcpy.SetProgressor("step", "Running IESUT…", 0, len(POLLUTANTS_TO_RUN) * 8, 1)

    output_fc_paths = []

    for pollutant in POLLUTANTS_TO_RUN:
        result = run_single_site(pollutant)
        if result:
            output_fc_paths.append(result["fc_path"])
            arcpy.SetProgressorPosition()

    msg(f"\n{'='*60}")
    msg("ANALYSIS COMPLETE")
    msg(f"Site : {_site_key}")
    msg(f"GDB  : {Config.GDB_PATH}")
    msg("Feature classes written:")
    for p in output_fc_paths:
        msg(f"  {os.path.basename(p)}")
    msg("")
    msg("Next step: run IESUT Charts tool to generate all figures")
    msg(f"{'='*60}")

    if output_fc_paths:
        arcpy.SetParameter(9, output_fc_paths[-1])


run_analysis()
