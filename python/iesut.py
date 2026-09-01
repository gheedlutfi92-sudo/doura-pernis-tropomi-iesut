"""
IESUT — Industrial Emission Separator & Uncertainty Tool
=========================================================
MSc Dissertation: Ghid Albazrkan, University of Aberdeen (GG5910/GG5912)

Purpose: Separate industrial atmospheric emissions (NO2, SO2) from urban
         background using RF-LUR with uncertainty quantification.

Usage:   Run inside ArcGIS Pro Python environment or standalone with arcpy.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION — user sets these parameters
# ---------------------------------------------------------------------------

class Config:
    """All user-configurable parameters in one place."""

    # Paths
    GDB_PATH = os.path.join(os.environ.get("IESUT_DATA_DIR", "./data"), "dissertation_final.gdb")
    OUTPUT_DIR = os.path.join(os.environ.get("IESUT_DATA_DIR", "./data"), "outputs")
    WIND_DATA_PATH = os.path.join(os.environ.get("IESUT_DATA_DIR", "./data"), "meteorological_data", "meteorological_dataset_baghdad.xlsx")

    # Site definitions
    SITES = {
        "doura": {
            "name": "Doura Refinery",
            "lon": 44.427,
            "lat": 33.264,
            "layer": "doura_monthly_points",
            "country": "Iraq",
            "regulated": False,
        },
        "pernis": {
            "name": "Shell Pernis Refinery",
            "lon": 4.334,
            "lat": 51.888,
            "layer": "pernis_monthly_points",
            "country": "Netherlands",
            "regulated": True,
        },
    }

    # Analysis parameters
    BUFFER_KM = 50
    BACKGROUND_THRESHOLD_KM = 12
    POLLUTANTS = ["no2", "so2"]
    PREDICTORS = [
        "viirs_ntl", "population", "elevation",
        "lst_celsius", "wind_speed", "temperature_c",
        "dist_from_refinery",
        "month_sin", "month_cos",
    ]

    # RF parameters
    N_TREES = 300
    CV_FOLDS = 5
    RANDOM_STATE = 42

    # Sensitivity analysis buffer distances (km)
    SENSITIVITY_DISTANCES = [8, 10, 12, 14, 16, 18, 20, 25]

    # Wind sectors
    WIND_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    WIND_SECTOR_BOUNDS = [
        (337.5, 22.5), (22.5, 67.5), (67.5, 112.5), (112.5, 157.5),
        (157.5, 202.5), (202.5, 247.5), (247.5, 292.5), (292.5, 337.5),
    ]

    # WHO guidelines (annual mean, µmol/m²)
    WHO_NO2_GUIDELINE = 0.0000532  # 10 µg/m³ converted to approx mol/m²
    WHO_SO2_GUIDELINE = 0.0000625  # 40 µg/m³ 24-hour converted


# ---------------------------------------------------------------------------
# MODULE 1: DATA PREPARATION
# ---------------------------------------------------------------------------

def load_data_from_gdb(layer_name, gdb_path=None):
    """Read point features from geodatabase into pandas DataFrame."""
    try:
        import arcpy
        gdb = gdb_path or Config.GDB_PATH
        fc = os.path.join(gdb, layer_name)
        fields = [f.name for f in arcpy.ListFields(fc)
                  if f.type not in ("Geometry", "OID")]
        data = [row for row in arcpy.da.SearchCursor(fc, fields)]
        return pd.DataFrame(data, columns=fields)
    except Exception as e:
        print(f"[INFO] Could not read from geodatabase: {e}")
        print("[INFO] Falling back to CSV...")
        return None


def load_data_from_csv(csv_path):
    """Read TROPOMI data from CSV file."""
    return pd.read_csv(csv_path)


def prepare_training_data(df, pollutant, threshold_km=None):
    """
    Split data into background training set and full prediction set.
    Background = pixels > threshold_km from refinery (urban only).
    """
    threshold = threshold_km or Config.BACKGROUND_THRESHOLD_KM
    threshold_m = threshold * 1000

    # Feature engineering — compute derived predictors
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["so2_no2_ratio"] = np.where(
            df["no2"] > 0, df["so2"] / df["no2"], 0.0
        )

    # Remove target pollutant and any feature derived from it (prevents leakage)
    leaky = {pollutant}
    if pollutant == "so2":
        leaky.add("so2_no2_ratio")

    available_predictors = [
        p for p in Config.PREDICTORS if p in df.columns and p not in leaky
    ]
    if len(available_predictors) < len(Config.PREDICTORS) - len(leaky):
        missing = set(Config.PREDICTORS) - set(df.columns) - leaky
        if missing:
            print(f"[WARN] Missing predictors: {missing}")

    mask_background = df["dist_from_refinery"] > threshold_m
    train_df = df[mask_background].copy()
    predict_df = df.copy()

    train_df = train_df.dropna(subset=[pollutant] + available_predictors)
    predict_df = predict_df.dropna(subset=available_predictors)
    predict_df = predict_df.reset_index(drop=True)

    # Remove rows where pollutant is zero (poor TROPOMI coverage months)
    train_df = train_df[train_df[pollutant] > 0]
    train_df = train_df.reset_index(drop=True)

    X_train = train_df[available_predictors].values
    y_train = train_df[pollutant].values
    X_predict = predict_df[available_predictors].values

    print(f"[DATA] Training samples: {len(X_train)} | "
          f"Prediction samples: {len(X_predict)} | "
          f"Predictors: {len(available_predictors)}")

    return X_train, y_train, X_predict, predict_df, available_predictors, train_df


# ---------------------------------------------------------------------------
# MODULE 2: QUANTILE RANDOM FOREST MODEL
# ---------------------------------------------------------------------------

def train_quantile_rf(X_train, y_train, n_trees=None, random_state=None):
    """
    Train a Random Forest and extract individual tree predictions
    for quantile-based uncertainty estimation.
    """
    n = n_trees or Config.N_TREES
    rs = random_state or Config.RANDOM_STATE

    rf = RandomForestRegressor(
        n_estimators=n,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        oob_score=True,
        random_state=rs,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    print(f"[MODEL] OOB Score (R²): {rf.oob_score_:.4f}")
    return rf


def predict_with_uncertainty(rf, X_predict):
    """
    Get predictions from each tree in the forest to calculate
    median prediction and confidence intervals.
    """
    # Extract predictions from every individual tree
    tree_predictions = np.array(
        [tree.predict(X_predict) for tree in rf.estimators_]
    )

    # Calculate quantiles across trees
    pred_lower = np.percentile(tree_predictions, 5, axis=0)   # 5th percentile
    pred_median = np.percentile(tree_predictions, 50, axis=0)  # median
    pred_upper = np.percentile(tree_predictions, 95, axis=0)   # 95th percentile
    pred_mean = np.mean(tree_predictions, axis=0)

    # Uncertainty = width of 90% confidence interval
    uncertainty = pred_upper - pred_lower

    return {
        "predicted_mean": pred_mean,
        "predicted_median": pred_median,
        "predicted_lower": pred_lower,
        "predicted_upper": pred_upper,
        "uncertainty": uncertainty,
    }


def cross_validate(rf, X_train, y_train, cv_folds=None, train_df=None):
    """
    Temporally blocked cross-validation (Roberts et al., 2017).
    Data is sorted by year/month; consecutive blocks are held out as
    validation sets so no temporal leakage occurs between folds.
    """
    k = cv_folds or Config.CV_FOLDS

    r2_scores = []
    rmse_scores = []
    mae_scores = []

    n = len(X_train)

    # Build temporal sort order from train_df if available
    if train_df is not None and "year" in train_df.columns and "month" in train_df.columns:
        time_key = train_df["year"] * 100 + train_df["month"]
        sort_idx = np.argsort(time_key.values)
        print(f"[CV] Temporally blocked CV ({k} folds) — sorted by year/month")
    else:
        sort_idx = np.arange(n)
        print(f"[CV] Temporally blocked CV ({k} folds) — no time info, using row order")

    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]

    fold_size = n // k

    for fold in range(k):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < k - 1 else n

        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_start:val_end] = True

        X_t, X_v = X_sorted[~val_mask], X_sorted[val_mask]
        y_t, y_v = y_sorted[~val_mask], y_sorted[val_mask]

        fold_rf = RandomForestRegressor(
            n_estimators=Config.N_TREES,
            max_features="sqrt",
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
        )
        fold_rf.fit(X_t, y_t)
        y_pred = fold_rf.predict(X_v)

        r2_scores.append(r2_score(y_v, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_v, y_pred)))
        mae_scores.append(mean_absolute_error(y_v, y_pred))

        print(f"  Fold {fold+1} (rows {val_start}–{val_end-1}): "
              f"R²={r2_scores[-1]:.4f}  "
              f"RMSE={rmse_scores[-1]:.6f}  MAE={mae_scores[-1]:.6f}")

    results = {
        "r2_mean": np.mean(r2_scores), "r2_std": np.std(r2_scores),
        "rmse_mean": np.mean(rmse_scores), "rmse_std": np.std(rmse_scores),
        "mae_mean": np.mean(mae_scores), "mae_std": np.std(mae_scores),
        "r2_scores": r2_scores,
        "rmse_scores": rmse_scores,
    }
    print(f"[CV] Mean R²: {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
    return results


def get_feature_importance(rf, X_train, y_train, feature_names):
    """Calculate both impurity-based and permutation-based importance."""
    # Impurity-based (built into RF)
    impurity_importance = rf.feature_importances_

    # Permutation-based (more reliable)
    perm_result = permutation_importance(
        rf, X_train, y_train,
        n_repeats=30,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "impurity_importance": impurity_importance,
        "permutation_importance": perm_result.importances_mean,
        "permutation_std": perm_result.importances_std,
    }).sort_values("permutation_importance", ascending=False)

    print("[IMPORTANCE] Feature ranking (permutation-based):")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']:25s} {row['permutation_importance']:.6f} "
              f"± {row['permutation_std']:.6f}")

    return importance_df


# ---------------------------------------------------------------------------
# MODULE 3: UNCERTAINTY QUANTIFICATION
# ---------------------------------------------------------------------------

def bootstrap_uncertainty(X_train, y_train, X_predict, n_bootstraps=100):
    """
    Bootstrap resampling: train RF on resampled data multiple times
    to estimate prediction uncertainty.
    """
    n_samples = len(X_train)
    bootstrap_predictions = np.zeros((n_bootstraps, len(X_predict)))

    print(f"[BOOTSTRAP] Running {n_bootstraps} iterations...")
    for i in range(n_bootstraps):
        # Resample training data with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_train[idx]
        y_boot = y_train[idx]

        rf_boot = RandomForestRegressor(
            n_estimators=100,
            max_features="sqrt",
            min_samples_split=5,
            random_state=i,
            n_jobs=-1,
        )
        rf_boot.fit(X_boot, y_boot)
        bootstrap_predictions[i] = rf_boot.predict(X_predict)

        if (i + 1) % 25 == 0:
            print(f"  Completed {i + 1}/{n_bootstraps}")

    boot_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
    boot_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)
    boot_std = np.std(bootstrap_predictions, axis=0)

    return {
        "boot_lower": boot_lower,
        "boot_upper": boot_upper,
        "boot_std": boot_std,
        "boot_uncertainty": boot_upper - boot_lower,
    }


def calculate_residuals_with_uncertainty(actual, predictions, bootstrap=None):
    """
    Calculate residuals and propagate uncertainty.
    Residual = actual - predicted (industrial signal).
    """
    residual = actual - predictions["predicted_median"]
    residual_lower = actual - predictions["predicted_upper"]
    residual_upper = actual - predictions["predicted_lower"]

    results = {
        "residual": residual,
        "residual_lower": residual_lower,
        "residual_upper": residual_upper,
        "residual_uncertainty": residual_upper - residual_lower,
        "model_uncertainty": predictions["uncertainty"],
    }

    if bootstrap is not None:
        results["bootstrap_uncertainty"] = bootstrap["boot_uncertainty"]
        # Combined uncertainty (model + bootstrap)
        results["combined_uncertainty"] = np.sqrt(
            predictions["uncertainty"] ** 2 +
            bootstrap["boot_uncertainty"] ** 2
        )

    return results


# ---------------------------------------------------------------------------
# MODULE 4: SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------

def buffer_sensitivity_analysis(df, pollutant, distances=None):
    """
    Test how results change with different background threshold distances.
    Proves the methodology is robust (or identifies optimal distance).
    """
    distances = distances or Config.SENSITIVITY_DISTANCES
    results = []

    print(f"[SENSITIVITY] Testing {len(distances)} buffer distances...")

    for dist_km in distances:
        dist_m = dist_km * 1000
        df_eng = df.copy()
        df_eng["month_sin"] = np.sin(2 * np.pi * df_eng["month"] / 12)
        df_eng["month_cos"] = np.cos(2 * np.pi * df_eng["month"] / 12)
        with np.errstate(divide="ignore", invalid="ignore"):
            df_eng["so2_no2_ratio"] = np.where(
                df_eng["no2"] > 0, df_eng["so2"] / df_eng["no2"], 0.0
            )
        leaky = {pollutant}
        if pollutant == "so2":
            leaky.add("so2_no2_ratio")
        available_predictors = [
            p for p in Config.PREDICTORS if p in df_eng.columns and p not in leaky
        ]

        mask_bg = df_eng["dist_from_refinery"] > dist_m
        train_data = df_eng[mask_bg].dropna(subset=[pollutant] + available_predictors)
        train_data = train_data[train_data[pollutant] > 0]

        if len(train_data) < 50:
            print(f"  {dist_km}km: Too few training samples ({len(train_data)})")
            continue

        X = train_data[available_predictors].values
        y = train_data[pollutant].values

        rf = RandomForestRegressor(
            n_estimators=150,
            max_features="sqrt",
            min_samples_split=5,
            oob_score=True,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
        )
        rf.fit(X, y)

        # Predict all pixels and calculate mean residual near refinery (<5km)
        all_data = df_eng.dropna(subset=available_predictors)
        X_all = all_data[available_predictors].values
        y_all = all_data[pollutant].values
        y_pred = rf.predict(X_all)
        residuals = y_all - y_pred

        near_refinery = all_data["dist_from_refinery"] < 5000
        mean_residual_near = np.mean(residuals[near_refinery.values])

        cv_scores = cross_val_score(
            rf, X, y, cv=3, scoring="r2"
        )

        results.append({
            "distance_km": dist_km,
            "n_training": len(train_data),
            "oob_r2": rf.oob_score_,
            "cv_r2_mean": np.mean(cv_scores),
            "cv_r2_std": np.std(cv_scores),
            "mean_residual_near_refinery": mean_residual_near,
        })

        print(f"  {dist_km:2d}km: n={len(train_data):5d}  "
              f"OOB R²={rf.oob_score_:.4f}  "
              f"CV R²={np.mean(cv_scores):.4f}  "
              f"Mean residual (0-5km)={mean_residual_near:.8f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# MODULE 5: WIND-ADJUSTED RESIDUAL ANALYSIS
# ---------------------------------------------------------------------------

def load_wind_data(path=None):
    """Load Abu Ghraib meteorological station data."""
    wind_path = path or Config.WIND_DATA_PATH
    try:
        df_raw = pd.read_excel(wind_path, header=None)
        # Find the English header row (contains "Date")
        header_row = None
        for i in range(min(10, len(df_raw))):
            row_vals = [str(v) for v in df_raw.iloc[i].values]
            if any("Date" in v for v in row_vals):
                header_row = i
                break
        if header_row is None:
            header_row = 4  # fallback

        wind_df = pd.read_excel(wind_path, header=header_row)
        # Clean column names (remove newlines)
        wind_df.columns = [str(c).replace("\n", " ").strip() for c in wind_df.columns]
        # Rename to standard names
        col_map = {}
        for c in wind_df.columns:
            cl = c.lower()
            if "date" in cl:
                col_map[c] = "date"
            elif "ws avg" in cl or "wind speed" in cl or ("ws" in cl and "avg" in cl):
                col_map[c] = "wind_speed_avg"
            elif "ws max" in cl or ("ws" in cl and "max" in cl):
                col_map[c] = "wind_speed_max"
            elif "wd" in cl or "wind dir" in cl:
                col_map[c] = "wind_direction"
            elif "at avg" in cl or "temp avg" in cl:
                col_map[c] = "temp_avg"
        wind_df = wind_df.rename(columns=col_map)
        wind_df = wind_df.dropna(subset=["date"])
        wind_df["date"] = pd.to_datetime(wind_df["date"], errors="coerce")
        wind_df = wind_df.dropna(subset=["date"])
        # Convert numeric columns
        for col in ["wind_speed_avg", "wind_speed_max", "wind_direction", "temp_avg"]:
            if col in wind_df.columns:
                wind_df[col] = pd.to_numeric(wind_df[col], errors="coerce")
        print(f"[WIND] Loaded {len(wind_df)} records from met station")
        ascii_cols = [c for c in wind_df.columns if c.isascii()]
        print(f"[WIND] Usable columns: {ascii_cols}")
        return wind_df
    except Exception as e:
        print(f"[WIND] Could not load wind data: {e}")
        return None


def classify_wind_direction(degrees):
    """Convert wind direction in degrees to 8-sector classification."""
    try:
        if pd.isna(degrees):
            return "CALM"
        degrees = float(degrees) % 360
    except (TypeError, ValueError):
        return "CALM"
    sectors = Config.WIND_SECTORS
    bounds = Config.WIND_SECTOR_BOUNDS
    for sector, (low, high) in zip(sectors, bounds):
        if low > high:  # wraps around 0 (N sector)
            if degrees >= low or degrees < high:
                return sector
        else:
            if low <= degrees < high:
                return sector
    return "N"


def wind_adjusted_analysis(df, residuals, wind_df, pollutant):
    """
    Analyse how residuals vary by wind direction.
    Groups residuals by prevailing wind direction per month.
    """
    if wind_df is None:
        print("[WIND] No wind data available — skipping wind analysis")
        return None

    # Use standardised column names from load_wind_data
    wind_df["year"] = wind_df["date"].dt.year
    wind_df["month"] = wind_df["date"].dt.month

    dir_col = "wind_direction" if "wind_direction" in wind_df.columns else None
    speed_col = "wind_speed_avg" if "wind_speed_avg" in wind_df.columns else None

    if dir_col is None:
        print("[WIND] No wind direction column found")
        return None

    # Monthly prevailing wind direction
    agg_dict = {dir_col: "mean"}
    if speed_col and speed_col in wind_df.columns:
        agg_dict[speed_col] = "mean"
    monthly_wind = wind_df.groupby(["year", "month"]).agg(agg_dict).reset_index()

    monthly_wind["wind_sector"] = monthly_wind[dir_col].apply(classify_wind_direction)

    # Merge with TROPOMI data
    df_with_residual = df.copy()
    df_with_residual["residual"] = residuals

    if "year" in df_with_residual.columns and "month" in df_with_residual.columns:
        merged = df_with_residual.merge(
            monthly_wind[["year", "month", "wind_sector", dir_col]],
            on=["year", "month"],
            how="left",
        )

        # Calculate mean residual per wind sector
        sector_stats = merged.groupby("wind_sector").agg(
            mean_residual=("residual", "mean"),
            std_residual=("residual", "std"),
            count=("residual", "count"),
        ).reset_index()

        print("[WIND] Residual by wind sector:")
        for _, row in sector_stats.iterrows():
            print(f"  {row['wind_sector']:3s}: mean={row['mean_residual']:.8f}  "
                  f"std={row['std_residual']:.8f}  n={row['count']}")

        return {"sector_stats": sector_stats, "merged": merged}

    return None


# ---------------------------------------------------------------------------
# MODULE 6: REGULATORY SCENARIO MODELLING
# ---------------------------------------------------------------------------

def regulatory_comparison(results_site1, results_site2, site1_name, site2_name):
    """
    Compare two sites and model regulatory scenarios.
    Site1 = unregulated (Doura), Site2 = regulated (Pernis).
    """
    res1 = results_site1["residual"]
    res2 = results_site2["residual"]

    # Remove NaN and zero values for comparison
    res1_clean = res1[~np.isnan(res1) & (res1 != 0)]
    res2_clean = res2[~np.isnan(res2) & (res2 != 0)]

    # Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(res1_clean, res2_clean, alternative="two-sided")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(res1_clean), len(res2_clean)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)

    # Percentage difference
    mean1 = np.mean(res1_clean)
    mean2 = np.mean(res2_clean)
    pct_diff = ((mean1 - mean2) / mean2) * 100 if mean2 != 0 else np.inf

    print(f"\n[REGULATORY COMPARISON] {site1_name} vs {site2_name}")
    print(f"  {site1_name} mean residual: {mean1:.8f}")
    print(f"  {site2_name} mean residual: {mean2:.8f}")
    print(f"  Difference: {pct_diff:+.1f}%")
    print(f"  Mann-Whitney U: {u_stat:.0f}, p = {p_value:.2e}")
    print(f"  Effect size (r): {effect_size:.4f}")

    return {
        "mean_residual_site1": mean1,
        "mean_residual_site2": mean2,
        "pct_difference": pct_diff,
        "mann_whitney_u": u_stat,
        "p_value": p_value,
        "effect_size": effect_size,
    }


def project_regulatory_scenario(df_unregulated, df_regulated, pollutant):
    """
    Project: if unregulated site adopted regulation, how would emissions change?
    Uses regulated site's annual reduction rate as the model.
    """
    # Calculate annual trends for regulated site
    reg_annual = df_regulated.groupby("year")[pollutant].mean()
    if len(reg_annual) < 2:
        print("[SCENARIO] Not enough years for projection")
        return None

    # Mann-Kendall trend for regulated site
    years = reg_annual.index.values.astype(float)
    values = reg_annual.values
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
    annual_reduction_rate = slope / np.mean(values)  # fractional change per year

    # Calculate annual trend for unregulated site
    unreg_annual = df_unregulated.groupby("year")[pollutant].mean()
    unreg_slope, _, _, unreg_p, _ = stats.linregress(
        unreg_annual.index.values.astype(float), unreg_annual.values
    )

    # Project unregulated site with regulation applied
    last_year = int(unreg_annual.index.max())
    last_value = unreg_annual.iloc[-1]
    projection_years = list(range(last_year + 1, last_year + 16))  # 15-year projection
    projected_values = []

    current = last_value
    for yr in projection_years:
        current = current * (1 + annual_reduction_rate)
        projected_values.append(current)
        if current <= 0:
            break

    # Years to reach WHO guideline
    guideline = Config.WHO_NO2_GUIDELINE if pollutant == "no2" else Config.WHO_SO2_GUIDELINE
    years_to_compliance = None
    for i, val in enumerate(projected_values):
        if val <= guideline:
            years_to_compliance = i + 1
            break

    print(f"\n[SCENARIO] Regulatory projection for unregulated site:")
    print(f"  Regulated site annual rate: {annual_reduction_rate*100:.2f}% per year")
    print(f"  Unregulated site current trend: {unreg_slope:.10f} per year (p={unreg_p:.4f})")
    print(f"  Current emission level: {last_value:.8f}")
    if years_to_compliance:
        print(f"  Years to WHO compliance: {years_to_compliance}")
    else:
        print(f"  WHO compliance: Not achievable in 15 years at regulated site's rate")

    return {
        "regulated_annual_rate": annual_reduction_rate,
        "unregulated_trend_slope": unreg_slope,
        "unregulated_trend_p": unreg_p,
        "projection_years": projection_years,
        "projected_values": projected_values,
        "years_to_compliance": years_to_compliance,
        "who_guideline": guideline,
    }


# ---------------------------------------------------------------------------
# MODULE 7: VISUALISATION
# ---------------------------------------------------------------------------

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


def plot_feature_importance(importance_df, site_name, pollutant):
    """Bar chart of feature importance with error bars."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    importance_df_sorted = importance_df.sort_values(
        "permutation_importance", ascending=True
    )
    ax.barh(
        importance_df_sorted["feature"],
        importance_df_sorted["permutation_importance"],
        xerr=importance_df_sorted["permutation_std"],
        color="#2196F3", edgecolor="black", capsize=3,
    )
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Feature Importance — {site_name} ({pollutant.upper()})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(Config.OUTPUT_DIR,
                        f"feature_importance_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_predicted_vs_actual(y_actual, y_predicted, site_name, pollutant):
    """Scatter plot of predicted vs actual values."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_actual, y_predicted, alpha=0.3, s=10, c="#2196F3")
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")

    r2 = r2_score(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.6f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xlabel(f"Actual {pollutant.upper()}")
    ax.set_ylabel(f"Predicted {pollutant.upper()}")
    ax.set_title(f"Predicted vs Actual — {site_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(Config.OUTPUT_DIR,
                        f"predicted_vs_actual_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_residual_distribution(residuals, site_name, pollutant):
    """Histogram of residual values with normal curve overlay."""
    ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(residuals, bins=50, density=True, alpha=0.7,
            color="#FF9800", edgecolor="black", label="Residuals")

    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
            label=f"Normal (μ={mu:.6f}, σ={sigma:.6f})")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero line")
    ax.set_xlabel(f"Residual ({pollutant.upper()})")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution — {site_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(Config.OUTPUT_DIR,
                        f"residual_distribution_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_uncertainty_map(df, uncertainty, site_name, pollutant):
    """Scatter plot showing spatial uncertainty at each pixel."""
    ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Residual map
    residual_clean = df["residual"].dropna()
    vlim = np.percentile(np.abs(residual_clean), 95) if len(residual_clean) > 0 else 1
    sc1 = axes[0].scatter(
        df["lon"], df["lat"], c=df["residual"],
        cmap="RdBu_r", s=8, alpha=0.7,
        vmin=-vlim, vmax=vlim,
    )
    axes[0].set_title(f"Industrial Residual — {site_name}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axes[0], label=f"Residual ({pollutant.upper()})")

    # Uncertainty map
    sc2 = axes[1].scatter(
        df["lon"], df["lat"], c=uncertainty,
        cmap="YlOrRd", s=8, alpha=0.7,
    )
    axes[1].set_title(f"Prediction Uncertainty — {site_name}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(sc2, ax=axes[1], label="90% CI Width")

    plt.tight_layout()
    path = os.path.join(Config.OUTPUT_DIR,
                        f"uncertainty_map_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_sensitivity(sensitivity_df, site_name, pollutant):
    """Plot R² and mean residual vs buffer distance."""
    ensure_output_dir()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # R² vs distance
    ax1.errorbar(
        sensitivity_df["distance_km"], sensitivity_df["cv_r2_mean"],
        yerr=sensitivity_df["cv_r2_std"], marker="o", capsize=5,
        color="#2196F3", linewidth=2,
    )
    ax1.set_xlabel("Background Threshold Distance (km)")
    ax1.set_ylabel("Cross-Validation R²")
    ax1.set_title(f"Model Performance vs Buffer Distance — {site_name}")
    ax1.grid(alpha=0.3)

    # Mean residual near refinery vs distance
    ax2.plot(
        sensitivity_df["distance_km"],
        sensitivity_df["mean_residual_near_refinery"],
        marker="s", color="#FF5722", linewidth=2,
    )
    ax2.set_xlabel("Background Threshold Distance (km)")
    ax2.set_ylabel("Mean Residual (0-5km from refinery)")
    ax2.set_title(f"Industrial Signal Stability — {site_name}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(Config.OUTPUT_DIR,
                        f"sensitivity_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_wind_rose(sector_stats, site_name, pollutant):
    """Polar plot showing residual magnitude by wind direction."""
    if sector_stats is None:
        return
    ensure_output_dir()

    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles = np.linspace(0, 2 * np.pi, len(sectors), endpoint=False)

    values = []
    for s in sectors:
        row = sector_stats[sector_stats["wind_sector"] == s]
        values.append(row["mean_residual"].values[0] if len(row) > 0 else 0)

    values = np.array(values)
    # Close the polygon
    angles = np.concatenate([angles, [angles[0]]])
    values = np.concatenate([values, [values[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, values, "o-", color="#FF5722", linewidth=2)
    ax.fill(angles, values, alpha=0.25, color="#FF5722")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(sectors)
    ax.set_title(f"Pollution Rose — {site_name}\nMean Residual by Wind Direction",
                 pad=20)

    path = os.path.join(Config.OUTPUT_DIR,
                        f"wind_rose_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def plot_regulatory_projection(scenario, site_name, pollutant,
                                df_unreg, df_reg):
    """Plot historical trends and future projection."""
    if scenario is None:
        return
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Historical — unregulated
    unreg_annual = df_unreg.groupby("year")[pollutant].mean()
    ax.plot(unreg_annual.index, unreg_annual.values,
            "o-", color="#F44336", linewidth=2, markersize=6,
            label=f"{site_name} (unregulated) — historical")

    # Historical — regulated
    reg_annual = df_reg.groupby("year")[pollutant].mean()
    ax.plot(reg_annual.index, reg_annual.values,
            "s-", color="#2196F3", linewidth=2, markersize=6,
            label="Pernis (regulated) — historical")

    # Projection
    ax.plot(scenario["projection_years"], scenario["projected_values"],
            "--", color="#F44336", linewidth=2, alpha=0.6,
            label=f"{site_name} — projected with regulation")

    # WHO guideline
    ax.axhline(scenario["who_guideline"], color="green", linestyle=":",
               linewidth=2, label=f"WHO Guideline ({pollutant.upper()})")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Mean {pollutant.upper()} (mol/m²)")
    ax.set_title(f"Regulatory Scenario — What if {site_name} adopted EU ETS?")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(Config.OUTPUT_DIR,
                        f"regulatory_projection_{site_name}_{pollutant}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Saved: {path}")


def generate_summary_report(all_results, output_dir=None):
    """Generate a text summary of all analyses."""
    out_dir = output_dir or Config.OUTPUT_DIR
    ensure_output_dir()
    path = os.path.join(out_dir, "IESUT_summary_report.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("IESUT — Industrial Emission Separator & Uncertainty Tool\n")
        f.write("Summary Report\n")
        f.write("=" * 70 + "\n\n")

        for site_key, site_results in all_results.items():
            site_name = Config.SITES[site_key]["name"]
            f.write(f"\n{'─' * 50}\n")
            f.write(f"SITE: {site_name}\n")
            f.write(f"{'─' * 50}\n\n")

            for pollutant, res in site_results.items():
                if pollutant in ("comparison", "scenario"):
                    continue
                f.write(f"  Pollutant: {pollutant.upper()}\n")

                if "cv" in res:
                    cv = res["cv"]
                    f.write(f"  Cross-Validation R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}\n")
                    f.write(f"  Cross-Validation RMSE: {cv['rmse_mean']:.6f} ± {cv['rmse_std']:.4f}\n")
                    f.write(f"  Cross-Validation MAE: {cv['mae_mean']:.6f}\n")

                if "oob_r2" in res:
                    f.write(f"  OOB R²: {res['oob_r2']:.4f}\n")

                if "residual_stats" in res:
                    rs = res["residual_stats"]
                    f.write(f"  Mean Residual: {rs['mean']:.8f}\n")
                    f.write(f"  Std Residual: {rs['std']:.8f}\n")
                    f.write(f"  Mean Uncertainty (90% CI): {rs['mean_uncertainty']:.8f}\n")

                f.write("\n")

            if "comparison" in site_results:
                comp = site_results["comparison"]
                f.write(f"  Regulatory Comparison:\n")
                f.write(f"    Difference: {comp['pct_difference']:+.1f}%\n")
                f.write(f"    Mann-Whitney p: {comp['p_value']:.2e}\n")
                f.write(f"    Effect size: {comp['effect_size']:.4f}\n\n")

    print(f"[REPORT] Saved: {path}")


# ---------------------------------------------------------------------------
# MODULE 8: GEODATABASE EXPORT (writes results back to ArcGIS)
# ---------------------------------------------------------------------------

def export_to_geodatabase(df, layer_name, gdb_path=None):
    """Export results as CSV (can be imported into ArcGIS Pro manually)."""
    csv_path = os.path.join(Config.OUTPUT_DIR, f"{layer_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXPORT] Saved CSV: {csv_path}")
    print(f"[EXPORT] To import into ArcGIS Pro: Map tab → Add Data → XY Table To Point")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

def run_single_site(site_key, pollutant="no2"):
    """Run complete IESUT analysis for one site and one pollutant."""
    site = Config.SITES[site_key]
    print(f"\n{'=' * 60}")
    print(f"IESUT — {site['name']} — {pollutant.upper()}")
    print(f"{'=' * 60}")

    # --- Load data ---
    print("\n[STEP 1] Loading data...")
    csv_dir = os.path.join(os.path.dirname(Config.GDB_PATH),
                           "..", "data", "TROPOMI_GEE_Exports")
    csv_file = f"{site_key}_monthly_samples_v33.csv"
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_file))

    if os.path.exists(csv_path):
        print(f"[DATA] Loading from CSV: {csv_file}")
        df = load_data_from_csv(csv_path)
    else:
        df = load_data_from_gdb(site["layer"])

    if df is None or len(df) == 0:
        print("[ERROR] No data loaded")
        return None

    # --- Prepare training data ---
    print("\n[STEP 2] Preparing training data...")
    X_train, y_train, X_predict, predict_df, features, train_df = prepare_training_data(
        df, pollutant
    )

    # --- Train Quantile RF ---
    print("\n[STEP 3] Training Quantile Random Forest...")
    rf = train_quantile_rf(X_train, y_train)

    # --- Cross-validation ---
    print("\n[STEP 4] Cross-validation (temporally blocked)...")
    cv_results = cross_validate(rf, X_train, y_train, train_df=train_df)

    # --- Feature importance ---
    print("\n[STEP 5] Feature importance...")
    importance = get_feature_importance(rf, X_train, y_train, features)
    plot_feature_importance(importance, site_key, pollutant)

    # --- Predict with uncertainty ---
    print("\n[STEP 6] Predicting with uncertainty...")
    predictions = predict_with_uncertainty(rf, X_predict)

    # --- Predicted vs Actual ---
    y_actual = predict_df[pollutant].values
    valid_mask = y_actual > 0
    plot_predicted_vs_actual(
        y_actual[valid_mask],
        predictions["predicted_mean"][valid_mask],
        site_key, pollutant,
    )

    # --- Bootstrap uncertainty ---
    print("\n[STEP 7] Bootstrap uncertainty (100 iterations)...")
    bootstrap = bootstrap_uncertainty(X_train, y_train, X_predict, n_bootstraps=100)

    # --- Calculate residuals with uncertainty ---
    print("\n[STEP 8] Calculating residuals with uncertainty...")
    residual_results = calculate_residuals_with_uncertainty(
        y_actual, predictions, bootstrap
    )

    predict_df["residual"] = residual_results["residual"]
    predict_df["residual_lower"] = residual_results["residual_lower"]
    predict_df["residual_upper"] = residual_results["residual_upper"]
    predict_df["uncertainty"] = residual_results["combined_uncertainty"]

    plot_residual_distribution(
        residual_results["residual"][valid_mask], site_key, pollutant
    )
    plot_uncertainty_map(predict_df, residual_results["combined_uncertainty"],
                         site_key, pollutant)

    # --- Sensitivity analysis ---
    print("\n[STEP 9] Buffer distance sensitivity analysis...")
    sensitivity = buffer_sensitivity_analysis(df, pollutant)
    plot_sensitivity(sensitivity, site_key, pollutant)

    # --- Wind analysis (Doura only — we have met data) ---
    wind_result = None
    if site_key == "doura":
        print("\n[STEP 10] Wind-adjusted analysis...")
        wind_df = load_wind_data()
        wind_result = wind_adjusted_analysis(
            predict_df, residual_results["residual"], wind_df, pollutant
        )
        if wind_result:
            plot_wind_rose(wind_result["sector_stats"], site_key, pollutant)

    # --- Export to geodatabase ---
    print("\n[STEP 11] Exporting results...")
    export_cols = ["lon", "lat", "year", "month", pollutant,
                   "residual", "residual_lower", "residual_upper", "uncertainty"]
    export_cols = [c for c in export_cols if c in predict_df.columns]
    export_to_geodatabase(
        predict_df[export_cols],
        f"{site_key}_iesut_{pollutant}",
    )

    # --- Compile results ---
    residual_valid = residual_results["residual"][valid_mask]
    uncertainty_valid = residual_results["combined_uncertainty"][valid_mask]

    return {
        "rf": rf,
        "cv": cv_results,
        "oob_r2": rf.oob_score_,
        "importance": importance,
        "predictions": predictions,
        "residual_results": residual_results,
        "sensitivity": sensitivity,
        "wind_result": wind_result,
        "predict_df": predict_df,
        "df": df,
        "residual_stats": {
            "mean": np.mean(residual_valid),
            "std": np.std(residual_valid),
            "median": np.median(residual_valid),
            "mean_uncertainty": np.mean(uncertainty_valid),
        },
    }


def run_full_analysis():
    """Run complete IESUT analysis for both sites, both pollutants."""
    ensure_output_dir()
    all_results = {}

    for site_key in ["doura", "pernis"]:
        all_results[site_key] = {}
        for pollutant in Config.POLLUTANTS:
            print(f"\n{'#' * 60}")
            print(f"# RUNNING: {site_key.upper()} — {pollutant.upper()}")
            print(f"{'#' * 60}")

            result = run_single_site(site_key, pollutant)
            if result:
                all_results[site_key][pollutant] = result

    # --- Regulatory comparison (NO2) ---
    if ("doura" in all_results and "no2" in all_results["doura"] and
            "pernis" in all_results and "no2" in all_results["pernis"]):

        print(f"\n{'#' * 60}")
        print("# REGULATORY COMPARISON & SCENARIO MODELLING")
        print(f"{'#' * 60}")

        comparison = regulatory_comparison(
            all_results["doura"]["no2"]["residual_results"],
            all_results["pernis"]["no2"]["residual_results"],
            "Doura", "Pernis",
        )
        all_results["doura"]["comparison"] = comparison

        scenario = project_regulatory_scenario(
            all_results["doura"]["no2"]["df"],
            all_results["pernis"]["no2"]["df"],
            "no2",
        )
        all_results["doura"]["scenario"] = scenario

        plot_regulatory_projection(
            scenario, "Doura", "no2",
            all_results["doura"]["no2"]["df"],
            all_results["pernis"]["no2"]["df"],
        )

    # --- Summary report ---
    generate_summary_report(all_results)

    print(f"\n{'=' * 60}")
    print("IESUT ANALYSIS COMPLETE")
    print(f"All outputs saved to: {Config.OUTPUT_DIR}")
    print(f"{'=' * 60}")

    return all_results


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_full_analysis()
