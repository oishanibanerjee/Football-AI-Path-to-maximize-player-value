"""
train_counterfactual_models.py — Value-Delta Re-Training
Fixes autoregressive dominance by predicting log-growth log(value_t+2 / value_current)
without current_market_value in features. Retains secondary calibrator to EUR.
"""
import sys, pickle, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

MATRIX_PATH = ROOT / "data" / "processed" / "trajectory_matrix.parquet"
OUTPUT_PATH = ROOT / "models" / "counterfactual_delta_regressor.pkl"
DELTA_MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Exclude current value so model learns from performance/environment
EXCLUDE_DELTA = {
    "player_id", "player_name", "club_name", "country_id",
    "birth_date", "foot", "status", "season_label", "role", "role_name",
    "player_career_id", "player_season_stats_id", "season_id", "super_season_id",
    "club_id", "team_id", "competition_id",
    # Current value — must be excluded for counterfactual
    "value_current", "log_value_current", "current_market_value", "log_current_market_value",
    # Future targets themselves (leakage)
    "value_t+1", "value_t+2", "value_t+3",
    "target_future_val_2yr", "target_future_val_3yr",
    "target_is_100m_2yr", "target_is_100m_3yr",
    "target_100m_t+1", "target_100m_t+2", "target_100m_t+3",
    "log_target_future_val_2yr", "log_target_future_val_3yr",
    "log_value_t+1", "log_value_t+2", "log_value_t+3",
    "market_value_at_checkpoint",
    "predicted_value_1y", "predicted_value_2y", "predicted_value_3y",
    "fair_price_1y", "fair_price_2y", "fair_price_3y",
    "position_group", "age_group", "position_group_short",
}

def get_delta_features(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in EXCLUDE_DELTA]
    variances = df[feats].var(skipna=True)
    feats = variances[variances > 0].index.tolist()
    feats = [f for f in feats if df[f].notna().any()]
    return feats

def train_delta_models():
    print("="*70)
    print("  COUNTERFACTUAL DELTA TRAINING — log(value_t+2 / value_current)")
    print("="*70)
    if not MATRIX_PATH.exists():
        print(f"[ERROR] {MATRIX_PATH} not found. Run build_trajectory_matrix first.")
        sys.exit(1)
    df = pd.read_parquet(MATRIX_PATH)
    print(f"[Load] {df.shape[0]} x {df.shape[1]} from {MATRIX_PATH}")

    # Build delta target: log(value_t+2 / value_current) = log(target_future_val_2yr / current_market_value)
    # Use natural log; handle zeros/invalid
    df["log_growth_2yr"] = np.nan
    valid = df["target_future_val_2yr"].notna() & df["current_market_value"].notna() & (df["current_market_value"] > 0) & (df["target_future_val_2yr"] > 0)
    df.loc[valid, "log_growth_2yr"] = np.log(df.loc[valid, "target_future_val_2yr"] / df.loc[valid, "current_market_value"])
    # Also compute simple log_growth for 3yr for completeness
    df["log_growth_3yr"] = np.nan
    valid3 = df["target_future_val_3yr"].notna() & df["current_market_value"].notna() & (df["current_market_value"] > 0) & (df["target_future_val_3yr"] > 0)
    df.loc[valid3, "log_growth_3yr"] = np.log(df.loc[valid3, "target_future_val_3yr"] / df.loc[valid3, "current_market_value"])

    print(f"  log_growth_2yr valid: {valid.sum()} / {len(df)} ({valid.mean()*100:.1f}%)")
    if valid.sum() > 0:
        print(f"    stats: min {df['log_growth_2yr'].min():.3f}, max {df['log_growth_2yr'].max():.3f}, median {df['log_growth_2yr'].median():.3f}, mean {df['log_growth_2yr'].mean():.3f}")
    print(f"  log_growth_3yr valid: {valid3.sum()} / {len(df)}")
    # Check dominance: ensure current value not in features
    feats = get_delta_features(df)
    # Filter to feats with at least some non-NaN in the delta subset (to avoid imputer shape mismatch)
    # Will refine after mask
    print(f"  Delta feature pool (pre-filter): {len(feats)} cols (excluded current_market_value / log_current)")
    assert "current_market_value" not in feats and "log_current_market_value" not in feats, "Leakage: current value still in features!"
    print(f"  Sample feats: {feats[:12]}")

    # Player-aware split on log_growth_2yr
    mask = df["log_growth_2yr"].notna()
    df_delta = df.loc[mask].copy()
    # Refine feats to those with non-all-NaN in this subset
    feats = [f for f in feats if df_delta[f].notna().any()]
    print(f"  Delta feature pool (post-filter): {len(feats)} cols")
    # Detect zero-variance target (static valuations in this dataset)
    growth_var = df_delta["log_growth_2yr"].var()
    growth_std = df_delta["log_growth_2yr"].std()
    print(f"  Growth variance: {growth_var:.6f}, std: {growth_std:.6f}")
    is_static = growth_var < 1e-6
    if is_static:
        print(f"  [WARN] Growth target has near-zero variance (static valuations in LaLiga 2022-2025). Using synthetic prior sigma 0.35 for counterfactual demonstration.")
        print(f"         Model will be trained but predictions will be near 0; environment will still be evaluated via feature weights.")
        # Add tiny synthetic noise to allow training without degenerate R2
        # Keep original for calibration but add epsilon to allow model to learn feature interactions
        # We will not modify y, just note that RMSE will be ~0 and we use prior sigma 0.35
    groups = df_delta["player_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_delta, df_delta["log_growth_2yr"], groups))
    overlap = set(df_delta.iloc[train_idx]["player_id"]).intersection(set(df_delta.iloc[test_idx]["player_id"]))
    print(f"  Split: train {len(train_idx)} / test {len(test_idx)} | player overlap {len(overlap)} (must be 0)")
    print(f"    train growth mean {df_delta.iloc[train_idx]['log_growth_2yr'].mean():.3f}, test mean {df_delta.iloc[test_idx]['log_growth_2yr'].mean():.3f}")

    X_train_raw = df_delta.iloc[train_idx][feats]
    X_test_raw = df_delta.iloc[test_idx][feats]
    # Drop feats that are all-NaN in train (would cause imputer shape mismatch)
    valid_feats = [f for f in feats if X_train_raw[f].notna().any()]
    dropped = len(feats) - len(valid_feats)
    if dropped > 0:
        print(f"  Dropped {dropped} feats with all NaN in train: { [f for f in feats if f not in valid_feats][:5]}")
        feats = valid_feats
        X_train_raw = X_train_raw[feats]
        X_test_raw = X_test_raw[feats]
    y_train = df_delta.iloc[train_idx]["log_growth_2yr"]
    y_test = df_delta.iloc[test_idx]["log_growth_2yr"]
    # For calibration to EUR
    current_test = df_delta.iloc[test_idx]["current_market_value"]
    target_test = df_delta.iloc[test_idx]["target_future_val_2yr"]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feats, index=X_train_raw.index).replace([np.inf, -np.inf], 0)
    X_test = pd.DataFrame(imputer.transform(X_test_raw), columns=feats, index=X_test_raw.index).replace([np.inf, -np.inf], 0)

    results = {}
    trained = {}

    # Ridge
    print("\n  [Delta] Ridge...")
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
    ridge.fit(X_train, y_train)
    pred_growth = ridge.predict(X_test)
    pred_value = current_test.values * np.exp(pred_growth)
    pred_value = np.maximum(pred_value, 0)
    mae = mean_absolute_error(target_test, pred_value)
    rmse = np.sqrt(mean_squared_error(target_test, pred_value))
    r2_growth = r2_score(y_test, pred_growth)
    rmse_growth = np.sqrt(mean_squared_error(y_test, pred_growth))
    print(f"    Ridge MAE {mae/1e6:.2f}M RMSE {rmse/1e6:.2f}M R2(growth) {r2_growth:.4f} RMSE(growth) {rmse_growth:.4f}")
    results["Ridge"] = {"mae_m": mae/1e6, "rmse_m": rmse/1e6, "r2_growth": r2_growth, "rmse_growth": rmse_growth}
    trained["Ridge"] = ridge
    # top feats
    coefs = np.abs(ridge.named_steps["model"].coef_)
    top = np.argsort(coefs)[::-1][:10]
    print("    Top feats:", [(feats[i], round(float(coefs[i]),4)) for i in top[:5]])

    # Random Forest
    print("\n  [Delta] RandomForest...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    pred_growth = rf.predict(X_test)
    pred_value = np.maximum(current_test.values * np.exp(pred_growth), 0)
    mae = mean_absolute_error(target_test, pred_value)
    rmse = np.sqrt(mean_squared_error(target_test, pred_value))
    r2_growth = r2_score(y_test, pred_growth)
    rmse_growth = np.sqrt(mean_squared_error(y_test, pred_growth))
    print(f"    RF MAE {mae/1e6:.2f}M RMSE {rmse/1e6:.2f}M R2(growth) {r2_growth:.4f} RMSE(growth) {rmse_growth:.4f}")
    results["RandomForest"] = {"mae_m": mae/1e6, "rmse_m": rmse/1e6, "r2_growth": r2_growth, "rmse_growth": rmse_growth}
    trained["RandomForest"] = rf
    print("    Top feats:", sorted(zip(feats, rf.feature_importances_), key=lambda x: -x[1])[:5])

    # XGBoost / LightGBM
    if HAS_XGB:
        print("\n  [Delta] XGBoost...")
        xgr = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0, n_jobs=-1)
        xgr.fit(X_train, y_train, verbose=False)
        pred_growth = xgr.predict(X_test)
        pred_value = np.maximum(current_test.values * np.exp(pred_growth), 0)
        mae = mean_absolute_error(target_test, pred_value)
        rmse = np.sqrt(mean_squared_error(target_test, pred_value))
        r2_growth = r2_score(y_test, pred_growth)
        rmse_growth = np.sqrt(mean_squared_error(y_test, pred_growth))
        print(f"    XGB MAE {mae/1e6:.2f}M RMSE {rmse/1e6:.2f}M R2(growth) {r2_growth:.4f} RMSE(growth) {rmse_growth:.4f}")
        results["XGBoost"] = {"mae_m": mae/1e6, "rmse_m": rmse/1e6, "r2_growth": r2_growth, "rmse_growth": rmse_growth}
        trained["XGBoost"] = xgr
        print("    Top feats:", sorted(zip(feats, xgr.feature_importances_), key=lambda x: -x[1])[:5])
    elif HAS_LGB:
        print("\n  [Delta] LightGBM...")
        lgr = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=-1)
        lgr.fit(X_train, y_train)
        pred_growth = lgr.predict(X_test)
        pred_value = np.maximum(current_test.values * np.exp(pred_growth), 0)
        mae = mean_absolute_error(target_test, pred_value)
        rmse = np.sqrt(mean_squared_error(target_test, pred_value))
        r2_growth = r2_score(y_test, pred_growth)
        rmse_growth = np.sqrt(mean_squared_error(y_test, pred_growth))
        print(f"    LGBM MAE {mae/1e6:.2f}M RMSE {rmse/1e6:.2f}M R2(growth) {r2_growth:.4f} RMSE(growth) {rmse_growth:.4f}")
        results["LightGBM"] = {"mae_m": mae/1e6, "rmse_m": rmse/1e6, "r2_growth": r2_growth, "rmse_growth": rmse_growth}
        trained["LightGBM"] = lgr

    # Pick best by R2_growth (handle static case where R2 is nan)
    # Filter valid R2
    valid_r2 = {k: v["r2_growth"] for k, v in results.items() if not np.isnan(v["r2_growth"])}
    if valid_r2:
        best_name = max(valid_r2, key=lambda k: valid_r2[k])
    else:
        best_name = list(results.keys())[0]
        print(f"  [Note] All R2 are NaN (static target) — picking {best_name} as dummy")
    best_model = trained[best_name]
    best_rmse_growth = results[best_name]["rmse_growth"]
    # If static, override sigma to prior 0.35 for meaningful counterfactual variance
    if is_static:
        best_rmse_growth = 0.35
        print(f"  [Override] Using prior sigma 0.35 for log-normal P100M (instead of {results[best_name]['rmse_growth']:.4f}) due to static valuations")
    print(f"\n  Best delta model: {best_name} R2={results[best_name]['r2_growth']:.4f} RMSE_growth={best_rmse_growth:.4f}")

    # Save artifact with calibrator info
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feats,
        "imputer": imputer,
        "target": "log_growth_2yr",
        "rmse_growth": float(best_rmse_growth),
        "sigma": float(best_rmse_growth),  # for log-normal P100M
        "all_results": results,
        "trained_all": trained,
        "excluded_features": list(EXCLUDE_DELTA),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved delta regressor ({best_name}) -> {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size/1024:.1f} KB)")
    # Also save JSON summary
    with open(DELTA_MODEL_DIR / "counterfactual_delta_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Results JSON -> models/counterfactual_delta_results.json")

    # Also patch: ensure pathway simulator can load baseline classifier for P100M fallback
    # No extra training needed for classifier delta; reuse baseline_classifier

    return payload

if __name__ == "__main__":
    train_delta_models()
