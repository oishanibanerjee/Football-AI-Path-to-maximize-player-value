"""
train_baselines.py — Baseline Regression & Classification for €100M Pathway
Temporal/player-aware split, multiple baselines, metric reporting, feature importance.
"""
import sys
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
)

warnings.filterwarnings("ignore")

# Try imports
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
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REGRESSOR_PATH = MODEL_DIR / "baseline_regressor.pkl"
CLASSIFIER_PATH = MODEL_DIR / "baseline_classifier.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature exclusion — ids, raw targets, leaking future info, non-numeric artifacts
EXCLUDE_COLS = {
    "player_id", "player_name", "club_name", "country_id",
    "birth_date", "foot", "status",
    "season_label", "role", "role_name",
    "player_career_id", "player_season_stats_id", "season_id", "super_season_id",
    "club_id", "team_id", "competition_id",
    # Raw financial targets — keep log_current as feature but exclude future values themselves
    "value_current", "value_t+1", "value_t+2", "value_t+3",
    "target_future_val_2yr", "target_future_val_3yr",
    "target_is_100m_2yr", "target_is_100m_3yr",
    "target_100m_t+1", "target_100m_t+2", "target_100m_t+3",
    "log_target_future_val_2yr", "log_target_future_val_3yr",
    "log_value_t+1", "log_value_t+2", "log_value_t+3",
    "market_value_at_checkpoint",
    "predicted_value_1y", "predicted_value_2y", "predicted_value_3y",
    "fair_price_1y", "fair_price_2y", "fair_price_3y",
    "accuracy_1y", "accuracy_2y", "accuracy_3y",
    "position_group", "age_group", "position_group_short",
    # Keep numeric dummies pos_*, age_*  — they are useful
}

def get_feature_columns(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feats = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    # add dummy cols that are numeric (they already are)
    # Keep only cols with some variance and not all NaN
    feats = [f for f in feats if f in df.columns]
    # variance filter
    variances = df[feats].var(skipna=True)
    feats = variances[variances > 0].index.tolist()
    # Drop columns that are 100% NaN
    feats = [f for f in feats if df[f].notna().any()]
    return feats

def player_aware_split(df: pd.DataFrame, target_col: str, groups: pd.Series, test_size=0.2):
    """GroupShuffleSplit by player_id to avoid leakage."""
    # Drop rows where target is null for this stage
    mask = df[target_col].notna()
    df_valid = df.loc[mask].copy()
    groups_valid = groups.loc[mask]
    X = df_valid # placeholder, actual X built outside
    y = df_valid[target_col]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_valid, y, groups_valid))
    return df_valid.iloc[train_idx], df_valid.iloc[test_idx], train_idx, test_idx

def print_feature_importance(model, feature_names, top_n=15, title=""):
    print(f"\n  --- {title} Top {top_n} Features ---")
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef[0]
        importances = np.abs(coef)
    elif hasattr(model, "named_steps"):
        # Pipeline: get final estimator
        est = model.named_steps.get("model") or model.named_steps.get("regressor") or model.named_steps.get("classifier")
        if est is not None and hasattr(est, "feature_importances_"):
            importances = est.feature_importances_
        elif est is not None and hasattr(est, "coef_"):
            coef = est.coef_
            if coef.ndim > 1:
                coef = coef[0]
            importances = np.abs(coef)
    if importances is None:
        print("    (no importance available)")
        return []
    # Align length
    if len(importances) != len(feature_names):
        print(f"    Warning: importance len {len(importances)} != features {len(feature_names)}")
        # truncate
        feature_names = feature_names[:len(importances)]
    idx = np.argsort(importances)[::-1][:top_n]
    rows = []
    for rank, i in enumerate(idx, 1):
        print(f"    {rank:2d}. {feature_names[i]:35s}  {importances[i]:.5f}")
        rows.append((feature_names[i], float(importances[i])))
    return rows

def train_and_evaluate():
    print("=" * 70)
    print("  BASELINE TRAINING — REGRESSION & CLASSIFICATION")
    print("=" * 70)

    if not MATRIX_PATH.exists():
        print(f"[ERROR] Matrix not found at {MATRIX_PATH}")
        print("  Run src/build_trajectory_matrix.py first")
        sys.exit(1)

    df = pd.read_parquet(MATRIX_PATH)
    print(f"\n[Load] Matrix: {df.shape[0]} rows x {df.shape[1]} cols from {MATRIX_PATH}")

    feature_cols = get_feature_columns(df)
    print(f"  Feature pool: {len(feature_cols)} numeric columns")
    print(f"  Sample features: {feature_cols[:10]}")

    # Ensure key spec features are present
    must_have = ["age", "height_cm", "total_minutes", "minutes_share", "start_ratio",
                 "current_market_value", "log_current_market_value"]
    for c in must_have:
        status = "OK" if c in feature_cols else ("MISSING (will be excluded)" if c not in df.columns else "EXCLUDED/filtered")
        print(f"    {c}: {status}")

    # Groups for player-aware split
    groups = df["player_id"]

    # ---------------- REGRESSION: log(target_future_val_2yr) ----------------
    print("\n" + "=" * 70)
    print("  REGRESSION — Predicting log(target_future_val_2yr)")
    print("=" * 70)

    reg_target = "log_target_future_val_2yr"
    if reg_target not in df.columns:
        # fallback compute
        df[reg_target] = np.log1p(df["target_future_val_2yr"].fillna(0))
        df.loc[df["target_future_val_2yr"].isna(), reg_target] = np.nan

    # Player-aware split for regression
    reg_mask = df[reg_target].notna()
    df_reg = df.loc[reg_mask].copy()
    groups_reg = groups.loc[reg_mask]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df_reg, df_reg[reg_target], groups_reg))
    # Overlap check
    train_players = set(df_reg.iloc[train_idx]["player_id"])
    test_players = set(df_reg.iloc[test_idx]["player_id"])
    overlap = train_players.intersection(test_players)
    print(f"\n  Regression split: train {len(train_idx)} / test {len(test_idx)}  | overlap players: {len(overlap)} (should be 0)")

    X_train_reg_raw = df_reg.iloc[train_idx][feature_cols]
    X_test_reg_raw = df_reg.iloc[test_idx][feature_cols]
    y_train_reg = df_reg.iloc[train_idx][reg_target]
    y_test_reg = df_reg.iloc[test_idx][reg_target]

    # Also need true euro values for MAE in millions
    y_test_reg_euro = df_reg.iloc[test_idx]["target_future_val_2yr"]

    # Preprocess: median imputation + optional scaling for Ridge/LogReg
    imputer = SimpleImputer(strategy="median")
    X_train_reg = pd.DataFrame(imputer.fit_transform(X_train_reg_raw), columns=feature_cols, index=X_train_reg_raw.index)
    X_test_reg = pd.DataFrame(imputer.transform(X_test_reg_raw), columns=feature_cols, index=X_test_reg_raw.index)
    # Replace inf
    X_train_reg = X_train_reg.replace([np.inf, -np.inf], 0)
    X_test_reg = X_test_reg.replace([np.inf, -np.inf], 0)

    # Keep imputer for saving
    reg_results = {}
    trained_regressors = {}

    # Baseline 1: Ridge
    print("\n  [Reg] Training Ridge Regression...")
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))])
    ridge_pipe.fit(X_train_reg, y_train_reg)
    pred_log = ridge_pipe.predict(X_test_reg)
    # Metrics: convert log preds back to euro for MAE/RMSE in millions
    pred_euro = np.expm1(pred_log)
    pred_euro = np.maximum(pred_euro, 0)
    mae_euro = mean_absolute_error(y_test_reg_euro, pred_euro)
    rmse_euro = np.sqrt(mean_squared_error(y_test_reg_euro, pred_euro))
    # R2 on log scale (as model trained)
    r2_log = r2_score(y_test_reg, pred_log)
    # Also R2 on euro scale
    r2_euro = r2_score(y_test_reg_euro, pred_euro)
    mae_m = mae_euro / 1e6
    rmse_m = rmse_euro / 1e6
    print(f"    Ridge — MAE: {mae_euro:,.0f} EUR ({mae_m:.2f}M) | RMSE: {rmse_euro:,.0f} ({rmse_m:.2f}M) | R2(log): {r2_log:.4f} | R2(euro): {r2_euro:.4f}")
    reg_results["Ridge"] = {"mae_euro": mae_euro, "mae_m": mae_m, "rmse_euro": rmse_euro, "rmse_m": rmse_m, "r2_log": r2_log, "r2_euro": r2_euro}
    trained_regressors["Ridge"] = ridge_pipe
    print_feature_importance(ridge_pipe.named_steps["model"], feature_cols, title="Ridge")

    # Baseline 2: Random Forest
    print("\n  [Reg] Training Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train_reg, y_train_reg)
    pred_log = rf.predict(X_test_reg)
    pred_euro = np.maximum(np.expm1(pred_log), 0)
    mae_euro = mean_absolute_error(y_test_reg_euro, pred_euro)
    rmse_euro = np.sqrt(mean_squared_error(y_test_reg_euro, pred_euro))
    r2_log = r2_score(y_test_reg, pred_log)
    mae_m = mae_euro / 1e6; rmse_m = rmse_euro / 1e6
    print(f"    RF    — MAE: {mae_euro:,.0f} EUR ({mae_m:.2f}M) | RMSE: {rmse_euro:,.0f} ({rmse_m:.2f}M) | R2(log): {r2_log:.4f}")
    reg_results["RandomForest"] = {"mae_euro": mae_euro, "mae_m": mae_m, "rmse_euro": rmse_euro, "rmse_m": rmse_m, "r2_log": r2_log}
    trained_regressors["RandomForest"] = rf
    print_feature_importance(rf, feature_cols, title="RandomForest")

    # Baseline 3: XGBoost / LightGBM
    xgb_reg = None
    lgb_reg = None
    if HAS_XGB:
        print("\n  [Reg] Training XGBoost Regressor...")
        xgb_reg = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        )
        xgb_reg.fit(X_train_reg, y_train_reg, verbose=False)
        pred_log = xgb_reg.predict(X_test_reg)
        pred_euro = np.maximum(np.expm1(pred_log), 0)
        mae_euro = mean_absolute_error(y_test_reg_euro, pred_euro)
        rmse_euro = np.sqrt(mean_squared_error(y_test_reg_euro, pred_euro))
        r2_log = r2_score(y_test_reg, pred_log)
        mae_m = mae_euro / 1e6; rmse_m = rmse_euro / 1e6
        print(f"    XGB   — MAE: {mae_euro:,.0f} EUR ({mae_m:.2f}M) | RMSE: {rmse_euro:,.0f} ({rmse_m:.2f}M) | R2(log): {r2_log:.4f}")
        reg_results["XGBoost"] = {"mae_euro": mae_euro, "mae_m": mae_m, "rmse_euro": rmse_euro, "rmse_m": rmse_m, "r2_log": r2_log}
        trained_regressors["XGBoost"] = xgb_reg
        print_feature_importance(xgb_reg, feature_cols, title="XGBoost Regressor")
    elif HAS_LGB:
        print("\n  [Reg] Training LightGBM Regressor (XGB not available)...")
        lgb_reg = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=-1, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=-1
        )
        lgb_reg.fit(X_train_reg, y_train_reg)
        pred_log = lgb_reg.predict(X_test_reg)
        pred_euro = np.maximum(np.expm1(pred_log), 0)
        mae_euro = mean_absolute_error(y_test_reg_euro, pred_euro)
        rmse_euro = np.sqrt(mean_squared_error(y_test_reg_euro, pred_euro))
        r2_log = r2_score(y_test_reg, pred_log)
        mae_m = mae_euro / 1e6; rmse_m = rmse_euro / 1e6
        print(f"    LGBM  — MAE: {mae_euro:,.0f} EUR ({mae_m:.2f}M) | RMSE: {rmse_euro:,.0f} ({rmse_m:.2f}M) | R2(log): {r2_log:.4f}")
        reg_results["LightGBM"] = {"mae_euro": mae_euro, "mae_m": mae_m, "rmse_euro": rmse_euro, "rmse_m": rmse_m, "r2_log": r2_log}
        trained_regressors["LightGBM"] = lgb_reg
        print_feature_importance(lgb_reg, feature_cols, title="LightGBM Regressor")
    else:
        print("\n  [Reg] Skipping XGB/LGBM — neither installed")

    # Pick best regressor by R2
    best_reg_name = max(reg_results, key=lambda k: reg_results[k]["r2_log"])
    best_reg = trained_regressors[best_reg_name]
    print(f"\n  [Reg] Best model: {best_reg_name} (R2_log={reg_results[best_reg_name]['r2_log']:.4f})")

    # ---------------- CLASSIFICATION: target_is_100m_2yr ----------------
    print("\n" + "=" * 70)
    print("  CLASSIFICATION — Predicting target_is_100m_2yr (P >= €100M at t+2)")
    print("=" * 70)

    cls_target = "target_is_100m_2yr"
    # Need to handle nullable Int64
    cls_mask = df[cls_target].notna()
    df_cls = df.loc[cls_mask].copy()
    # Convert to int
    df_cls[cls_target] = df_cls[cls_target].astype(int)
    groups_cls = df_cls["player_id"]
    # Check class balance
    n_pos = (df_cls[cls_target] == 1).sum()
    n_neg = (df_cls[cls_target] == 0).sum()
    print(f"\n  Full classification pool: {len(df_cls)} rows | Positives: {n_pos} ({n_pos/len(df_cls)*100:.2f}%) | Negatives: {n_neg}")

    # Player-aware split for classification — handle extreme imbalance (only 4 positives at true 100M)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx_c, test_idx_c = next(gss2.split(df_cls, df_cls[cls_target], groups_cls))
    for alt_seed in [1, 7, 123, 99, 2024, 303, 404]:
        test_pos = (df_cls.iloc[test_idx_c][cls_target]==1).sum()
        train_pos = (df_cls.iloc[train_idx_c][cls_target]==1).sum()
        if test_pos >= 1 and train_pos >= 1:
            break
        gss_alt = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=alt_seed)
        train_idx_c, test_idx_c = next(gss_alt.split(df_cls, df_cls[cls_target], groups_cls))
    test_pos = (df_cls.iloc[test_idx_c][cls_target]==1).sum()
    if test_pos == 0:
        print(f"  [Warn] Test split has 0 positives (only {n_pos} total at 100M) — constructing manual player-disjoint balanced split")
        pos_idx_list = df_cls.index[df_cls[cls_target]==1].tolist()
        pos_players = df_cls.loc[pos_idx_list, "player_id"].unique().tolist()
        test_player = pos_players[0]
        test_pos_idx = df_cls.index[df_cls["player_id"]==test_player].tolist()
        pos_to_iloc = {idx:i for i, idx in enumerate(df_cls.index)}
        test_idx_c = [pos_to_iloc[i] for i in test_pos_idx]
        remaining = df_cls[~df_cls["player_id"].isin([test_player])]
        n_needed = int(len(df_cls)*TEST_SIZE) - len(test_idx_c)
        neg_pool = remaining[remaining[cls_target]==0]
        neg_sample = neg_pool.sample(n=min(n_needed, len(neg_pool)), random_state=RANDOM_STATE)
        test_idx_c += [pos_to_iloc[i] for i in neg_sample.index]
        all_idx = set(range(len(df_cls)))
        train_idx_c = sorted(list(all_idx - set(test_idx_c)))

    train_players_c = set(df_cls.iloc[train_idx_c]["player_id"])
    test_players_c = set(df_cls.iloc[test_idx_c]["player_id"])
    overlap_c = train_players_c.intersection(test_players_c)
    print(f"  Classification split: train {len(train_idx_c)} / test {len(test_idx_c)} | overlap players: {len(overlap_c)} (should be 0)")
    print(f"    Train positives: {(df_cls.iloc[train_idx_c][cls_target]==1).sum()} | Test positives: {(df_cls.iloc[test_idx_c][cls_target]==1).sum()}")

    X_train_cls_raw = df_cls.iloc[train_idx_c][feature_cols]
    X_test_cls_raw = df_cls.iloc[test_idx_c][feature_cols]
    y_train_cls = df_cls.iloc[train_idx_c][cls_target].astype(int)
    y_test_cls = df_cls.iloc[test_idx_c][cls_target].astype(int)

    # Impute
    imputer_c = SimpleImputer(strategy="median")
    X_train_cls = pd.DataFrame(imputer_c.fit_transform(X_train_cls_raw), columns=feature_cols, index=X_train_cls_raw.index)
    X_test_cls = pd.DataFrame(imputer_c.transform(X_test_cls_raw), columns=feature_cols, index=X_test_cls_raw.index)
    X_train_cls = X_train_cls.replace([np.inf, -np.inf], 0)
    X_test_cls = X_test_cls.replace([np.inf, -np.inf], 0)

    # Class weight for imbalance
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight (for XGB): {scale_pos_weight:.1f}")

    cls_results = {}
    trained_classifiers = {}

    # Logistic Regression
    print("\n  [Cls] Training Logistic Regression...")
    logreg_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))])
    logreg_pipe.fit(X_train_cls, y_train_cls)
    proba = logreg_pipe.predict_proba(X_test_cls)[:, 1]
    pred = (proba >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test_cls, proba)
    try:
        roc_auc = roc_auc_score(y_test_cls, proba)
    except:
        roc_auc = float("nan")
    f1 = f1_score(y_test_cls, pred, zero_division=0)
    prec = precision_score(y_test_cls, pred, zero_division=0)
    rec = recall_score(y_test_cls, pred, zero_division=0)
    print(f"    LogReg — PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    cls_results["LogisticRegression"] = {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1, "precision": prec, "recall": rec}
    trained_classifiers["LogisticRegression"] = logreg_pipe
    print_feature_importance(logreg_pipe.named_steps["model"], feature_cols, title="LogisticRegression")

    # XGBoost Classifier
    if HAS_XGB:
        print("\n  [Cls] Training XGBoost Classifier (scale_pos_weight)...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            eval_metric="logloss", use_label_encoder=False
        )
        xgb_clf.fit(X_train_cls, y_train_cls, verbose=False)
        proba = xgb_clf.predict_proba(X_test_cls)[:, 1]
        pred = (proba >= 0.5).astype(int)
        # Also try optimal threshold via F1
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            f1_t = f1_score(y_test_cls, (proba >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1, best_t = f1_t, t
        pr_auc = average_precision_score(y_test_cls, proba)
        try:
            roc_auc = roc_auc_score(y_test_cls, proba)
        except:
            roc_auc = float("nan")
        f1 = f1_score(y_test_cls, (proba >= 0.5).astype(int), zero_division=0)
        prec = precision_score(y_test_cls, (proba >= 0.5).astype(int), zero_division=0)
        rec = recall_score(y_test_cls, (proba >= 0.5).astype(int), zero_division=0)
        print(f"    XGB    — PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1@0.5: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Best F1 {best_f1:.4f} @ t={best_t:.2f}")
        cls_results["XGBoost"] = {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1, "precision": prec, "recall": rec, "best_f1": best_f1, "best_thresh": best_t}
        trained_classifiers["XGBoost"] = xgb_clf
        print_feature_importance(xgb_clf, feature_cols, title="XGBoost Classifier")
    else:
        print("\n  [Cls] Skipping XGB — not installed")

    # Pick best classifier by PR-AUC
    best_cls_name = max(cls_results, key=lambda k: cls_results[k]["pr_auc"])
    best_cls = trained_classifiers[best_cls_name]
    print(f"\n  [Cls] Best model: {best_cls_name} (PR-AUC={cls_results[best_cls_name]['pr_auc']:.4f})")

    # ---------------- SAVE ARTIFACTS ----------------
    print("\n" + "=" * 70)
    print("  SAVING ARTIFACTS")
    print("=" * 70)

    # Save best regressor with metadata
    reg_payload = {
        "model": best_reg,
        "model_name": best_reg_name,
        "feature_columns": feature_cols,
        "imputer": imputer,
        "target": reg_target,
        "all_results": reg_results,
        "test_metrics": reg_results[best_reg_name],
    }
    # For pipeline regressors, the imputer is separate; for RF/XGB, imputer already applied.
    # To make inference consistent, we save imputer + feature list + model
    with open(REGRESSOR_PATH, "wb") as f:
        pickle.dump(reg_payload, f)
    print(f"  Saved regressor ({best_reg_name}) -> {REGRESSOR_PATH} ({REGRESSOR_PATH.stat().st_size/1024:.1f} KB)")

    cls_payload = {
        "model": best_cls,
        "model_name": best_cls_name,
        "feature_columns": feature_cols,
        "imputer": imputer_c,
        "target": cls_target,
        "all_results": cls_results,
        "test_metrics": cls_results[best_cls_name],
        "threshold": 0.5,
    }
    with open(CLASSIFIER_PATH, "wb") as f:
        pickle.dump(cls_payload, f)
    print(f"  Saved classifier ({best_cls_name}) -> {CLASSIFIER_PATH} ({CLASSIFIER_PATH.stat().st_size/1024:.1f} KB)")

    # ---------------- FINAL SUMMARY ----------------
    print("\n" + "=" * 70)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 70)

    print("\n  Regression (log(target_future_val_2yr) -> EUR):")
    print(f"  {'Model':<18} {'MAE (M)':>9} {'RMSE (M)':>9} {'R2(log)':>8}")
    print("  " + "-" * 50)
    for name, m in reg_results.items():
        print(f"  {name:<18} {m['mae_m']:9.2f} {m['rmse_m']:9.2f} {m['r2_log']:8.4f}")

    print("\n  Classification (target_is_100m_2yr):")
    print(f"  {'Model':<22} {'PR-AUC':>7} {'ROC-AUC':>8} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print("  " + "-" * 60)
    for name, m in cls_results.items():
        print(f"  {name:<22} {m['pr_auc']:7.4f} {m['roc_auc']:8.4f} {m['f1']:6.4f} {m['precision']:6.4f} {m['recall']:6.4f}")

    print("\n  Artifacts:")
    print(f"    Matrix:     {MATRIX_PATH} ({MATRIX_PATH.stat().st_size/1024/1024:.2f} MB)")
    print(f"    Regressor:  {REGRESSOR_PATH}")
    print(f"    Classifier: {CLASSIFIER_PATH}")
    print("\n  Done.")

if __name__ == "__main__":
    train_and_evaluate()
