"""
models.py — Machine Learning Model Development (Staged Pipeline)
LightGBM, XGBoost, CatBoost with player-stratified cross-validation.
Stage 1: Continuous Value Regressor (log market value).
Stage 2: High-Valuation Classifier (€100M probability).
Stage 3: Expected Minutes Predictor.
"""

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    classification_report,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import catboost as cb
except ImportError:
    cb = None

try:
    import shap
except ImportError:
    shap = None

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models_saved"
MODELS_DIR.mkdir(exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _player_groups(X: pd.DataFrame, meta: pd.DataFrame) -> np.ndarray:
    """Extract player_id groups aligned with X's index."""
    return meta.loc[X.index, "player_id"].values


def _save_model(model: Any, name: str, fold: int):
    path = MODELS_DIR / f"{name}_fold{fold}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)


def _log_results(results: dict, stage: str):
    path = MODELS_DIR / f"results_{stage}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[Models] Results saved to {path}")


# ---------------------------------------------------------------------------
# Stage 1 — Value Regressor
# ---------------------------------------------------------------------------

def train_value_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    target_name: str = "log_value",
    feature_names: list[str] | None = None,
) -> dict:
    """
    Stage 1: Train LightGBM / XGBoost / CatBoost regressors for log(value_{t+k}).
    Returns dict of trained models and evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Value Regressor ({target_name})")
    print(f"{'='*60}")

    groups = _player_groups(X, meta)
    gkf = GroupKFold(n_splits=N_FOLDS)

    results = {"stage": "value_regressor", "target": target_name, "folds": []}
    trained_models = {name: [] for name in ["lgb", "xgb", "cb"]}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")

        fold_metrics = {"fold": fold_idx + 1}

        # --- LightGBM ---
        if lgb is not None:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            lgb_params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
                "seed": RANDOM_STATE,
            }
            callbacks = [lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)]
            lgb_model = lgb.train(
                lgb_params, lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )
            pred_lgb = lgb_model.predict(X_val)
            _save_model(lgb_model, f"value_lgb_{target_name}", fold_idx)
            trained_models["lgb"].append(lgb_model)

            mae = mean_absolute_error(y_val, pred_lgb)
            rmse = np.sqrt(mean_squared_error(y_val, pred_lgb))
            r2 = r2_score(y_val, pred_lgb)
            fold_metrics["lgb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    LGB  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        # --- XGBoost ---
        if xgb is not None:
            xgb_model = xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_child_weight=5,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50,
                eval_metric="mae",
                verbosity=0,
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            pred_xgb = xgb_model.predict(X_val)
            _save_model(xgb_model, f"value_xgb_{target_name}", fold_idx)
            trained_models["xgb"].append(xgb_model)

            mae = mean_absolute_error(y_val, pred_xgb)
            rmse = np.sqrt(mean_squared_error(y_val, pred_xgb))
            r2 = r2_score(y_val, pred_xgb)
            fold_metrics["xgb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    XGB  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        # --- CatBoost ---
        if cb is not None:
            cb_model = cb.CatBoostRegressor(
                iterations=1000,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_seed=RANDOM_STATE,
                verbose=0,
                early_stopping_rounds=50,
            )
            cb_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=0,
            )
            pred_cb = cb_model.predict(X_val)
            _save_model(cb_model, f"value_cb_{target_name}", fold_idx)
            trained_models["cb"].append(cb_model)

            mae = mean_absolute_error(y_val, pred_cb)
            rmse = np.sqrt(mean_squared_error(y_val, pred_cb))
            r2 = r2_score(y_val, pred_cb)
            fold_metrics["cb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    CB   MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

        results["folds"].append(fold_metrics)

    # Aggregate
    for model_name in ["lgb", "xgb", "cb"]:
        if results["folds"][0].get(model_name):
            avg_mae = np.mean([f[model_name]["mae"] for f in results["folds"]])
            avg_rmse = np.mean([f[model_name]["rmse"] for f in results["folds"]])
            avg_r2 = np.mean([f[model_name]["r2"] for f in results["folds"]])
            print(f"\n  {model_name.upper()} AVG -> MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}  R2={avg_r2:.4f}")

    _log_results(results, f"value_regressor_{target_name}")
    return {"models": trained_models, "results": results}


# ---------------------------------------------------------------------------
# Stage 1b — Delta Value Regressor (counterfactual, no current value)
# ---------------------------------------------------------------------------

def train_delta_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    target_name: str = "log_growth_2yr",
    feature_names: list[str] | None = None,
) -> dict:
    """
    Counterfactual delta regressor: predicts log(value_t+2 / value_current)
    WITHOUT current_market_value in features. Same GroupKFold logic as
    train_value_regressor but exposes rmse_growth for log-normal P100M.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 1b — Delta Regressor ({target_name}) [no current value]")
    print(f"{'='*60}")
    groups = _player_groups(X, meta)
    gkf = GroupKFold(n_splits=N_FOLDS)
    results = {"stage": "delta_regressor", "target": target_name, "folds": []}
    trained_models = {name: [] for name in ["lgb", "xgb", "cb"]}
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        print(f"\n  Fold {fold_idx+1}/{N_FOLDS}: train={len(train_idx)}, val={len(val_idx)}")
        fold_metrics = {"fold": fold_idx+1}
        if lgb is not None:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            lgb_params = {"objective": "regression", "metric": "mae", "boosting_type": "gbdt", "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5, "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 0.1, "verbose": -1, "seed": RANDOM_STATE}
            callbacks = [lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)]
            lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, valid_sets=[lgb_val], callbacks=callbacks)
            pred = lgb_model.predict(X_val)
            _save_model(lgb_model, f"delta_lgb_{target_name}", fold_idx)
            trained_models["lgb"].append(lgb_model)
            mae = mean_absolute_error(y_val, pred); rmse = np.sqrt(mean_squared_error(y_val, pred)); r2 = r2_score(y_val, pred)
            fold_metrics["lgb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    LGB  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
        if xgb is not None:
            xgb_model = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, min_child_weight=5, random_state=RANDOM_STATE, early_stopping_rounds=50, eval_metric="mae", verbosity=0)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred = xgb_model.predict(X_val)
            _save_model(xgb_model, f"delta_xgb_{target_name}", fold_idx)
            trained_models["xgb"].append(xgb_model)
            mae = mean_absolute_error(y_val, pred); rmse = np.sqrt(mean_squared_error(y_val, pred)); r2 = r2_score(y_val, pred)
            fold_metrics["xgb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    XGB  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
        if cb is not None:
            cb_model = cb.CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.05, l2_leaf_reg=3, random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=50)
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred = cb_model.predict(X_val)
            _save_model(cb_model, f"delta_cb_{target_name}", fold_idx)
            trained_models["cb"].append(cb_model)
            mae = mean_absolute_error(y_val, pred); rmse = np.sqrt(mean_squared_error(y_val, pred)); r2 = r2_score(y_val, pred)
            fold_metrics["cb"] = {"mae": mae, "rmse": rmse, "r2": r2}
            print(f"    CB   MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")
        results["folds"].append(fold_metrics)
    for model_name in ["lgb", "xgb", "cb"]:
        if results["folds"][0].get(model_name):
            avg_mae = np.mean([f[model_name]["mae"] for f in results["folds"]])
            avg_rmse = np.mean([f[model_name]["rmse"] for f in results["folds"]])
            avg_r2 = np.mean([f[model_name]["r2"] for f in results["folds"]])
            print(f"\n  {model_name.upper()} AVG -> MAE={avg_mae:.4f}  RMSE={avg_rmse:.4f}  R2={avg_r2:.4f}")
    _log_results(results, f"delta_regressor_{target_name}")
    return {"models": trained_models, "results": results}


# ---------------------------------------------------------------------------
# Stage 2 — €100M Classifier
# ---------------------------------------------------------------------------

def train_100m_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Stage 2: Train classifier for P(V >= €100M).
    Handles extreme class imbalance with scale_pos_weight.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — €100M Classifier")
    print(f"{'='*60}")

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    ratio = n_neg / max(n_pos, 1)
    print(f"  Class imbalance: {n_pos} positive / {n_neg} negative (ratio={ratio:.1f}:1)")

    groups = _player_groups(X, meta)
    gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = {"stage": "100m_classifier", "folds": [], "class_ratio": ratio}
    trained_models = {name: [] for name in ["lgb", "xgb", "cb"]}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}: train={len(train_idx)} (pos={y_train.sum()}), val={len(val_idx)} (pos={y_val.sum()})")

        fold_metrics = {"fold": fold_idx + 1}

        # --- LightGBM ---
        if lgb is not None:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            lgb_params = {
                "objective": "binary",
                "metric": "average_precision",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 10,
                "scale_pos_weight": ratio,
                "verbose": -1,
                "seed": RANDOM_STATE,
            }
            callbacks = [lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)]
            lgb_model = lgb.train(
                lgb_params, lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )
            pred_proba = lgb_model.predict(X_val)
            _save_model(lgb_model, f"100m_lgb", fold_idx)
            trained_models["lgb"].append(lgb_model)

            pr_auc = average_precision_score(y_val, pred_proba)
            try:
                roc = roc_auc_score(y_val, pred_proba)
            except ValueError:
                roc = 0.0
            # Optimal threshold via F1
            thresholds = np.arange(0.05, 0.95, 0.01)
            best_f1, best_thresh = 0, 0.5
            for t in thresholds:
                f1 = f1_score(y_val, (pred_proba >= t).astype(int))
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
            fold_metrics["lgb"] = {"pr_auc": pr_auc, "roc_auc": roc, "f1": best_f1, "threshold": best_thresh}
            print(f"    LGB  PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}  F1={best_f1:.4f}  (t={best_thresh:.2f})")

        # --- XGBoost ---
        if xgb is not None:
            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=ratio,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50,
                eval_metric="aucpr",
                verbosity=0,
                use_label_encoder=False,
                enable_categorical=False,
            )
            X_train_xgb = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_val_xgb = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
            xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
            pred_proba = xgb_model.predict_proba(X_val)[:, 1]
            _save_model(xgb_model, f"100m_xgb", fold_idx)
            trained_models["xgb"].append(xgb_model)

            pr_auc = average_precision_score(y_val, pred_proba)
            try:
                roc = roc_auc_score(y_val, pred_proba)
            except ValueError:
                roc = 0.0
            best_f1, best_thresh = 0, 0.5
            for t in np.arange(0.05, 0.95, 0.01):
                f1 = f1_score(y_val, (pred_proba >= t).astype(int))
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
            fold_metrics["xgb"] = {"pr_auc": pr_auc, "roc_auc": roc, "f1": best_f1, "threshold": best_thresh}
            print(f"    XGB  PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}  F1={best_f1:.4f}  (t={best_thresh:.2f})")

        # --- CatBoost ---
        if cb is not None:
            cb_model = cb.CatBoostClassifier(
                iterations=1000,
                depth=5,
                learning_rate=0.05,
                auto_class_weights="SqrtBalanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                early_stopping_rounds=50,
            )
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred_proba = cb_model.predict_proba(X_val)[:, 1]
            _save_model(cb_model, f"100m_cb", fold_idx)
            trained_models["cb"].append(cb_model)

            pr_auc = average_precision_score(y_val, pred_proba)
            try:
                roc = roc_auc_score(y_val, pred_proba)
            except ValueError:
                roc = 0.0
            best_f1, best_thresh = 0, 0.5
            for t in np.arange(0.05, 0.95, 0.01):
                f1 = f1_score(y_val, (pred_proba >= t).astype(int))
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
            fold_metrics["cb"] = {"pr_auc": pr_auc, "roc_auc": roc, "f1": best_f1, "threshold": best_thresh}
            print(f"    CB   PR-AUC={pr_auc:.4f}  ROC-AUC={roc:.4f}  F1={best_f1:.4f}  (t={best_thresh:.2f})")

        results["folds"].append(fold_metrics)

    _log_results(results, "100m_classifier")
    return {"models": trained_models, "results": results}


# ---------------------------------------------------------------------------
# Stage 3 — Minutes Predictor
# ---------------------------------------------------------------------------

def train_minutes_predictor(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> dict:
    """
    Stage 3: Regress Expected_Minutes_Pct at a destination.
    Features: age, ability proxies, target club quality, competition depth.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 3 — Minutes Predictor")
    print(f"{'='*60}")

    groups = _player_groups(X, meta)
    gkf = GroupKFold(n_splits=N_FOLDS)

    results = {"stage": "minutes_predictor", "folds": []}
    trained_models = {name: [] for name in ["lgb", "xgb", "cb"]}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}")

        fold_metrics = {"fold": fold_idx + 1}

        if lgb is not None:
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            lgb_params = {
                "objective": "regression",
                "metric": "mae",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "verbose": -1,
                "seed": RANDOM_STATE,
            }
            callbacks = [lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)]
            lgb_model = lgb.train(
                lgb_params, lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=callbacks,
            )
            pred = lgb_model.predict(X_val)
            _save_model(lgb_model, "minutes_lgb", fold_idx)
            trained_models["lgb"].append(lgb_model)

            mae = mean_absolute_error(y_val, pred)
            fold_metrics["lgb"] = {"mae": mae}
            print(f"    LGB  MAE={mae:.4f}")

        if xgb is not None:
            xgb_model = xgb.XGBRegressor(
                n_estimators=1000, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                early_stopping_rounds=50, eval_metric="mae", verbosity=0,
            )
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            pred = xgb_model.predict(X_val)
            _save_model(xgb_model, "minutes_xgb", fold_idx)
            trained_models["xgb"].append(xgb_model)
            mae = mean_absolute_error(y_val, pred)
            fold_metrics["xgb"] = {"mae": mae}
            print(f"    XGB  MAE={mae:.4f}")

        if cb is not None:
            cb_model = cb.CatBoostRegressor(
                iterations=1000, depth=6, learning_rate=0.05,
                random_seed=RANDOM_STATE, verbose=0, early_stopping_rounds=50,
            )
            cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
            pred = cb_model.predict(X_val)
            _save_model(cb_model, "minutes_cb", fold_idx)
            trained_models["cb"].append(cb_model)
            mae = mean_absolute_error(y_val, pred)
            fold_metrics["cb"] = {"mae": mae}
            print(f"    CB   MAE={mae:.4f}")

        results["folds"].append(fold_metrics)

    _log_results(results, "minutes_predictor")
    return {"models": trained_models, "results": results}


# ---------------------------------------------------------------------------
# Ensemble predict
# ---------------------------------------------------------------------------

def ensemble_predict(models_dict: list, X: pd.DataFrame, task: str = "regression") -> np.ndarray:
    """Average predictions across folds and models."""
    if not models_dict:
        if task == "regression":
            return np.zeros(len(X))
        return np.full(len(X), 0.5)
    all_preds = []
    for model in models_dict:
        if task == "regression":
            pred = model.predict(X)
        else:
            pred = model.predict_proba(X)[:, 1]
        all_preds.append(pred)
    return np.mean(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def train_all_stages(
    X: pd.DataFrame,
    y_log_value: pd.Series,
    meta: pd.DataFrame,
    feature_names: list[str],
    X_cls: pd.DataFrame | None = None,
    y_cls: pd.Series | None = None,
    meta_cls: pd.DataFrame | None = None,
    X_min: pd.DataFrame | None = None,
    y_min: pd.Series | None = None,
    meta_min: pd.DataFrame | None = None,
) -> dict:
    """Train all three stages and return results."""
    results = {}

    # Stage 1: Value regressor
    stage1 = train_value_regressor(X, y_log_value, meta, "log_value", feature_names)
    results["value_regressor"] = stage1

    # Stage 2: Elite classifier
    if X_cls is not None and y_cls is not None and y_cls.sum() > 0:
        meta_cls = meta_cls if meta_cls is not None else meta
        stage2 = train_100m_classifier(X_cls, y_cls, meta_cls, feature_names)
        results["100m_classifier"] = stage2
    else:
        print("\n  [SKIP] Elite Classifier: no positive instances in training data")

    # Stage 3: Minutes predictor
    if X_min is not None and y_min is not None:
        valid_min = y_min.notna()
        if valid_min.sum() > 50:
            meta_m = meta_min if meta_min is not None else meta
            stage3 = train_minutes_predictor(
                X_min[valid_min], y_min[valid_min], meta_m[valid_min]
            )
            results["minutes_predictor"] = stage3
        else:
            print("\n  [SKIP] Minutes Predictor: insufficient training data")
    else:
        print("\n  [SKIP] Minutes Predictor: no data provided")

    return results
