"""
explainer.py — SHAP Explainability Engine
Provides feature importance breakdowns for pathway recommendations,
comparing why one path outranks another.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models_saved"


# ---------------------------------------------------------------------------
# SHAP value computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> dict:
    """
    Compute SHAP values for a given model and feature matrix.
    Returns dict with 'shap_values', 'expected_value', 'feature_importance'.
    """
    if shap is None:
        print("[Explainer] SHAP not installed. Using feature importances fallback.")
        return _fallback_importance(model, X, feature_names)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classifiers, shap_values may be a list [class_0, class_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        expected_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

        # Mean absolute SHAP importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            "feature": feature_names or X.columns.tolist(),
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": np.mean(shap_values, axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return {
            "shap_values": shap_values,
            "expected_value": expected_value,
            "feature_importance": feature_importance,
        }
    except Exception as e:
        print(f"[Explainer] SHAP computation failed: {e}")
        return _fallback_importance(model, X, feature_names)


def _fallback_importance(model, X, feature_names):
    """Fallback to native feature importances when SHAP unavailable."""
    names = feature_names or X.columns.tolist()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "feature_importance"):
        importances = model.feature_importance(importance_type="gain")
    else:
        importances = np.zeros(len(names))

    feature_importance = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": importances / max(importances.sum(), 1),
        "mean_shap": importances,
    }).sort_values("mean_abs_shap", ascending=False)

    return {
        "shap_values": None,
        "expected_value": 0,
        "feature_importance": feature_importance,
    }


# ---------------------------------------------------------------------------
# Pathway comparison explanations
# ---------------------------------------------------------------------------

def compare_pathways_shap(
    model,
    X_path_a: pd.DataFrame,
    X_path_b: pd.DataFrame,
    feature_names: list[str],
    path_a_name: str = "Path A",
    path_b_name: str = "Path B",
) -> dict:
    """
    Explain why Path B ranks higher than Path A (or vice versa).
    Returns top positive and negative drivers.
    """
    if shap is None:
        return _compare_fallback(model, X_path_a, X_path_b, feature_names, path_a_name, path_b_name)

    try:
        explainer = shap.TreeExplainer(model)

        shap_a = explainer.shap_values(X_path_a)
        shap_b = explainer.shap_values(X_path_b)

        if isinstance(shap_a, list):
            shap_a = shap_a[1]
        if isinstance(shap_b, list):
            shap_b = shap_b[1]

        # Difference in SHAP values
        delta_shap = (shap_b - shap_a).flatten()

        comparison = pd.DataFrame({
            "feature": feature_names[:len(delta_shap)],
            "delta_shap": delta_shap,
            "abs_delta": np.abs(delta_shap),
        }).sort_values("abs_delta", ascending=False)

        top_positive = comparison[comparison["delta_shap"] > 0].head(10)
        top_negative = comparison[comparison["delta_shap"] < 0].head(10)

        return {
            "delta_shap": delta_shap,
            "comparison_df": comparison,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "summary": _generate_explanation_text(
                top_positive, top_negative, path_a_name, path_b_name
            ),
        }
    except Exception as e:
        print(f"[Explainer] Pathway comparison failed: {e}")
        return _compare_fallback(model, X_path_a, X_path_b, feature_names, path_a_name, path_b_name)


def _compare_fallback(model, X_a, X_b, feature_names, name_a, name_b):
    """Fallback comparison using prediction differences."""
    pred_a = model.predict(X_a)
    pred_b = model.predict(X_b)
    delta = (pred_b - pred_a).flatten()

    # Use feature differences as proxy
    feat_diff = (X_b.values - X_a.values).flatten()
    importance = model.feature_importances_ if hasattr(model, "feature_importances_") else np.ones(len(feature_names))

    comparison = pd.DataFrame({
        "feature": feature_names[:len(feat_diff)],
        "value_diff": feat_diff,
        "weighted_imp": feat_diff * importance[:len(feat_diff)],
    }).sort_values("weighted_imp", key=abs, ascending=False)

    return {
        "delta_shap": delta,
        "comparison_df": comparison,
        "top_positive_drivers": comparison[comparison["weighted_imp"] > 0].head(10),
        "top_negative_drivers": comparison[comparison["weighted_imp"] < 0].head(10),
        "summary": f"{name_b} scores higher than {name_a} by {abs(delta[0]):.4f} (log-value scale)",
    }


def _generate_explanation_text(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    path_a_name: str,
    path_b_name: str,
) -> str:
    """Generate human-readable explanation."""
    lines = [
        f"**Why {path_b_name} outranks {path_a_name}:**\n",
        "**Positive drivers (favoring the recommended path):**",
    ]

    for _, row in top_positive.iterrows():
        feat = row["feature"]
        val = row["delta_shap"]
        direction = "increases" if val > 0 else "decreases"
        lines.append(f"  • {feat}: {direction} predicted value by {abs(val):.4f}")

    lines.append(f"\n**Negative drivers (costs of this path):**")
    for _, row in top_negative.iterrows():
        feat = row["feature"]
        val = row["delta_shap"]
        lines.append(f"  • {feat}: reduces predicted value by {abs(val):.4f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single-player explanation
# ---------------------------------------------------------------------------

def explain_player_recommendation(
    model,
    player_features: pd.DataFrame,
    feature_names: list[str],
    player_name: str,
) -> dict:
    """Generate full SHAP explanation for a single player's recommendation."""
    result = compute_shap_values(model, player_features, feature_names)

    top_features = result["feature_importance"].head(15)

    explanation = {
        "player_name": player_name,
        "expected_value": result["expected_value"],
        "top_features": top_features,
        "shap_values_available": result["shap_values"] is not None,
    }

    return explanation


# ---------------------------------------------------------------------------
# Plotting helpers (return figure objects for Streamlit)
# ---------------------------------------------------------------------------

def create_shap_summary_plot(model, X: pd.DataFrame, feature_names: list[str]):
    """Create SHAP summary plot figure."""
    if shap is None:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig = shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    return fig


def create_shap_waterfall_plot(model, X_row: pd.DataFrame, feature_names: list[str]):
    """Create SHAP waterfall plot for a single prediction."""
    if shap is None:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    try:
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value,
            data=X_row.iloc[0].values,
            feature_names=feature_names,
        )
        return explanation
    except Exception:
        return None
