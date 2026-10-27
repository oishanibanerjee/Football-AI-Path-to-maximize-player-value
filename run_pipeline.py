"""
run_pipeline.py — Full Pipeline Orchestrator
Ingests data, engineers features, trains models, simulates pathways.
Run this before launching the Streamlit dashboard.
"""

import sys
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import run_ingestion, build_player_trajectories
from src.feature_engineering import engineer_features, get_feature_columns, build_feature_matrix
from src.models import train_all_stages
from src.simulator import simulate_all_pathways
from src.explainer import explain_player_recommendation, compute_shap_values

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models_saved"


def main():
    print("=" * 70)
    print("  FOOTBALL PLAYER VALUE MAXIMIZATION — PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Task 1: Data Ingestion
    # ------------------------------------------------------------------
    print("\n\n[Stage 1] Data Ingestion & Trajectory Assembly")
    print("-" * 50)

    trajectories, summaries = run_ingestion(verbose=False)
    trajectories.to_parquet(DATA_DIR / "trajectories_raw.parquet", index=False)
    print(f"\n  Raw trajectories saved: {len(trajectories)} rows")

    # ------------------------------------------------------------------
    # Task 2: Feature Engineering
    # ------------------------------------------------------------------
    print("\n\n[Stage 2] Feature Engineering")
    print("-" * 50)

    processed = engineer_features(trajectories)

    # Add position & age dummies to the main dataframe so the simulator can access them
    if "position_group" in processed.columns:
        pos_dummies = pd.get_dummies(processed["position_group"], prefix="pos", dtype=int)
        processed = pd.concat([processed, pos_dummies], axis=1)
    if "age_group" in processed.columns:
        age_dummies = pd.get_dummies(processed["age_group"], prefix="age", dtype=int)
        processed = pd.concat([processed, age_dummies], axis=1)

    processed.to_parquet(DATA_DIR / "trajectories_engineered.parquet", index=False)
    print(f"  Engineered features saved: {processed.shape[1]} columns")

    # ------------------------------------------------------------------
    # Task 3: Build ML datasets and train
    # ------------------------------------------------------------------
    print("\n\n[Stage 3] ML Model Training")
    print("-" * 50)

    feature_cols = get_feature_columns(processed)
    print(f"  Available feature columns: {len(feature_cols)}")

    # --- Compute targets first ---
    # Stage 1: log(value_t+2)
    target_log_val = "log_value_t+2" if "log_value_t+2" in processed.columns else "log_value_t+1"
    if target_log_val not in processed.columns:
        processed["log_value_current"] = np.log1p(processed["value_current"].fillna(0))
        target_log_val = "log_value_current"

    # Stage 2: Elite-value classifier (auto-calibrate threshold)
    elite_threshold = 20_000_000
    for k in [2, 1]:
        fwd_col = f"value_t+{k}"
        if fwd_col in processed.columns and processed[fwd_col].notna().sum() > 0:
            p90 = processed[fwd_col].quantile(0.90)
            elite_threshold = max(p90, 10_000_000)
            processed["target_elite"] = (processed[fwd_col] >= elite_threshold).astype(int)
            n_pos = processed["target_elite"].sum()
            if n_pos >= 5:
                print(f"  Elite classifier threshold: EUR{elite_threshold/1e6:.1f}M (from {fwd_col}, {n_pos} positives)")
                break
    else:
        elite_threshold = processed["value_current"].quantile(0.90)
        processed["target_elite"] = (processed["value_current"] >= elite_threshold).astype(int)
        print(f"  Elite classifier threshold (fallback): EUR{elite_threshold/1e6:.1f}M")

    # --- Build one common feature set from the value regressor stage ---
    X_val, y_val, feat_list_val = build_feature_matrix(processed, target_log_val)
    X_val = X_val.replace([np.inf, -np.inf], 0).fillna(0)
    meta_val = processed.loc[X_val.index, ["player_id"]].copy()
    print(f"  Value regressor dataset: {X_val.shape} | Target: {target_log_val}")
    print(f"  Features: {len(feat_list_val)}")

    # Use the same features for classifier and minutes predictor
    common_features = [f for f in feat_list_val if f in processed.columns and f != "target_elite"]

    # Classifier - fill NaNs/inf in features so we don't lose all rows
    cls_target_valid = processed["target_elite"].notna()
    X_100m = processed.loc[cls_target_valid, common_features].replace([np.inf, -np.inf], 0).fillna(0)
    y_100m = processed.loc[cls_target_valid, "target_elite"].astype(int)
    meta_100m = processed.loc[cls_target_valid, ["player_id"]].copy()
    print(f"  Elite classifier dataset: {X_100m.shape} | Positives: {y_100m.sum()}")

    # Minutes predictor - fill NaNs/inf in features
    min_target_valid = processed["expected_minutes_pct"].notna()
    X_min = processed.loc[min_target_valid, common_features].replace([np.inf, -np.inf], 0).fillna(0)
    y_min = processed.loc[min_target_valid, "expected_minutes_pct"]
    meta_min = processed.loc[min_target_valid, ["player_id"]].copy()
    print(f"  Minutes predictor dataset: {X_min.shape}")

    # Train all stages (each with its own feature matrix and meta)
    models = train_all_stages(
        X_val, y_val, meta_val, feat_list_val,
        X_100m, y_100m, meta_100m,
        X_min, y_min, meta_min,
    )

    # Save feature columns for inference
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump({
            "value_features": feat_list_val,
            "classifier_features": common_features,
            "minutes_features": common_features,
        }, f)

    # ------------------------------------------------------------------
    # Task 4: Pathway Simulation (example for a few players)
    # ------------------------------------------------------------------
    print("\n\n[Stage 4] Pathway Simulation")
    print("-" * 50)

    # Get players with most data
    player_counts = processed.groupby("player_id").size().sort_values(ascending=False)
    top_players = player_counts.head(10).index.tolist()

    all_pathway_results = {}
    for pid in top_players:
        pdata = processed[processed["player_id"] == pid]
        if pdata.empty:
            continue

        player_name = pdata["player_name"].iloc[0] if "player_name" in pdata.columns else f"Player_{pid}"
        current_club = pdata["club_name"].iloc[-1] if "club_name" in pdata.columns else "Unknown"
        current_tier = pdata["club_tier"].iloc[-1] if "club_tier" in pdata.columns else 3
        league_tier = pdata["league_tier"].iloc[-1] if "league_tier" in pdata.columns else 1

        print(f"\n  Simulating pathways for: {player_name} (id={pid})")

        pathways = simulate_all_pathways(
            player_id=pid,
            player_name=player_name,
            current_club=current_club,
            current_tier=int(current_tier),
            current_league_tier=int(league_tier),
            trajectory_df=processed,
            feature_columns=feat_list_val,
            models=models,
        )

        # Generate SHAP explanation for top 2 pathways
        if len(pathways) >= 2 and models.get("value_regressor", {}).get("models", {}).get("lgb"):
            lgb_model = models["value_regressor"]["models"]["lgb"][0]
            player_latest = pdata[feat_list_val].fillna(0).iloc[[-1]]

            explanation = explain_player_recommendation(
                lgb_model, player_latest, feat_list_val, player_name
            )
            print(f"    Top SHAP features: {explanation['top_features']['feature'].head(5).tolist()}")

        all_pathway_results[pid] = {
            "name": player_name,
            "club": current_club,
            "pathways": [
                {
                    "name": p.name,
                    "description": p.description,
                    "score": p.composite_score,
                    "risk": p.risk_penalty,
                    "steps": [
                        {
                            "season": s.season,
                            "club": s.club,
                            "minutes_pct": s.predicted_minutes_pct,
                            "value": s.predicted_value,
                            "prob_100m": s.predicted_100m_prob,
                        }
                        for s in p.steps
                    ],
                }
                for p in pathways
            ],
        }

    # Save pathway results
    with open(MODELS_DIR / "pathway_results.json", "w") as f:
        json.dump(all_pathway_results, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Raw trajectories:    {trajectories.shape}")
    print(f"  Engineered features: {processed.shape}")
    print(f"  Features used:       {len(feat_list_val)}")
    print(f"  Models trained:      3 stages × 3 algorithms × 5 folds")
    print(f"  Players simulated:   {len(all_pathway_results)}")
    print(f"\n  Saved artifacts:")
    print(f"    data/trajectories_raw.parquet")
    print(f"    data/trajectories_engineered.parquet")
    print(f"    models_saved/*.pkl  (trained models)")
    print(f"    models_saved/results_*.json  (metrics)")
    print(f"    models_saved/feature_columns.json")
    print(f"    models_saved/pathway_results.json")
    print(f"\n  Launch dashboard:  streamlit run app.py")

    return processed, models, all_pathway_results


if __name__ == "__main__":
    processed, models, pathways = main()
