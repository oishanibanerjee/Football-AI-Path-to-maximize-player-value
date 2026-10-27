"""
app.py — Interactive Streamlit Dashboard
Football Player Value Maximization & €100M Pathway Optimization Engine
"""

import json
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Ensure src is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.feature_engineering import engineer_features, get_feature_columns, build_feature_matrix, POSITION_MAP
from src.simulator import (
    simulate_all_pathways,
    generate_status_quo_path,
    generate_elite_jump_path,
    generate_stepping_stone_path,
    generate_loan_path,
    CLUB_UNIVERSE,
)
from src.explainer import compute_shap_values, compare_pathways_shap

st.set_page_config(
    page_title="Football Player Value Optimizer",
    page_icon="⚽",
    layout="wide",
)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models_saved"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_engineered_data():
    path = DATA_DIR / "trajectories_engineered.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_raw_data():
    path = DATA_DIR / "trajectories_raw.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data
def load_feature_columns():
    path = MODELS_DIR / "feature_columns.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_pathway_results():
    # Check primary location (legacy pipeline)
    for p in [MODELS_DIR / "pathway_results.json", ROOT / "models" / "pathway_demo_results.json", ROOT / "models" / "counterfactual_delta_results.json"]:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                    # Normalize demo format (keyed A/B/C/D) vs pipeline format (pid -> {name, pathways})
                    if data and isinstance(data, dict):
                        # If data looks like demo output (has "pathways" key at top level is sample player, not useful)
                        # Return raw; render functions handle both
                        # But prefer pipeline format if it has many players
                        if "pathways" in data and "sample_player" in data:
                            # This is pathway_demo_results.json — wrap as single-player entry for compatibility
                            sp = data.get("sample_player", {})
                            pname = sp.get("player_name", "Demo Player")
                            # Convert A/B/C/D dict to list format
                            demo_pathways = []
                            for k, v in data.get("pathways", {}).items():
                                demo_pathways.append({
                                    "name": v.get("name", k),
                                    "description": v.get("name", k),
                                    "score": v.get("score", 0),
                                    "risk": v.get("risk", 0.1),
                                    "steps": [
                                        {"season": "t+1", "club": "Simulated", "minutes_pct": v.get("avg_minutes_share", 0.5), "value": v.get("predicted_value_2yr", 0), "prob_100m": v.get("p100", 0)},
                                        {"season": "t+2", "club": "Simulated", "minutes_pct": v.get("avg_minutes_share", 0.5), "value": v.get("predicted_value_2yr", 0), "prob_100m": v.get("p100", 0)},
                                    ],
                                })
                            return {"demo": {"name": pname, "pathways": demo_pathways}}
                        # If we loaded models_saved/pathway_results.json and it is non-empty, return it immediately
                        if len(data) >= 5:
                            return data
                        # Otherwise keep searching; but if this is the only file, return it
                        if p == MODELS_DIR / "pathway_results.json":
                            return data
            except Exception:
                continue
    # Fallback: try legacy path again
    path = MODELS_DIR / "pathway_results.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_models():
    models = {}
    for stage, name_prefix in [
        ("value_regressor", "value_lgb_log_value_current"),
        ("100m_classifier", "100m_lgb"),
        ("minutes_predictor", "minutes_lgb"),
    ]:
        stage_models = []
        for fold in range(5):
            model_path = MODELS_DIR / f"{name_prefix}_fold{fold}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    stage_models.append(pickle.load(f))
        if stage_models:
            models[stage] = stage_models
    return models


# ---------------------------------------------------------------------------
# Page layouts
# ---------------------------------------------------------------------------

def render_header():
    st.title("Football Player Value Maximization")
    st.subheader("€100M Pathway Optimization Engine")
    st.divider()


def render_player_selector(df: pd.DataFrame) -> dict | None:
    """Sidebar: Player filtering and selection."""
    st.sidebar.header("Player Selector")

    if df is None or df.empty:
        st.sidebar.warning("No data loaded")
        return None

    # Filters
    if "position_group" in df.columns:
        positions = sorted(df["position_group"].dropna().unique())
        sel_pos = st.sidebar.multiselect("Position Group", positions, default=positions)

    if "club_name" in df.columns:
        clubs = sorted(df["club_name"].dropna().unique())
        sel_club = st.sidebar.selectbox("Club", ["All"] + clubs)

    if "age" in df.columns:
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        sel_age = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max))

    if "value_current" in df.columns:
        val_min = float(df["value_current"].min())
        val_max = float(df["value_current"].max())
        sel_val = st.sidebar.slider(
            "Market Value Range (€M)",
            float(val_min) / 1e6,
            float(val_max) / 1e6,
            (float(val_min) / 1e6, float(val_max) / 1e6),
        )

    # Filter
    filtered = df.copy()
    if "position_group" in df.columns:
        filtered = filtered[filtered["position_group"].isin(sel_pos)]
    if sel_club != "All" and "club_name" in df.columns:
        filtered = filtered[filtered["club_name"] == sel_club]
    if "age" in df.columns:
        filtered = filtered[(filtered["age"] >= sel_age[0]) & (filtered["age"] <= sel_age[1])]
    if "value_current" in df.columns:
        filtered = filtered[
            (filtered["value_current"] >= sel_val[0] * 1e6)
            & (filtered["value_current"] <= sel_val[1] * 1e6)
        ]

    # Player selection
    if "player_name" in filtered.columns:
        player_names = sorted(filtered["player_name"].dropna().unique())
    else:
        player_names = [f"Player {pid}" for pid in filtered["player_id"].unique()]

    sel_player = st.sidebar.selectbox("Select Player", player_names)

    if sel_player and "player_name" in filtered.columns:
        player_data = filtered[filtered["player_name"] == sel_player]
    else:
        player_data = filtered.head(1)

    return {
        "player_data": player_data,
        "filtered_data": filtered,
        "player_name": sel_player,
    }


def render_diagnostic_panel(player_data: pd.DataFrame, player_name: str):
    """Current Diagnostic Panel with radar chart."""
    st.header(f"Diagnostic: {player_name}")

    if player_data.empty:
        st.warning("No data available for this player")
        return

    latest = player_data.sort_values("super_season_id").iloc[-1] if "super_season_id" in player_data.columns else player_data.iloc[-1]

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        val = latest.get("value_current", 0)
        if pd.notna(val) and val > 0:
            st.metric("Market Value", f"€{val/1e6:.1f}M")
        else:
            st.metric("Market Value", "N/A")

    with col2:
        age = latest.get("age", "N/A")
        st.metric("Age", f"{age:.0f}" if pd.notna(age) else "N/A")

    with col3:
        mins = latest.get("expected_minutes_pct", 0)
        if pd.notna(mins):
            st.metric("Minutes %", f"{mins*100:.0f}%")
        else:
            st.metric("Minutes %", "N/A")

    with col4:
        pos = latest.get("position_group", "N/A")
        st.metric("Position", str(pos))

    # Radar chart — position-specific per-90 metrics
    st.subheader("Performance Radar")

    radar_metrics = _get_radar_metrics(latest, str(latest.get("position_group", "Unknown")))

    if radar_metrics:
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        values += values[:1]  # close the polygon
        categories += categories[:1]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=player_name,
            line=dict(color="#1f77b4"),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Development gaps
    st.subheader("Development Gaps")
    gaps = _identify_gaps(latest, str(latest.get("position_group", "Unknown")))
    if gaps:
        for gap in gaps:
            st.info(gap)


def _get_radar_metrics(player_row: pd.Series, position_group: str) -> dict:
    """Get position-appropriate radar metrics normalized 0-1."""
    metrics = {}

    # Common metrics
    radar_cols = {
        "goals_p90": ("Goals/90", 0.8),
        "assists_p90": ("Assists/90", 0.6),
        "expected_minutes_pct": ("Playing Time", 1.0),
        "shots_p90": ("Shots/90", 4.0),
        "key_passes_p90": ("Key Passes/90", 2.5),
        "tackles_p90": ("Tackles/90", 3.0),
        "interceptions_p90": ("Interceptions/90", 2.0),
        "dribbles_p90": ("Dribbles/90", 3.0),
        "aerials_won_p90": ("Aerials Won/90", 3.0),
    }

    for col, (label, max_val) in radar_cols.items():
        if col in player_row.index and pd.notna(player_row[col]):
            metrics[label] = min(player_row[col] / max_val, 1.0)

    return metrics


def _identify_gaps(player_row: pd.Series, position_group: str) -> list[str]:
    """Identify development gaps based on position."""
    gaps = []

    if position_group == "Attacker":
        if player_row.get("goals_p90", 0) < 0.3:
            gaps.append("Goals per 90 is below elite attacker benchmark (0.3)")
        if player_row.get("xg_p90", 0) < 0.3:
            gaps.append("Expected goals (xG) suggests underperformance in finishing")
        if player_row.get("dribbles_p90", 0) < 1.5:
            gaps.append("Dribble success rate below average for creative attackers")

    elif position_group == "Midfielder":
        if player_row.get("passes_p90", 0) < 30:
            gaps.append("Pass volume below standard for starting midfielder")
        if player_row.get("key_passes_p90", 0) < 0.5:
            gaps.append("Key passes below average — chance creation gap")
        if player_row.get("tackles_p90", 0) < 1.0:
            gaps.append("Defensive contribution low for a midfielder")

    elif position_group == "Defender":
        if player_row.get("tackles_p90", 0) < 1.5:
            gaps.append("Tackle rate below benchmark for starting defender")
        if player_row.get("interceptions_p90", 0) < 1.0:
            gaps.append("Interception rate low — reading of the game to develop")
        if player_row.get("aerials_won_p90", 0) < 1.5:
            gaps.append("Aerial duels won below average")

    # Common
    if player_row.get("expected_minutes_pct", 1) < 0.4:
        gaps.append("Limited playing time — high bench risk detected")

    return gaps


def render_pathway_comparison(player_name: str, pathway_results: dict, df: pd.DataFrame):
    """Pathway Comparison Matrix."""
    st.header("Pathway Comparison Matrix")

    # Find player in pathway results
    player_pathways = None
    for pid, pdata in pathway_results.items():
        if pdata.get("name") == player_name:
            player_pathways = pdata
            break

    if not player_pathways:
        st.info("No pre-computed pathways found. Showing template comparison.")

        # Create template pathways for display
        player_data = df[df.get("player_name", "player_name") == player_name] if "player_name" in df.columns else df
        if player_data.empty:
            return

        latest = player_data.iloc[-1]
        current_club = str(latest.get("club_name", "Unknown"))

        pathways = [
            ("Path A — Status Quo", f"Remain at {current_club}", 0.01, 0.005, 0.55, 0.3),
            ("Path B — Elite Jump", f"Move to Real Madrid", 0.04, 0.02, 0.30, 0.3),
            ("Path C — Stepping Stone", f"Real Sociedad → Real Madrid", 0.06, 0.03, 0.75, 0.1),
            ("Path D — Loan Path", f"Loan → Return to {current_club}", 0.03, 0.01, 0.50, 0.2),
        ]

        rows = []
        for name, desc, score, prob, mins, risk in pathways:
            rows.append({
                "Pathway": name,
                "Description": desc,
                "Composite Score": f"{score:.3f}",
                "P(≥€100M)": f"{prob*100:.1f}%",
                "Expected Minutes %": f"{mins*100:.0f}%",
                "Risk Level": f"{risk:.1f}",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    # Real pathway data
    rows = []
    for pw in player_pathways.get("pathways", []):
        final_step = pw["steps"][-1] if pw["steps"] else {}
        avg_mins = np.mean([s.get("minutes_pct", 0) for s in pw["steps"]]) if pw["steps"] else 0
        rows.append({
            "Pathway": pw["name"],
            "Description": pw.get("description", ""),
            "Composite Score": f"{pw['score']:.3f}",
            "P(≥€100M)": f"{final_step.get('prob_100m', 0)*100:.1f}%",
            "Expected Value (€M)": f"{final_step.get('value', 0)/1e6:.1f}",
            "Expected Minutes %": f"{avg_mins*100:.0f}%",
            "Risk Level": f"{pw['risk']:.1f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Score bar chart
    if rows:
        names = [r["Pathway"] for r in rows]
        scores = [float(r["Composite Score"]) for r in rows]

        fig = go.Figure(data=go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=["#2ecc71", "#3498db", "#e74c3c", "#f39c12"],
        ))
        fig.update_layout(
            title="Pathway Composite Scores",
            xaxis_title="Score",
            margin=dict(l=200, r=40, t=50, b=40),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_counterfactual_flow(player_name: str, pathway_results: dict):
    """Counterfactual Flow Visualizer — with live fallback via simulator."""
    st.header("Counterfactual Flow Visualizer")

    player_pathways = None
    for pid, pdata in pathway_results.items():
        if pdata.get("name") == player_name:
            player_pathways = pdata
            break

    # Live fallback: generate pathways on-the-fly if not precomputed (e.g., 1255/1265 players not in top-10)
    if not player_pathways or not player_pathways.get("pathways"):
        # Try live simulation using available models / new delta engine
        try:
            df = load_engineered_data()
            feature_cols = load_feature_columns()
            models = load_models()
            # Prefer new PathwaySimulator (delta) if available, else legacy simulator
            use_new = (ROOT / "src" / "pathway_simulator.py").exists() and (ROOT / "models" / "counterfactual_delta_regressor.pkl").exists()
            if df is not None and not df.empty and feature_cols:
                # Locate player row
                pdata_sel = df[df["player_name"] == player_name] if "player_name" in df.columns else df[df["player_id"].astype(str) == str(player_name)]
                if not pdata_sel.empty:
                    st.info(f"No pre-computed pathways for '{player_name}' (only top-10 cached) — generating live counterfactuals...")
                    if use_new:
                        from src.pathway_simulator import PathwaySimulator
                        from src.pathway_ranker import PathwayRanker
                        base_row = pdata_sel.sort_values("super_season_id").iloc[-1].to_dict() if "super_season_id" in pdata_sel.columns else pdata_sel.iloc[-1].to_dict()
                        curr_val = float(base_row.get("value_current", base_row.get("current_market_value", 5_000_000)) or 5_000_000)
                        sim = PathwaySimulator(load_from_disk=True)
                        raw_results = sim.simulate(base_row, current_value=curr_val, verbose=False)
                        ranker = PathwayRanker()
                        scored, _ = ranker.rank_and_explain(raw_results, baseline_key="A")
                        # Convert to app's expected format (models_saved/pathway_results.json shape)
                        pathways = []
                        for key in sorted(scored, key=lambda k: scored[k]["rank"]):
                            r = scored[key]
                            steps = []
                            for pred in r["steps_pred"]:
                                syn = pred["synthetic_row"]
                                # Map season names: t+1, t+2
                                steps.append({
                                    "season": pred["season"],
                                    "club": syn.get("club_name", pred.get("name", "Simulated Club")) if isinstance(syn.get("club_name"), str) else r["name"].split("—")[-1].strip()[:20],
                                    "minutes_pct": float(syn.get("minutes_share", syn.get("expected_minutes_pct", 0.5))),
                                    "value": float(pred["predicted_value"]),
                                    "prob_100m": float(pred["p100"]),
                                    "description": r["description"],
                                })
                            # Fallback club extraction from description if needed
                            pathways.append({
                                "name": r["name"],
                                "description": r["description"],
                                "score": float(r["score"]),
                                "risk": float(r["risk"]),
                                "steps": steps,
                            })
                        player_pathways = {"name": player_name, "pathways": pathways}
                    else:
                        # Legacy fallback via src.simulator
                        feat_list = feature_cols.get("value_features", [])
                        if feat_list and not pdata_sel.empty:
                            latest = pdata_sel.sort_values("super_season_id").iloc[-1] if "super_season_id" in pdata_sel.columns else pdata_sel.iloc[-1]
                            cur_club = str(latest.get("club_name", "Unknown"))
                            cur_tier = int(latest.get("club_tier", 3)) if pd.notna(latest.get("club_tier", 3)) else 3
                            cur_league = int(latest.get("league_tier", latest.get("league_uefa_rank", 1))) if pd.notna(latest.get("league_tier", 1)) else 1
                            pid_val = int(latest.get("player_id", 0))
                            live = simulate_all_pathways(
                                player_id=pid_val, player_name=player_name,
                                current_club=cur_club, current_tier=cur_tier,
                                current_league_tier=cur_league,
                                trajectory_df=df, feature_columns=feat_list, models=models, n_seasons=3
                            )
                            if live:
                                pathways = []
                                for pw in live:
                                    pathways.append({
                                        "name": pw.name,
                                        "description": pw.description,
                                        "score": float(pw.composite_score),
                                        "risk": float(pw.risk_penalty),
                                        "steps": [
                                            {"season": s.season, "club": s.club, "minutes_pct": float(s.predicted_minutes_pct), "value": float(s.predicted_value), "prob_100m": float(s.predicted_100m_prob), "description": s.description}
                                            for s in pw.steps
                                        ],
                                    })
                                player_pathways = {"name": player_name, "pathways": pathways}
        except Exception as e:
            st.warning(f"Live simulation failed: {e}")
            # fall through to no-data message
            pass

    if not player_pathways or not player_pathways.get("pathways"):
        st.info("No pathway data available. Run the pipeline first.")
        st.caption("Tip: Precomputed pathways only cover top-10 players by history; live fallback will auto-generate for any of the 1,265 players if models exist.")
        return

    # Select pathway to visualize
    pw_names = [p["name"] for p in player_pathways["pathways"]]
    sel_pw = st.selectbox("Select Pathway to Visualize", pw_names)

    for pw in player_pathways["pathways"]:
        if pw["name"] == sel_pw:
            steps = pw["steps"]
            if not steps:
                continue

            # Create Sankey-like flow
            fig = go.Figure()

            seasons = [s.get("season", f"t+{i+1}") for i, s in enumerate(steps)]
            clubs = [s.get("club", "Unknown") for s in steps]
            minutes = [s.get("minutes_pct", 0) * 100 for s in steps]
            values = [s.get("value", 0) / 1e6 for s in steps]

            # Line chart: minutes trajectory
            fig.add_trace(go.Scatter(
                x=seasons,
                y=minutes,
                mode="lines+markers+text",
                name="Expected Minutes %",
                text=[f"{m:.0f}%" for m in minutes],
                textposition="top center",
                line=dict(width=3, color="#3498db"),
                marker=dict(size=12),
            ))

            fig.update_layout(
                title=f"Trajectory: {sel_pw}",
                xaxis_title="Season",
                yaxis_title="Expected Playing Time %",
                yaxis=dict(range=[0, 100]),
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Club flow table
            flow_data = []
            for i, s in enumerate(steps):
                flow_data.append({
                    "Season": s.get("season", f"t+{i+1}"),
                    "Club": s.get("club", "Unknown"),
                    "Minutes %": f"{s.get('minutes_pct', 0)*100:.0f}%",
                    "Predicted Value": f"€{s.get('value', 0)/1e6:.1f}M",
                    "P(€100M)": f"{s.get('prob_100m', 0)*100:.1f}%",
                    "Description": s.get("description", ""),
                })

            st.table(pd.DataFrame(flow_data))
            break


def render_shap_panel(player_name: str, df: pd.DataFrame, feature_cols: dict, models: dict):
    """SHAP Explanation Panel."""
    st.header("SHAP Feature Importance")

    if not models.get("value_regressor"):
        st.warning("No trained value regressor models found.")
        return

    lgb_model = models["value_regressor"][0]

    features_list = feature_cols.get("value_features", [])
    if not features_list:
        st.warning("No feature columns found.")
        return

    # Get player data
    if "player_name" in df.columns:
        player_data = df[df["player_name"] == player_name]
    else:
        player_data = df

    if player_data.empty:
        st.warning("No data for this player.")
        return

    X_player = player_data[features_list].fillna(0).iloc[[-1]]

    # Compute SHAP values
    result = compute_shap_values(lgb_model, X_player, features_list)

    if result["feature_importance"] is not None:
        top_n = result["feature_importance"].head(20)

        fig = go.Figure(data=go.Bar(
            x=top_n["mean_abs_shap"].values,
            y=top_n["feature"].values,
            orientation="h",
            marker_color="#e74c3c",
        ))
        fig.update_layout(
            title=f"Top 20 Feature Importances — {player_name}",
            xaxis_title="Mean |SHAP value|",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=250, r=40, t=50, b=40),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        st.subheader("Explanation Summary")
        top_feat = result["feature_importance"].iloc[0]
        st.write(
            f"The most important feature for this prediction is **{top_feat['feature']}** "
            f"(mean |SHAP| = {top_feat['mean_abs_shap']:.4f})."
        )

        if result["shap_values"] is not None:
            shap_vals = result["shap_values"][0]
            top_5_idx = np.argsort(np.abs(shap_vals))[-5:][::-1]
            for idx in top_5_idx:
                feat = features_list[idx]
                val = shap_vals[idx]
                direction = "↑ positive" if val > 0 else "↓ negative"
                st.write(f"  • **{feat}**: {direction} impact ({val:.4f})")

    # Pathway comparison (if multiple pathways exist)
    st.subheader("Path C vs Path B — Why Stepping Stone Wins")
    if result["shap_values"] is not None:
        # Create synthetic comparison
        X_path_b = X_player.copy()
        X_path_c = X_player.copy()

        if "club_tier" in X_path_b.columns:
            X_path_b["club_tier"] = 1
            X_path_c["club_tier"] = 3
        if "expected_minutes_pct" in X_path_b.columns:
            X_path_b["expected_minutes_pct"] = 0.30
            X_path_c["expected_minutes_pct"] = 0.75
        if "squad_competition_density" in X_path_b.columns:
            X_path_b["squad_competition_density"] = 8
            X_path_c["squad_competition_density"] = 4
        if "teammate_avg_value" in X_path_b.columns:
            X_path_b["teammate_avg_value"] = X_path_b["teammate_avg_value"] * 2
            X_path_c["teammate_avg_value"] = X_path_c["teammate_avg_value"] * 0.8

        comparison = compare_pathways_shap(
            lgb_model, X_path_b, X_path_c, features_list,
            "Path B (Elite Jump)", "Path C (Stepping Stone)",
        )

        if comparison.get("summary"):
            st.markdown(comparison["summary"])

        if comparison.get("comparison_df") is not None:
            comp_df = comparison["comparison_df"].head(10)
            fig = go.Figure(data=go.Bar(
                x=comp_df["delta_shap"].values if "delta_shap" in comp_df.columns else comp_df.get("weighted_imp", pd.Series()).values,
                y=comp_df["feature"].values,
                orientation="h",
                marker_color=["#2ecc71" if v > 0 else "#e74c3c" for v in
                              (comp_df["delta_shap"].values if "delta_shap" in comp_df.columns else comp_df.get("weighted_imp", pd.Series()).values)],
            ))
            fig.update_layout(
                title="SHAP Delta: Path C vs Path B",
                xaxis_title="SHAP Delta (C - B)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=250, r=40, t=50, b=40),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    render_header()

    df = load_engineered_data()
    feature_cols = load_feature_columns()
    pathway_results = load_pathway_results()
    models = load_models()

    if df is None:
        st.error(
            "No processed data found. Run `python run_pipeline.py` first to "
            "ingest data, engineer features, and train models."
        )
        st.code("python run_pipeline.py", language="bash")
        st.stop()

    st.sidebar.success(f"Data loaded: {len(df)} records")

    selector = render_player_selector(df)
    if selector is None:
        st.stop()

    player_data = selector["player_data"]
    player_name = selector["player_name"]

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Diagnostic",
        "Pathway Comparison",
        "Counterfactual Flow",
        "SHAP Explanation",
    ])

    with tab1:
        render_diagnostic_panel(player_data, player_name)

    with tab2:
        render_pathway_comparison(player_name, pathway_results, df)

    with tab3:
        render_counterfactual_flow(player_name, pathway_results)

    with tab4:
        render_shap_panel(player_name, df, feature_cols, models)


if __name__ == "__main__":
    main()
