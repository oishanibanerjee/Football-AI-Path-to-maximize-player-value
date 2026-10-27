"""
simulator.py — Counterfactual Pathway Simulation & Recommendation Engine
Generates hypothetical career trajectories, scores them, and recommends
the optimal pathway for maximizing market value and P(>=€100M).
"""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Pathway definitions
# ---------------------------------------------------------------------------

@dataclass
class PathwayStep:
    season: str
    club: str
    club_tier: int
    league_tier: int
    predicted_minutes_pct: float
    predicted_value: float
    predicted_100m_prob: float
    description: str = ""


@dataclass
class Pathway:
    name: str
    description: str
    steps: list[PathwayStep] = field(default_factory=list)
    composite_score: float = 0.0
    risk_penalty: float = 0.0


# ---------------------------------------------------------------------------
# Club universe for simulation
# ---------------------------------------------------------------------------

CLUB_UNIVERSE = {
    # Tier 1 — Elite
    1: [
        ("Real Madrid", "Spain", 1),
        ("FC Barcelona", "Spain", 1),
        ("Atlético Madrid", "Spain", 1),
        ("Manchester City", "England", 1),
        ("Liverpool", "England", 1),
        ("Arsenal", "England", 1),
        ("Chelsea", "England", 1),
        ("Bayern Munich", "Germany", 1),
        ("Paris Saint-Germain", "France", 1),
        ("Juventus", "Italy", 1),
        ("AC Milan", "Italy", 1),
        ("Inter Milan", "Italy", 1),
        ("Napoli", "Italy", 1),
        ("Borussia Dortmund", "Germany", 1),
    ],
    # Tier 2 — Strong
    2: [
        ("Sevilla", "Spain", 1),
        ("Real Sociedad", "Spain", 1),
        ("Villarreal", "Spain", 1),
        ("Athletic Bilbao", "Spain", 1),
        ("Tottenham", "England", 1),
        ("Newcastle United", "England", 1),
        ("Aston Villa", "England", 1),
        ("Atalanta", "Italy", 1),
        ("AS Roma", "Italy", 1),
        ("Bayer Leverkusen", "Germany", 1),
        ("Eintracht Frankfurt", "Germany", 1),
        ("Monaco", "France", 1),
        ("Benfica", "Portugal", 2),
        ("Porto", "Portugal", 2),
        ("Sporting CP", "Portugal", 2),
    ],
    # Tier 3 — Development
    3: [
        ("Real Betis", "Spain", 1),
        ("Valencia", "Spain", 1),
        ("Girona", "Spain", 1),
        ("Celta Vigo", "Spain", 1),
        ("Brighton", "England", 1),
        ("West Ham", "England", 1),
        ("Crystal Palace", "England", 1),
        ("Wolfsburg", "Germany", 1),
        ("RB Leipzig", "Germany", 1),
        ("Marseille", "France", 1),
        ("Lyon", "France", 1),
        ("Lille", "France", 1),
        ("PSV", "Netherlands", 3),
        ("Ajax", "Netherlands", 3),
        ("Feyenoord", "Netherlands", 3),
        ("Club Brugge", "Belgium", 3),
        ("Red Bull Salzburg", "Austria", 3),
    ],
}

# Known development clubs with strong track records
DEVELOPMENT_CLUBS = {
    "Real Sociedad", "Villarreal", "Athletic Bilbao", "Real Betis",
    "RB Leipzig", "Bayer Leverkusen",
    "Brighton", "Atalanta",
    "PSV", "Ajax", "Feyenoord",
    "Benfica", "Porto", "Sporting CP",
    "Monaco", "Lille", "Lyon",
    "Club Brugge", "Red Bull Salzburg",
}


# ---------------------------------------------------------------------------
# Pathway generators
# ---------------------------------------------------------------------------

def generate_status_quo_path(
    current_club: str,
    current_tier: int,
    current_league_tier: int,
    n_seasons: int = 3,
) -> Pathway:
    """Path A: Stay at current club."""
    steps = []
    for k in range(n_seasons):
        steps.append(PathwayStep(
            season=f"t+{k+1}",
            club=current_club,
            club_tier=current_tier,
            league_tier=current_league_tier,
            predicted_minutes_pct=0.55,  # baseline assumption
            predicted_value=0,  # filled by model
            predicted_100m_prob=0,
            description=f"Remain at {current_club}",
        ))
    return Pathway(
        name="Path A — Status Quo",
        description=f"Continue development at {current_club} without transfer",
        steps=steps,
    )


def generate_elite_jump_path(
    current_club: str,
    current_tier: int,
    n_seasons: int = 3,
) -> Pathway:
    """Path B: Immediate move to a Tier 1 club."""
    target_club = None
    for tier_clubs in CLUB_UNIVERSE.values():
        for club, league, lt in tier_clubs:
            if club != current_club:
                target_club = (club, league, lt)
                break
        if target_club:
            break

    if target_club is None:
        target_club = ("Real Madrid", "Spain", 1)

    steps = []
    # Season 1: Move to elite club — lower minutes expected
    steps.append(PathwayStep(
        season="t+1",
        club=target_club[0],
        club_tier=1,
        league_tier=target_club[2],
        predicted_minutes_pct=0.30,  # bench risk
        predicted_value=0,
        predicted_100m_prob=0,
        description=f"Transfer to {target_club[0]} — competition for minutes",
    ))
    # Season 2-3: Gradual integration
    for k in range(1, n_seasons):
        steps.append(PathwayStep(
            season=f"t+{k+1}",
            club=target_club[0],
            club_tier=1,
            league_tier=target_club[2],
            predicted_minutes_pct=min(0.30 + k * 0.15, 0.75),
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Establish at {target_club[0]}",
        ))

    return Pathway(
        name="Path B — Elite Jump",
        description=f"Immediate transfer to {target_club[0]} with bench risk initially",
        steps=steps,
        risk_penalty=0.3,
    )


def generate_stepping_stone_path(
    current_club: str,
    current_tier: int,
    n_seasons: int = 3,
) -> Pathway:
    """Path C: Move to mid-tier development club → elite club in year 2."""
    dev_club = None
    elite_club = None

    for club, _, _ in CLUB_UNIVERSE.get(3, []):
        if club != current_club and club in DEVELOPMENT_CLUBS:
            dev_club = club
            break
    if dev_club is None:
        dev_club = "Real Sociedad"

    for club, _, _ in CLUB_UNIVERSE.get(1, []):
        if club != dev_club:
            elite_club = club
            break
    if elite_club is None:
        elite_club = "Real Madrid"

    steps = [
        PathwayStep(
            season="t+1",
            club=dev_club,
            club_tier=3,
            league_tier=1,
            predicted_minutes_pct=0.75,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Move to {dev_club} for regular playing time",
        ),
        PathwayStep(
            season="t+2",
            club=dev_club,
            club_tier=3,
            league_tier=1,
            predicted_minutes_pct=0.80,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Star season at {dev_club} — build reputation",
        ),
        PathwayStep(
            season="t+3",
            club=elite_club,
            club_tier=1,
            league_tier=1,
            predicted_minutes_pct=0.55,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Transfer to {elite_club} as proven performer",
        ),
    ]

    return Pathway(
        name="Path C — Development Stepping Stone",
        description=f"{dev_club} (2 yrs) → {elite_club} — maximize development then elite move",
        steps=steps,
        risk_penalty=0.1,
    )


def generate_loan_path(
    current_club: str,
    current_tier: int,
    n_seasons: int = 3,
) -> Pathway:
    """Path D: Loan to mid-tier → return to parent club."""
    loan_club = None
    for club, _, _ in CLUB_UNIVERSE.get(3, []):
        if club != current_club:
            loan_club = club
            break
    if loan_club is None:
        loan_club = "Real Betis"

    steps = [
        PathwayStep(
            season="t+1",
            club=f"{loan_club} (Loan)",
            club_tier=3,
            league_tier=1,
            predicted_minutes_pct=0.70,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Season-long loan at {loan_club}",
        ),
        PathwayStep(
            season="t+2",
            club=current_club,
            club_tier=current_tier,
            league_tier=1,
            predicted_minutes_pct=0.40,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Return to {current_club} — compete for spot",
        ),
        PathwayStep(
            season="t+3",
            club=current_club,
            club_tier=current_tier,
            league_tier=1,
            predicted_minutes_pct=0.55,
            predicted_value=0,
            predicted_100m_prob=0,
            description=f"Establish at {current_club}",
        ),
    ]

    return Pathway(
        name="Path D — Loan Path",
        description=f"Loan to {loan_club} → Return to {current_club}",
        steps=steps,
        risk_penalty=0.2,
    )


# ---------------------------------------------------------------------------
# Pathway scoring
# ---------------------------------------------------------------------------

WEIGHTS = {
    "value": 0.40,
    "prob_100m": 0.30,
    "minutes": 0.15,
    "risk": 0.15,
}


def score_pathways(
    pathways: list[Pathway],
    value_regressor_models: list,
    classifier_models: list,
    minutes_models: list | None,
    base_features: pd.DataFrame,
    feature_columns: list[str],
    player_meta: dict[str, Any],
    value_target_col: str = "log_value_current",
) -> list[Pathway]:
    """Score each pathway using trained models."""
    from .models import ensemble_predict

    for pathway in pathways:
        step_scores = []

        for step in pathway.steps:
            # Construct a feature vector for this step
            step_features = _build_step_features(
                step, base_features, feature_columns, player_meta
            )

            if step_features is not None and len(step_features) > 0:
                X_step = pd.DataFrame([step_features], columns=feature_columns)
                # Fill any missing columns
                for col in feature_columns:
                    if col not in X_step.columns:
                        X_step[col] = 0
                X_step = X_step[feature_columns]

                # Predict value
                try:
                    pred_log_val = ensemble_predict(value_regressor_models, X_step, "regression")[0]
                    step.predicted_value = np.expm1(pred_log_val)
                except Exception:
                    step.predicted_value = player_meta.get("current_value", 5_000_000)

                # Predict €100M probability
                try:
                    step.predicted_100m_prob = ensemble_predict(
                        classifier_models, X_step, "classification"
                    )[0]
                except Exception:
                    step.predicted_100m_prob = 0.01

                # Predict minutes
                if minutes_models:
                    try:
                        step.predicted_minutes_pct = np.clip(
                            ensemble_predict(minutes_models, X_step, "regression")[0],
                            0, 1,
                        )
                    except Exception:
                        pass

            step_scores.append(step)

        # Composite score for the pathway
        final_step = pathway.steps[-1]
        avg_minutes = np.mean([s.predicted_minutes_pct for s in pathway.steps])

        pathway.composite_score = (
            WEIGHTS["value"] * _normalize_value(final_step.predicted_value)
            + WEIGHTS["prob_100m"] * final_step.predicted_100m_prob
            + WEIGHTS["minutes"] * avg_minutes
            - WEIGHTS["risk"] * pathway.risk_penalty
        )

    # Sort by composite score
    pathways.sort(key=lambda p: p.composite_score, reverse=True)
    return pathways


def _normalize_value(value: float) -> float:
    """Normalize predicted value to 0-1 scale (€100M = 1.0)."""
    return min(value / 100_000_000, 1.5)


def _build_step_features(
    step: PathwayStep,
    base_features: pd.DataFrame,
    feature_columns: list[str],
    player_meta: dict[str, Any],
) -> dict | None:
    """Construct a feature vector for a hypothetical pathway step."""
    if base_features.empty:
        return None

    # Start with the latest available feature row
    feat_dict = base_features.iloc[-1].to_dict()

    # Override environment features
    overrides = {
        "club_tier": step.club_tier,
        "league_tier": step.league_tier,
        "expected_minutes_pct": step.predicted_minutes_pct,
    }

    for key, val in overrides.items():
        if key in feat_dict:
            feat_dict[key] = val

    # Age progression
    if "age" in feat_dict and feat_dict["age"] is not None:
        season_num = int(step.season.replace("t+", "").replace("t", "0"))
        feat_dict["age"] = feat_dict["age"] + season_num
        if "age_sq" in feat_dict:
            feat_dict["age_sq"] = feat_dict["age"] ** 2

    return feat_dict


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def simulate_all_pathways(
    player_id: int,
    player_name: str,
    current_club: str,
    current_tier: int,
    current_league_tier: int,
    trajectory_df: pd.DataFrame,
    feature_columns: list[str],
    models: dict,
    n_seasons: int = 3,
) -> list[Pathway]:
    """Generate and score all pathways for a player."""

    # Get player's latest feature row
    player_data = trajectory_df[trajectory_df["player_id"] == player_id].sort_values(
        "super_season_id"
    )

    if player_data.empty:
        print(f"  [WARN] No data found for player {player_id}")
        return []

    base_features = player_data[feature_columns].fillna(0)

    player_meta = {
        "player_id": player_id,
        "player_name": player_name,
        "current_club": current_club,
        "current_value": player_data["value_current"].iloc[-1]
        if "value_current" in player_data.columns
        else 0,
        "age": player_data["age"].iloc[-1] if "age" in player_data.columns else 22,
    }

    # Generate pathways
    pathways = [
        generate_status_quo_path(current_club, current_tier, current_league_tier, n_seasons),
        generate_elite_jump_path(current_club, current_tier, n_seasons),
        generate_stepping_stone_path(current_club, current_tier, n_seasons),
        generate_loan_path(current_club, current_tier, n_seasons),
    ]

    # Score
    value_models = models.get("value_regressor", {}).get("models", {}).get("lgb", [])
    classifier_models = models.get("100m_classifier", {}).get("models", {}).get("lgb", [])
    minutes_models = models.get("minutes_predictor", {}).get("models", {}).get("lgb", [])

    if not value_models:
        value_models = models.get("value_regressor", {}).get("models", {}).get("xgb", [])
    if not classifier_models:
        classifier_models = models.get("100m_classifier", {}).get("models", {}).get("xgb", [])

    scored = score_pathways(
        pathways,
        value_models,
        classifier_models,
        minutes_models,
        base_features,
        feature_columns,
        player_meta,
    )

    print(f"\n  Pathway rankings for {player_name}:")
    for i, p in enumerate(scored, 1):
        final_val = p.steps[-1].predicted_value / 1_000_000
        final_prob = p.steps[-1].predicted_100m_prob * 100
        avg_min = np.mean([s.predicted_minutes_pct for s in p.steps]) * 100
        print(f"    #{i} {p.name}: Score={p.composite_score:.3f} | "
              f"Val=€{final_val:.1f}M | P(100M)={final_prob:.1f}% | Min={avg_min:.0f}%")

    return scored
