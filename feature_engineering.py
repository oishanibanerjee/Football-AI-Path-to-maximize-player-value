"""
feature_engineering.py — Advanced Feature Engineering & Normalization
Implements per-90 normalization, positional grouping, environment/opportunity
indexing, trajectory momentum, and produces the ML-ready feature matrix.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITION_MAP = {
    # Goalkeepers
    "Goalkeeper": "Goalkeeper", "GK": "Goalkeeper", "goalkeeper": "Goalkeeper",
    # Defenders
    "Defender": "Defender", "Centre-Back": "Defender", "Right-Back": "Defender",
    "Left-Back": "Defender", "Center-Back": "Defender",
    "defender": "Defender", "DF": "Defender",
    # Midfielders
    "Midfielder": "Midfielder", "Central Midfield": "Midfielder",
    "Defensive Midfield": "Midfielder", "Attacking Midfield": "Midfielder",
    "Right Midfield": "Midfielder", "Left Midfield": "Midfielder",
    "midfielder": "Midfielder", "MF": "Midfielder",
    # Attackers
    "Attacker": "Attacker", "Centre-Forward": "Attacker", "Second Striker": "Attacker",
    "Right Winger": "Attacker", "Left Winger": "Attacker",
    "attacker": "Attacker", "FW": "Attacker",
}

TOTAL_AVAILABLE_MINUTES = 3420  # 38 matches × 90

# Tier definitions
ELITE_LEAGUES = {"England", "Spain", "Germany", "France", "Italy"}
TOP_TIER_CLUBS = {
    "Real Madrid", "FC Barcelona", "Atlético Madrid",
    "Manchester City", "Liverpool", "Arsenal", "Chelsea", "Manchester United",
    "Bayern Munich", "Borussia Dortmund",
    "Paris Saint-Germain", "Olympique Marseille",
    "Juventus", "AC Milan", "Inter Milan", "Napoli",
}
MID_TIER_DEVELOPMENT_LEAGUES = {
    "Netherlands", "Portugal", "Belgium", "Austria", "Croatia",
    "Scotland", "Denmark", "Sweden", "Norway", "Switzerland",
}


# ---------------------------------------------------------------------------
# 1. Positional grouping
# ---------------------------------------------------------------------------

def assign_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw position strings to [Goalkeeper, Defender, Midfielder, Attacker]."""
    # Map raw position strings to [Goalkeeper, Defender, Midfielder, Attacker]
    pos_col = None
    for candidate in ["position", "position_group", "role", "role_name", "role_code"]:
        if candidate in df.columns:
            pos_col = candidate
            break
    if pos_col:
        df["position_group"] = df[pos_col].map(POSITION_MAP).fillna("Unknown")
    else:
        df["position_group"] = "Unknown"
    return df


# ---------------------------------------------------------------------------
# 2. Per-90 normalization (ensure p90 columns exist)
# ---------------------------------------------------------------------------

def ensure_per90_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create per-90 columns for key metrics if only totals exist."""
    minutes_col = _find_minutes_col(df)
    if minutes_col is None:
        return df

    per90_targets = {
        "goals": "goals_p90",
        "assists": "assists_p90",
        "total_passes": "passes_p90",
        "key_passes": "key_passes_p90",
        "total_scoring_att": "shots_p90",
        "saves": "saves_p90",
        "interceptions": "interceptions_p90",
        "tackles": "tackles_p90",
        "blocks": "blocks_p90",
        "aerials_won": "aerials_won_p90",
        "dribbles": "dribbles_p90",
        "fouls_committed": "fouls_p90",
        "yellow_cards": "yellow_cards_p90",
        "red_cards": "red_cards_p90",
        "minutes_played": "minutes_played",
        "total_xg": "xg_p90",
        "expected_assists": "xa_p90",
    }

    for raw_col, p90_col in per90_targets.items():
        # Check if p90 column already exists
        if p90_col in df.columns:
            continue
        # Find raw column (may have suffix from merges)
        candidates = [raw_col] + [c for c in df.columns if c.startswith(raw_col)]
        found = None
        for c in candidates:
            if c in df.columns and c != p90_col:
                found = c
                break
        if found is not None:
            minutes = df[minutes_col].replace(0, np.nan)
            df[p90_col] = (df[found] / minutes) * 90
    return df


def _find_minutes_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["minutes_played", "minutes", "mins", "playing_time"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "minute" in c.lower() and "played" in c.lower():
            return c
    return None


# ---------------------------------------------------------------------------
# 3. Environment & Opportunity Index
# ---------------------------------------------------------------------------

def compute_environment_index(df: pd.DataFrame) -> pd.DataFrame:
    """Derive Expected_Minutes_Pct, Squad_Competition_Density, Club_Tier, League_Tier."""
    minutes_col = _find_minutes_col(df)

    # Expected Minutes %
    if minutes_col and minutes_col in df.columns:
        df["expected_minutes_pct"] = df[minutes_col] / TOTAL_AVAILABLE_MINUTES
        df["expected_minutes_pct"] = df["expected_minutes_pct"].clip(0, 1)
    else:
        df["expected_minutes_pct"] = np.nan

    # Squad Competition Density: count of same-position-group teammates at same club+season
    if "club_name" in df.columns and "super_season_id" in df.columns and "position_group" in df.columns:
        group_key = ["club_name", "super_season_id", "position_group"]
        density = df.groupby(group_key).size().rename("squad_competition_density")
        df = df.merge(density, on=group_key, how="left")

        # Average value of same-position-group teammates (proxy for competition quality)
        if "value_current" in df.columns:
            teammate_val = (
                df.groupby(group_key)["value_current"]
                .agg(["mean", "max"])
                .rename(columns={"mean": "teammate_avg_value", "max": "teammate_max_value"})
            )
            df = df.merge(teammate_val, on=group_key, how="left")
    else:
        df["squad_competition_density"] = np.nan
        df["teammate_avg_value"] = np.nan
        df["teammate_max_value"] = np.nan

    # Club Tier (1-5)
    if "club_name" in df.columns:
        df["club_tier"] = df["club_name"].apply(_club_tier)
    else:
        df["club_tier"] = 3

    # League Tier (1-5)
    if "country_id" in df.columns:
        df["league_tier"] = df["country_id"].apply(_league_tier)
    else:
        df["league_tier"] = 3

    return df


def _club_tier(club_name: str) -> int:
    if club_name in TOP_TIER_CLUBS:
        return 1
    # Big-5 adjacent clubs
    big_clubs = {"Sevilla", "Villarreal", "Real Sociedad", "Athletic Bilbao",
                 "Tottenham", "Leicester", "West Ham", "Newcastle United",
                 "Roma", "Lazio", "Atalanta", "Fiorentina",
                 "Lyon", "Monaco", "Nice", "Sporting CP", "Benfica", "Porto"}
    if club_name in big_clubs:
        return 2
    mid_clubs = {"Celta Vigo", "Real Betis", "Valencia", "Getafe", "Osasuna",
                 "Girona", "Mallorca", "Rayo Vallecano",
                 "Brighton", "Aston Villa", "Crystal Palace", "Fulham",
                 "Torino", "Udinese", "Bologna", "Genoa",
                 "Lens", "Rennes", "Strasbourg", "Braga",
                 "AZ Alkmaar", "Feyenoord", "PSV", "Ajax"}
    if club_name in mid_clubs:
        return 3
    lower_clubs = {"Espanyol", "Alavés", "Cádiz", "Granada", "Valladolid",
                   "Elche", "Levante", "Huesca", "Eibar", "Leganes",
                   "Burnley", "Watford", "Norwich", "Bournemouth",
                   "Empoli", "Sassuolo", "Lecce", "Frosinone",
                   "Montpellier", "Toulouse", "Lorient", "Metz"}
    if club_name in lower_clubs:
        return 4
    return 5


def _league_tier(country_id) -> int:
    if isinstance(country_id, str):
        country = country_id
    else:
        country = str(country_id)
    if country in ("Spain", "108", "795"):
        return 1
    if country in ELITE_LEAGUES or country in ("189", "201", "101", "50", "75"):
        return 1
    if country in MID_TIER_DEVELOPMENT_LEAGUES or country in ("157", "176", "36", "39", "44"):
        return 3
    return 4


# ---------------------------------------------------------------------------
# 4. Trajectory momentum (rolling deltas)
# ---------------------------------------------------------------------------

def compute_trajectory_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 1-year and 2-year rolling momentum metrics."""
    df = df.sort_values(["player_id", "super_season_id"]).reset_index(drop=True)

    momentum_cols = {
        "value_current": "value",
        "expected_minutes_pct": "minutes_pct",
    }

    # Add per-90 momentum for available metrics
    for col in df.columns:
        if col.endswith("_p90") and not col.startswith("teammate"):
            short = col.replace("_p90", "")
            momentum_cols[col] = short + "_p90"

    for col, prefix in momentum_cols.items():
        if col not in df.columns:
            continue

        # 1-year delta
        df[f"delta_{prefix}_1y"] = (
            df.groupby("player_id")[col].diff(1)
        )
        # 2-year delta
        df[f"delta_{prefix}_2y"] = (
            df.groupby("player_id")[col].diff(2)
        )

        # 1-year pct change
        df[f"pctchange_{prefix}_1y"] = (
            df.groupby("player_id")[col].pct_change(1)
        )

    # Absolute momentum flags
    if "delta_value_1y" in df.columns:
        df["value_momentum_accelerating"] = (
            df.groupby("player_id")["delta_value_1y"].diff(1) > 0
        ).astype(int)
        df["value_momentum_accelerating"] = df["value_momentum_accelerating"].fillna(0)

    return df


# ---------------------------------------------------------------------------
# 5. Age features
# ---------------------------------------------------------------------------

def compute_age_features(df: pd.DataFrame) -> pd.DataFrame:
    if "birth_date" in df.columns:
        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
        if "super_season_id" in df.columns:
            # Approximate season end year
            season_year = df["super_season_id"].astype(str).str[:4].astype(float) + 1
            df["age"] = season_year - df["birth_date"].dt.year
        else:
            df["age"] = 2026 - df["birth_date"].dt.year

        # Age bucket
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 19, 21, 23, 25, 28, 31, 40],
            labels=["U20", "20-21", "22-23", "24-25", "26-28", "29-31", "32+"],
        )

        # Age squared (quadratic term)
        df["age_sq"] = df["age"] ** 2
    else:
        df["age"] = np.nan
        df["age_group"] = "Unknown"
        df["age_sq"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 6. Feature selection and matrix assembly
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns suitable for ML."""
    exclude = {
        "player_id", "player_name", "club_name", "country_id",
        "birth_date", "foot", "status", "metric_calculation_date",
        "as_of_date", "season_label", "role",
        # Targets
        "value_current", "value_t+1", "value_t+2", "value_t+3",
        "target_100m_t+1", "target_100m_t+2", "target_100m_t+3",
        "market_value", "market_value_at_checkpoint",
        # IDs
        "player_season_stats_id", "season_id", "super_season_id",
        "club_id", "team_id", "prediction_season", "prediction_calculation_date",
        # Dossier prediction targets
        "predicted_value_1y", "predicted_value_2y", "predicted_value_3y",
        "fair_price_1y", "fair_price_2y", "fair_price_3y",
        "accuracy_1y", "accuracy_2y", "accuracy_3y",
        "position_group",  # will be one-hot encoded separately
        "age_group",
    }
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in exclude]
    return features


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "value_t+2",
    min_seasons: int = 2,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the final feature matrix X, target y, and feature list.
    Filters to players with at least `min_seasons` of data and non-null targets.
    """
    # Ensure position dummies
    if "position_group" in df.columns:
        pos_dummies = pd.get_dummies(df["position_group"], prefix="pos", dtype=int)
        df = pd.concat([df, pos_dummies], axis=1)

    # Ensure age_group dummies
    if "age_group" in df.columns:
        age_dummies = pd.get_dummies(df["age_group"], prefix="age", dtype=int)
        df = pd.concat([df, age_dummies], axis=1)

    # Filter to players with enough history
    player_counts = df.groupby("player_id").size()
    valid_players = player_counts[player_counts >= min_seasons].index
    df = df[df["player_id"].isin(valid_players)].copy()

    # Drop rows with null target
    df = df.dropna(subset=[target_col]).copy()

    features = get_feature_columns(df)

    # Add position and age dummies to features
    dummy_cols = [c for c in df.columns if c.startswith("pos_") or c.startswith("age_")]
    features = features + dummy_cols
    features = list(dict.fromkeys(features))  # deduplicate preserving order

    # Only keep features that exist, are numeric, and have some variance
    features = [f for f in features if f in df.columns]
    # Exclude non-numeric columns
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    variances = df[numeric_features].var()
    features = variances[variances > 0].index.tolist()

    X = df[features].copy()
    y = df[target_col].copy()

    return X, y, features


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on the raw trajectory DataFrame."""
    print("[FE] Assigning position groups...")
    df = assign_position_group(df)

    print("[FE] Ensuring per-90 columns...")
    df = ensure_per90_columns(df)

    print("[FE] Computing age features...")
    df = compute_age_features(df)

    print("[FE] Computing environment & opportunity index...")
    df = compute_environment_index(df)

    print("[FE] Computing trajectory momentum...")
    df = compute_trajectory_momentum(df)

    # Log-transform the current value for modelling
    if "value_current" in df.columns:
        df["log_value_current"] = np.log1p(df["value_current"])

    for k in [1, 2, 3]:
        col = f"value_t+{k}"
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    print(f"[FE] Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    from data_loader import build_player_trajectories

    raw = build_player_trajectories()
    processed = engineer_features(raw)
    processed.to_parquet("data/trajectories_engineered.parquet", index=False)
    print("Saved engineered features.")
