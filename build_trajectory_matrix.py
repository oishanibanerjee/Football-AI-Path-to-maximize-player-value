"""
build_trajectory_matrix.py — Trajectory Matrix Assembly & Feature Engineering
Merges player performance, club/league metrics, market values across consecutive seasons
and constructs forward-looking targets for 2yr/3yr horizon.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from data_loader import build_player_trajectories
from feature_engineering import engineer_features, TOTAL_AVAILABLE_MINUTES

OUTPUT_PATH = ROOT / "data" / "processed" / "trajectory_matrix.parquet"

# Target threshold for 100M flag
THRESHOLD_100M = 100_000_000

def add_spec_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create spec-named forward targets from existing value_t+ columns."""
    # value_t+2, value_t+3 already exist from data_loader
    # Create canonical spec names
    col_map = {
        "value_t+2": "target_future_val_2yr",
        "value_t+3": "target_future_val_3yr",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = np.nan

    # Binary flags
    for k, dst in [(2, "target_is_100m_2yr"), (3, "target_is_100m_3yr")]:
        src = f"target_future_val_{k}yr"
        if src in df.columns:
            df[dst] = (df[src] >= THRESHOLD_100M).astype("Int64")
            # Where src is NaN, flag should be NA
            df.loc[df[src].isna(), dst] = pd.NA
        else:
            df[dst] = pd.NA

    # Log targets for regression convenience
    for k in [2, 3]:
        src = f"target_future_val_{k}yr"
        if src in df.columns:
            df[f"log_target_future_val_{k}yr"] = np.log1p(df[src].fillna(0))
            # keep NaN where src was NaN
            df.loc[df[src].isna(), f"log_target_future_val_{k}yr"] = np.nan

    # Ensure legacy aliases still exist
    if "target_100m_t+2" in df.columns and "target_is_100m_2yr" not in df.columns:
        df["target_is_100m_2yr"] = df["target_100m_t+2"]
    if "target_100m_t+3" in df.columns and "target_is_100m_3yr" not in df.columns:
        df["target_is_100m_3yr"] = df["target_100m_t+3"]

    return df


def add_spec_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all spec-required input features exist at season t."""

    # --- Physical & Demographic: age, position_group (FW,MF,DF,GK), height ---
    # position_group already created by engineer_features as Attacker/Midfielder/Defender/Goalkeeper
    # Map to FW/MF/DF/GK shorthand for spec compatibility, keep original too
    pos_short_map = {"Attacker": "FW", "Midfielder": "MF", "Defender": "DF", "Goalkeeper": "GK", "Unknown": "Unknown"}
    if "position_group" in df.columns:
        df["position_group_short"] = df["position_group"].map(pos_short_map)
    # height: already height_cm from player bios
    # age already computed

    # --- Playing Time: total_minutes, minutes_share, start_ratio ---
    # total_minutes: normalize name
    if "minutes_played" in df.columns:
        df["total_minutes"] = df["minutes_played"]
    elif "minutes" in df.columns:
        df["total_minutes"] = df["minutes"]
    else:
        # search any minutes col
        for c in df.columns:
            if c.lower() == "minutes_played":
                df["total_minutes"] = df[c]
                break
        if "total_minutes" not in df.columns:
            df["total_minutes"] = np.nan

    df["minutes_share"] = df["total_minutes"] / TOTAL_AVAILABLE_MINUTES
    df["minutes_share"] = df["minutes_share"].clip(0, 1)

    # start_ratio = matches_started / matches
    if "matches_started" in df.columns and "matches" in df.columns:
        df["start_ratio"] = df["matches_started"] / df["matches"].replace(0, np.nan)
        df["start_ratio"] = df["start_ratio"].clip(0, 1)
    elif "matches" in df.columns:
        df["start_ratio"] = np.nan
    else:
        # fallback from player_season_stats
        if "matches" in df.columns:
            df["start_ratio"] = df["matches_started"] / df["matches"] if "matches_started" in df.columns else np.nan
        else:
            df["start_ratio"] = np.nan

    # --- Performance Per 90 Already ensured via engineer_features: goals_p90, passes_p90 etc.
    # Ensure spec-named variants exist
    per90_alias = {
        "goals_p90": ["goals_p90", "goals_per90"],
        "xa_p90": ["xa_p90", "xg_assist_p90", "expected_assists_p90", "assists_p90"],
        "progressive_carries_per90": ["progressive_run_per90", "progressive_run_avg", "progressive_carries_per90"],
    }
    # We keep whatever engineer_features produced; just ensure progressive_carries alias
    if "progressive_carries_per90" not in df.columns:
        if "progressive_run_per90" in df.columns:
            df["progressive_carries_per90"] = df["progressive_run_per90"]
        elif "progressive_run_avg" in df.columns and "total_minutes" in df.columns:
            df["progressive_carries_per90"] = (df["progressive_run_avg"] / df["total_minutes"].replace(0, np.nan)) * 90

    # --- Performance Growth (t vs t-1): delta_minutes, delta_goals_per90, delta_xa_per90, delta_progressive_carries ---
    # engineer_features already creates delta_minutes_pct_1y etc. and per90 deltas.
    # Map to spec names
    growth_map = {
        "delta_minutes": "delta_minutes_pct_1y",  # will convert back to minutes later, but pct is more stable
        "delta_goals_per90": "delta_goals_p90_1y",
        "delta_xa_per90": "delta_xa_p90_1y",
        "delta_progressive_carries_per90": "delta_progressive_carries_per90_1y",
    }
    # If exact spec names not present, try alternatives
    # delta_minutes: absolute minutes delta
    if "delta_minutes" not in df.columns:
        if "total_minutes" in df.columns:
            df["delta_minutes"] = df.groupby("player_id")["total_minutes"].diff(1)
        elif "minutes_played" in df.columns:
            df["delta_minutes"] = df.groupby("player_id")["minutes_played"].diff(1)

    # delta_goals_per90
    if "delta_goals_per90" not in df.columns:
        # try alternatives
        for cand in ["delta_goals_p90_1y", "delta_goals_p90_1y", "delta_goals_per90_1y"]:
            if cand in df.columns:
                df["delta_goals_per90"] = df[cand]
                break
        if "delta_goals_per90" not in df.columns and "goals_p90" in df.columns:
            df["delta_goals_per90"] = df.groupby("player_id")["goals_p90"].diff(1)

    # delta_xa_per90
    if "delta_xa_per90" not in df.columns:
        for cand in ["delta_xa_p90_1y", "delta_assists_p90_1y", "delta_xg_assist_p90_1y"]:
            if cand in df.columns:
                df["delta_xa_per90"] = df[cand]
                break
        if "delta_xa_per90" not in df.columns:
            # compute from assists_p90 or xa_p90
            src = None
            for c in ["xa_p90", "assists_p90", "xg_assist_p90"]:
                if c in df.columns:
                    src = c
                    break
            if src:
                df["delta_xa_per90"] = df.groupby("player_id")[src].diff(1)

    # delta_progressive_carries_per90
    if "delta_progressive_carries_per90" not in df.columns:
        for cand in ["delta_progressive_carries_per90_1y", "delta_progressive_run_per90_1y"]:
            if cand in df.columns:
                df["delta_progressive_carries_per90"] = df[cand]
                break
        if "delta_progressive_carries_per90" not in df.columns and "progressive_carries_per90" in df.columns:
            df["delta_progressive_carries_per90"] = df.groupby("player_id")["progressive_carries_per90"].diff(1)

    # --- Environment Metrics: club_tier, league_uefa_rank, squad_avg_value, position_competition_count ---
    # club_tier already from engineer_features
    # league_uefa_rank -> map league_tier to uefa rank proxy (1 = best)
    if "league_tier" in df.columns:
        df["league_uefa_rank"] = df["league_tier"]  # 1-5, lower is better
    else:
        df["league_uefa_rank"] = 3

    # squad_avg_value -> teammate_avg_value already
    if "teammate_avg_value" in df.columns:
        df["squad_avg_value"] = df["teammate_avg_value"]
    else:
        df["squad_avg_value"] = np.nan

    # position_competition_count -> squad_competition_density
    if "squad_competition_density" in df.columns:
        df["position_competition_count"] = df["squad_competition_density"]
    else:
        df["position_competition_count"] = np.nan

    # --- Current Financials: current_market_value (log1p), contract_years_remaining ---
    if "value_current" in df.columns:
        df["current_market_value"] = df["value_current"]
        df["log_current_market_value"] = np.log1p(df["value_current"].fillna(0))
        df.loc[df["value_current"].isna(), "log_current_market_value"] = np.nan
    else:
        df["current_market_value"] = np.nan
        df["log_current_market_value"] = np.nan

    # contract_years_remaining: not in source data -> set NaN, will be imputed as median/0 later
    if "contract_years_remaining" not in df.columns:
        # Try to find contract related column
        contract_cands = [c for c in df.columns if "contract" in c.lower() or "years_remaining" in c.lower()]
        if not contract_cands:
            df["contract_years_remaining"] = np.nan

    return df


def build_trajectory_matrix() -> pd.DataFrame:
    print("=" * 70)
    print("  BUILDING TRAJECTORY MATRIX — SPEC COMPLIANT")
    print("=" * 70)
    print("\n[1/4] Loading raw trajectories via build_player_trajectories()...")
    raw = build_player_trajectories()
    print(f"  Raw shape: {raw.shape}")

    print("\n[2/4] Running engineer_features()...")
    df = engineer_features(raw)

    # Add position & age dummies to main df (as run_pipeline does)
    if "position_group" in df.columns:
        pos_dummies = pd.get_dummies(df["position_group"], prefix="pos", dtype=int)
        df = pd.concat([df, pos_dummies], axis=1)
    if "age_group" in df.columns:
        age_dummies = pd.get_dummies(df["age_group"], prefix="age", dtype=int)
        df = pd.concat([df, age_dummies], axis=1)

    print("\n[3/4] Adding spec targets (target_future_val_2yr/3yr, target_is_100m_2yr/3yr)...")
    df = add_spec_targets(df)
    # report target coverage
    for c in ["target_future_val_2yr", "target_future_val_3yr", "target_is_100m_2yr", "target_is_100m_3yr"]:
        if c in df.columns:
            nn = df[c].notna().sum()
            print(f"  {c}: {nn}/{len(df)} non-null ({nn/len(df)*100:.1f}%)")
            if "is_100m" in c:
                pos = (df[c] == 1).sum()
                print(f"    -> positives (1): {pos}")

    print("\n[4/4] Adding spec feature aliases...")
    df = add_spec_features(df)
    # Ensure key spec features exist
    spec_features = [
        "age", "position_group", "position_group_short", "height_cm",
        "total_minutes", "minutes_share", "start_ratio",
        "progressive_carries_per90",
        "delta_minutes", "delta_goals_per90", "delta_xa_per90", "delta_progressive_carries_per90",
        "club_tier", "league_uefa_rank", "squad_avg_value", "position_competition_count",
        "current_market_value", "log_current_market_value", "contract_years_remaining",
    ]
    print("  Spec feature coverage:")
    for c in spec_features:
        if c in df.columns:
            nn = df[c].notna().sum()
            print(f"    {c}: {nn}/{len(df)} ({nn/len(df)*100:.1f}%)")
        else:
            print(f"    {c}: MISSING")

    print(f"\n[Done] Final trajectory matrix: {df.shape[0]} rows x {df.shape[1]} cols")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved to {OUTPUT_PATH}")

    # Also save a lightweight summary
    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "targets": {c: int(df[c].notna().sum()) for c in ["target_future_val_2yr","target_future_val_3yr","target_is_100m_2yr","target_is_100m_3yr"] if c in df.columns},
        "positives_2yr": int((df["target_is_100m_2yr"]==1).sum()) if "target_is_100m_2yr" in df.columns else 0,
        "positives_3yr": int((df["target_is_100m_3yr"]==1).sum()) if "target_is_100m_3yr" in df.columns else 0,
    }
    print(f"  Summary: {summary}")
    return df


if __name__ == "__main__":
    build_trajectory_matrix()
