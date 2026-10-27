"""
data_loader.py — Data Ingestion & Player Trajectory Assembly
Loads all raw data sources, joins them into unified player-season trajectories,
and constructs forward-looking value targets.
"""

import os
import zipfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# 1. Raw loaders
# ---------------------------------------------------------------------------

def load_laliga_zip() -> dict[str, pd.DataFrame]:
    """Extract and load all parquet files from the La Liga zip."""
    zip_path = DATA_DIR / "laliga_2022_2025.zip"
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".parquet"):
                key = Path(name).stem
                with zf.open(name) as f:
                    frames[key] = pd.read_parquet(f)
    return frames


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


def load_xlsx(sheet: Optional[str] = None) -> pd.DataFrame | dict[str, pd.DataFrame]:
    path = DATA_DIR / "real_madrid_la_fabrica_deep_analysis_expanded_2006_2026 (1).xlsx"
    if sheet:
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_excel(path, sheet_name=None)


# ---------------------------------------------------------------------------
# 2. Schema summaries
# ---------------------------------------------------------------------------

def dataset_summary(name: str, df: pd.DataFrame) -> dict:
    return {
        "name": name,
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }


def print_all_summaries(summaries: list[dict]):
    for s in summaries:
        print(f"\n{'='*70}")
        print(f"  {s['name']}  —  {s['rows']} rows × {len(s['columns'])} cols")
        print(f"{'='*70}")
        for col in s["columns"]:
            nc = s["null_counts"][col]
            nt = s["null_pct"][col]
            dt = s["dtypes"][col]
            print(f"  {col:45s} {dt:15s} nulls={nc} ({nt}%)")


# ---------------------------------------------------------------------------
# 3. Trajectory assembly
# ---------------------------------------------------------------------------

def build_player_trajectories() -> pd.DataFrame:
    """Master function: load, join, and build the analysis-ready dataset."""

    # --- Load La Liga data ---
    laliga = load_laliga_zip()
    seasons_df = laliga["seasons"]
    players_df = laliga["players"]
    careers_df = laliga["player_careers"]
    valuations_df = laliga["player_valuations"]
    stats_total = laliga["player_season_stats_total"]
    stats_p90 = laliga["player_season_stats_per90"]
    stats_avg = laliga["player_season_stats_average"]
    stats_pct = laliga["player_season_stats_percent"]
    stats_base = laliga["player_season_stats"]
    teams_df = laliga["teams"]
    super_seasons_df = laliga["super_seasons"]

    # Map season_id → super_season_id → season_label
    season_map = seasons_df.merge(
        super_seasons_df[["super_season_id", "start_year", "end_year"]],
        on="super_season_id",
        how="left",
    )
    season_map["season_label"] = (
        season_map["start_year"].astype(str) + "/" + season_map["end_year"].astype(str)
    )

    # --- Build career table with season context ---
    careers = careers_df.merge(
        season_map[["season_id", "super_season_id", "season_label"]],
        on="season_id",
        how="left",
    )

    # Standardize club id: player_careers uses team_id
    if "team_id" in careers.columns and "club_id" not in careers.columns:
        careers = careers.rename(columns={"team_id": "club_id"})

    teams_rename = teams_df[["team_id", "name", "area_id"]].rename(
        columns={"team_id": "club_id", "name": "club_name", "area_id": "country_id"}
    )
    careers = careers.merge(
        teams_rename,
        on="club_id",
        how="left",
    )

    # --- Merge player bios ---
    player_cols = ["player_id", "birth_date", "foot", "height_cm", "weight_kg", "status"]
    # Use role_name as position
    if "role_name" in players_df.columns:
        player_cols.append("role_name")
    elif "role_code" in players_df.columns:
        player_cols.append("role_code")
    # Use short_name or construct name
    if "short_name" in players_df.columns:
        players_df = players_df.copy()
        players_df["player_name"] = players_df["short_name"]
        player_cols.append("player_name")
    elif "first_name" in players_df.columns and "last_name" in players_df.columns:
        players_df = players_df.copy()
        players_df["player_name"] = players_df["first_name"] + " " + players_df["last_name"]
        player_cols.append("player_name")
    player_cols = [c for c in player_cols if c in players_df.columns]
    careers = careers.merge(
        players_df[player_cols],
        on="player_id",
        how="left",
    )

    # --- Attach valuations per player-season ---
    valuations_df["valuation_date"] = pd.to_datetime(valuations_df["valuation_date"])
    # Take the latest valuation per player-season (use super_season_id from career)
    val_latest = (
        valuations_df.merge(
            careers[["player_id", "super_season_id"]].drop_duplicates(),
            left_on="player_id",
            right_on="player_id",
            how="inner",
        )
        .sort_values("valuation_date")
        .groupby(["player_id", "super_season_id"])
        .last()
        .reset_index()[["player_id", "super_season_id", "market_value"]]
        .rename(columns={"market_value": "value_current"})
    )
    careers = careers.merge(val_latest, on=["player_id", "super_season_id"], how="left")

    # --- Merge stats via player_season_stats bridge ---
    # stats_base has: player_season_stats_id, player_id, team_id, super_season_id, ...
    # careers has: player_id, club_id (=team_id), super_season_id
    # First, ensure careers has the right team_id key
    join_key = ["player_id", "super_season_id"]
    # Use club_id as team_id for the join
    stats_link = stats_base[["player_season_stats_id", "player_id", "team_id", "super_season_id"]].copy()
    stats_link = stats_link.rename(columns={"team_id": "club_id"})
    careers = careers.merge(stats_link, on=["player_id", "club_id", "super_season_id"], how="left")

    stat_cols_total = [c for c in stats_total.columns if c != "player_season_stats_id"]
    stat_cols_p90 = [c for c in stats_p90.columns if c != "player_season_stats_id"]
    stat_cols_avg = [c for c in stats_avg.columns if c != "player_season_stats_id"]
    stat_cols_pct = [c for c in stats_pct.columns if c != "player_season_stats_id"]

    # Merge stat tables using player_season_stats_id on the full dataframe
    # (loc[mask].merge can't add new columns back)
    n_before = len(careers.columns)
    careers = careers.merge(
        stats_total[["player_season_stats_id"] + stat_cols_total],
        on="player_season_stats_id",
        how="left",
        suffixes=("", "_total"),
    )
    careers = careers.merge(
        stats_p90[["player_season_stats_id"] + stat_cols_p90],
        on="player_season_stats_id",
        how="left",
        suffixes=("", "_p90"),
    )
    careers = careers.merge(
        stats_avg[["player_season_stats_id"] + stat_cols_avg],
        on="player_season_stats_id",
        how="left",
        suffixes=("", "_avg"),
    )
    careers = careers.merge(
        stats_pct[["player_season_stats_id"] + stat_cols_pct],
        on="player_season_stats_id",
        how="left",
        suffixes=("", "_pct"),
    )
    print(f"  Stats merged: {n_before} -> {len(careers.columns)} columns")

    # --- Derive season order for forward targets ---
    careers = careers.dropna(subset=["super_season_id"]).sort_values(
        ["player_id", "super_season_id"]
    ).reset_index(drop=True)
    career_season_idx = (
        careers.groupby("player_id")["super_season_id"]
        .rank(method="dense")
        .fillna(0)
        .astype(int)
        .rename("season_idx")
    )
    careers["season_idx"] = career_season_idx

    # --- Forward value targets (V_{t+k}) ---
    for k in [1, 2, 3]:
        careers[f"value_t+{k}"] = (
            careers.groupby("player_id")["value_current"]
            .shift(-k)
        )

    # Binary target: €100M threshold
    for k in [1, 2, 3]:
        careers[f"target_100m_t+{k}"] = (careers[f"value_t+{k}"] >= 100_000_000).astype(int)

    # --- Merge team dossier risk analytics (latest per player) ---
    try:
        dossier = load_csv("team_full_dossier.csv")
        dossier_latest = (
            dossier.sort_values("as_of_date")
            .groupby("player_id")
            .last()
            .reset_index()
        )
        risk_cols = [
            "market_value_at_checkpoint", "predicted_value_1y", "predicted_value_2y", "predicted_value_3y",
            "cagr_historical", "total_return", "rolling_return_6m",
            "volatility_1y", "sharpe_ratio_1y",
            "max_drawdown_historical", "career_value_slope_historical",
        ]
        risk_cols = [c for c in risk_cols if c in dossier_latest.columns]
        careers = careers.merge(
            dossier_latest[["player_id"] + risk_cols],
            on="player_id",
            how="left",
            suffixes=("", "_dossier"),
        )
    except Exception:
        pass

    # --- Merge club financial metrics (on club_id) ---
    try:
        club_fin = load_csv("club_financial_metrics.csv")
        numeric_fin_cols = club_fin.select_dtypes(include=[np.number]).columns.tolist()
        fin_merge_cols = [c for c in ["current_total_mv", "n_players_current",
                                       "cagr_12m", "total_return_12m", "volatility_12m",
                                       "sharpe_ratio_12m", "max_drawdown_12m",
                                       "total_fees_paid_12m", "total_fees_received_12m",
                                       "net_transfer_spend_12m"]
                          if c in club_fin.columns]
        # Take latest row per club
        club_fin_latest = club_fin.sort_values("as_of_date").groupby("club_id")[fin_merge_cols].last().reset_index()
        careers = careers.merge(
            club_fin_latest,
            on="club_id",
            how="left",
            suffixes=("", "_club_fin"),
        )
    except Exception:
        pass

    return careers


# ---------------------------------------------------------------------------
# 4. Summary runner
# ---------------------------------------------------------------------------

def run_ingestion(verbose: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Run full ingestion and return trajectory DataFrame + summaries."""
    summaries = []

    laliga = load_laliga_zip()
    for key, df in laliga.items():
        summaries.append(dataset_summary(f"laliga/{key}", df))

    for csv_name in [
        "club_financial_metrics.csv",
        "club_transfers (1).csv",
        "player_valuations_full (1).csv",
        "team_full_dossier.csv",
        "SD-DB(PLAYERS) (1).csv",
    ]:
        try:
            df = load_csv(csv_name)
            summaries.append(dataset_summary(csv_name, df))
        except Exception as e:
            print(f"Warning: could not load {csv_name}: {e}")

    xlsx = load_xlsx()
    for sheet_name, df in xlsx.items():
        summaries.append(dataset_summary(f"xlsx/{sheet_name}", df))

    if verbose:
        print_all_summaries(summaries)

    print("\n\n>>> Building player trajectories...")
    trajectories = build_player_trajectories()
    print(f">>> Trajectory dataset: {trajectories.shape[0]} rows × {trajectories.shape[1]} columns")

    return trajectories, summaries


if __name__ == "__main__":
    trajectories, summaries = run_ingestion()
    out_path = DATA_DIR / "trajectories_raw.parquet"
    trajectories.to_parquet(out_path, index=False)
    print(f">>> Saved raw trajectories to {out_path}")
