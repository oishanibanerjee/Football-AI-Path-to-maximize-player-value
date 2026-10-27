"""
run_pathway_demo.py — Verification & Inference for Counterfactual Pathways
Selects a sample 19-21 year old player from trajectory_matrix.parquet,
runs PathwaySimulator + PathwayRanker and prints comparison table.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathway_simulator import PathwaySimulator
from pathway_ranker import PathwayRanker

MATRIX_PATH = ROOT / "data" / "processed" / "trajectory_matrix.parquet"

def pick_sample_player(df: pd.DataFrame, age_min=19, age_max=21) -> pd.Series:
    # Filter 19-21, with valid current value and valid 2yr target history (for sanity)
    cand = df[(df["age"] >= age_min) & (df["age"] <= age_max) & df["current_market_value"].notna() & df["minutes_share"].notna()].copy()
    # Prefer players with at least 1 prior season delta available and mid value
    # Sort by current value descending but not top elite already, pick interesting mid-tier
    cand = cand.sort_values("current_market_value", ascending=False)
    # Try to pick a player with validated target_future_val_2yr not NaN in original? Not required for t snapshot
    # Choose one with age 20-21, moderate value 2-10M, decent minutes
    ideal = cand[(cand["current_market_value"].between(1_000_000, 15_000_000)) & (cand["minutes_share"] > 0.3)]
    if not ideal.empty:
        pick = ideal.iloc[0]
    elif not cand.empty:
        pick = cand.iloc[0]
    else:
        # fallback any
        pick = df.iloc[0]
    return pick

def main():
    print("="*70)
    print("  RUN PATHWAY DEMO — Counterfactual Simulation for Sample Young Player")
    print("="*70)
    if not MATRIX_PATH.exists():
        print(f"[ERROR] {MATRIX_PATH} not found. Run build_trajectory_matrix first.")
        sys.exit(1)
    df = pd.read_parquet(MATRIX_PATH)
    print(f"[Load] {df.shape[0]} x {df.shape[1]} from {MATRIX_PATH}")
    print(f"  Age 19-21 count: {df['age'].between(19,21).sum()} (valid 2yr target: {df[df['age'].between(19,21)]['target_future_val_2yr'].notna().sum()})")

    sample = pick_sample_player(df, 19, 21)
    pid = int(sample["player_id"])
    pname = sample.get("player_name", f"Player_{pid}")
    age = float(sample["age"])
    pos = sample.get("position_group", sample.get("position_group_short", "Unknown"))
    club = sample.get("club_name", "Unknown")
    tier = int(sample.get("club_tier", 3))
    league_rank = int(sample.get("league_uefa_rank", sample.get("league_tier", 3)))
    mins = float(sample.get("minutes_share", 0) )
    curr_val = float(sample.get("current_market_value", sample.get("value_current", 0)))
    print(f"\n[Sample] {pname} (id={pid}) | Age {age:.0f} | {pos} | {club} (tier {tier}, league rank {league_rank}) | minutes {mins:.0%} | value €{curr_val/1e6:.2f}M")
    print(f"  Spec row snapshot: height {sample.get('height_cm', 'NA')} cm | squad_avg {sample.get('squad_avg_value', 0)/1e6:.1f}M | competition {sample.get('position_competition_count', 'NA')}")

    # Build base row dict
    base_row = sample.to_dict()

    # Init simulator (loads delta model)
    print("\n[Simulator] Initializing PathwaySimulator...")
    sim = PathwaySimulator(load_from_disk=True)

    # Simulate
    print("\n[Simulator] Constructing 4 pathways (A Stay, B Development, C Elite, D Loan)...")
    results = sim.simulate(base_row, current_value=curr_val, verbose=True)

    # Rank
    print("\n[Ranker] Scoring pathways...")
    ranker = PathwayRanker()
    scored, explanations = ranker.rank_and_explain(results, baseline_key="A")

    # Comparison table
    print("\n" + "="*90)
    print("  COMPARISON TABLE — 4 Pathways (Projected at t+2)")
    print("="*90)
    table = ranker.to_comparison_table(scored, explanations)
    # Print required columns: Pathway Name | Projected Minutes Share | Projected 2YR Value (€M) | P(€100M+) | Development Risk | Rank
    # Show full table
    cols_to_show = ["Pathway Name", "Projected Minutes Share", "Projected 2YR Value (€M)", "P(€100M+)", "Development Risk", "Rank"]
    print(table[cols_to_show].to_string(index=False))
    print("\nDetailed table:")
    print(table.to_string(index=False))

    # Print explainability drivers
    print("\n" + "="*90)
    print("  EXPLAINABILITY DRIVERS (vs Path A Stay)")
    print("="*90)
    for key in sorted(scored, key=lambda k: scored[k]["rank"]):
        res = scored[key]
        exp = explanations[key]
        print(f"\n[{res['rank']}] {res['name']} — Score {res['score']:.3f} | Value €{res['predicted_value_2yr']/1e6:.2f}M (x{res['value_growth_multiplier']:.2f}) | P {res['p100']:.2%} | mins {res['avg_minutes_share']:.0%} | risk {res['risk']:.0%}")
        print(f"  Positives:")
        for d in exp["positive_drivers"]:
            print(f"    + {d}")
        print(f"  Risks:")
        for r in exp["negative_risks"]:
            print(f"    - {r}")
        print(f"  {exp['summary']}")

    # Also show per-step detail
    print("\n" + "="*90)
    print("  PER-STEP PREDICTIONS (log-growth calibrated)")
    print("="*90)
    for key, res in scored.items():
        print(f"\n{res['name']}:")
        for step in res["steps_pred"]:
            syn = step["synthetic_row"]
            print(f"  {step['season']}: mins {syn['minutes_share']:.0%} | club_tier {syn['club_tier']} | league {syn['league_uefa_rank']} | squadAvg €{syn['squad_avg_value']/1e6:.1f}M | comp {syn['position_competition_count']} -> pred log_growth {step['pred_log_growth']:.3f} -> value €{step['predicted_value']/1e6:.2f}M P100 {step['p100']:.2%}")

    # Save demo output
    out_json = ROOT / "models" / "pathway_demo_results.json"
    import json
    # Convert to serializable
    serial = {}
    for k, v in scored.items():
        serial[k] = {
            "name": v["name"],
            "predicted_value_2yr": float(v["predicted_value_2yr"]),
            "value_growth_multiplier": float(v["value_growth_multiplier"]),
            "p100": float(v["p100"]),
            "avg_minutes_share": float(v["avg_minutes_share"]),
            "risk": float(v["risk"]),
            "score": float(v["score"]),
            "rank": int(v["rank"]),
            "explanation": explanations[k],
        }
    payload = {
        "sample_player": {"player_id": int(pid), "player_name": str(pname), "age": float(age), "position": str(pos), "club": str(club), "current_value": float(curr_val), "minutes_share": float(mins)},
        "pathways": serial,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n[Save] Demo results -> {out_json}")

    print("\nDone. Deliverables: src/pathway_simulator.py, src/pathway_ranker.py, src/run_pathway_demo.py, plus this log.")

if __name__ == "__main__":
    main()
