"""
pathway_ranker.py — Pathway Ranker & Explainer
Combines Expected Future Value, P(>=100M), Minutes Share, Development Risk
into composite Pathway Utility Score and generates explainability drivers.
"""
import numpy as np
import pandas as pd

# Default weights per spec: w1*Value + w2*P + w3*Minutes - w4*Risk
# Tunable; we keep 0.40,0.30,0.15,0.15 as in simulator but also allow adjustment
DEFAULT_WEIGHTS = {
    "w1_value": 0.40,
    "w2_p100": 0.30,
    "w3_minutes": 0.15,
    "w4_risk": 0.15,
}

def _norm_value(value_eur: float, cap: float = 100_000_000) -> float:
    """Normalize value to 0-1.5 scale where 100M=1.0, cap at 1.5."""
    return float(min(value_eur / cap, 1.5))

class PathwayRanker:
    def __init__(self, weights: dict = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        print(f"[PathwayRanker] Weights: {self.weights}")

    def score(self, pathway_results: dict) -> dict:
        """
        pathway_results: output of PathwaySimulator.simulate() or predict_pathway()
          dict key -> {predicted_value_2yr, p100, avg_minutes_share, risk, ...}
        Returns dict with added 'score' and rank.
        """
        scored = {}
        for key, res in pathway_results.items():
            val_norm = _norm_value(res["predicted_value_2yr"])
            p100 = res["p100"]
            mins = res["avg_minutes_share"]
            risk = res["risk"]
            w = self.weights
            score = w["w1_value"] * val_norm + w["w2_p100"] * p100 + w["w3_minutes"] * mins - w["w4_risk"] * risk
            # Also compute contributions for explainability
            scored[key] = {
                **res,
                "value_norm": val_norm,
                "score": float(score),
                "components": {
                    "value_contrib": float(w["w1_value"] * val_norm),
                    "p100_contrib": float(w["w2_p100"] * p100),
                    "minutes_contrib": float(w["w3_minutes"] * mins),
                    "risk_penalty": float(w["w4_risk"] * risk),
                },
            }
        # Rank descending by score
        sorted_keys = sorted(scored, key=lambda k: scored[k]["score"], reverse=True)
        for rank, k in enumerate(sorted_keys, start=1):
            scored[k]["rank"] = rank
        return scored

    def explain(self, scored_results: dict, baseline_key: str = "A") -> dict:
        """
        Generate explainability drivers for each pathway vs baseline (Status Quo).
        Returns dict key -> {positive_drivers: [str], negative_risks: [str], summary: str}
        """
        explanations = {}
        if baseline_key not in scored_results:
            # pick lowest score as baseline fallback
            baseline_key = min(scored_results, key=lambda k: scored_results[k]["score"])

        baseline = scored_results[baseline_key]
        base_val = baseline["predicted_value_2yr"]
        base_p = baseline["p100"]
        base_ms = baseline["avg_minutes_share"]

        for key, res in scored_results.items():
            if key == baseline_key:
                explanations[key] = {
                    "positive_drivers": ["Baseline — no change"],
                    "negative_risks": [f"Risk {res['risk']:.0%} (stay risk)"],
                    "summary": f"Baseline {res['name']} holds value €{base_val/1e6:.2f}M, P={base_p:.2%}, mins {base_ms:.0%}"
                }
                continue
            drivers = []
            risks = []
            # Value delta
            val_delta = res["predicted_value_2yr"] - base_val
            val_delta_m = val_delta / 1e6
            pct = (res["predicted_value_2yr"] / max(base_val,1) - 1) * 100
            if val_delta > 0:
                drivers.append(f"+€{val_delta_m:.1f}M ({pct:+.0f}%) vs Stay — environment/market uplift")
            else:
                risks.append(f"{val_delta_m:.1f}M ({pct:+.0f}%) vs Stay — stunts 2-year value development")

            # Minutes delta
            ms_delta = res["avg_minutes_share"] - base_ms
            if ms_delta > 0.02:
                drivers.append(f"+{ms_delta:.0%} expected playing time ({res['avg_minutes_share']:.0%} vs {base_ms:.0%}) boosts chance creation & development")
            elif ms_delta < -0.02:
                risks.append(f"{ms_delta:.0%} minutes share at {res['name'].split('—')[-1].strip()} (only {res['avg_minutes_share']:.0%} vs {base_ms:.0%} baseline)")

            # P100M delta
            p_delta = res["p100"] - base_p
            if p_delta > 0.005:
                drivers.append(f"+{p_delta:.1%} P(≥100M) uplift via growth trajectory")
            elif p_delta < -0.005:
                risks.append(f"{p_delta:.1%} P(≥100M) drop")

            # Risk explicit
            if res["risk"] > baseline["risk"] + 0.05:
                risks.append(f"High development risk {res['risk']:.0%} (bench/transfer volatility vs {baseline['risk']:.0%} Stay)")
            elif res["risk"] < baseline["risk"]:
                drivers.append(f"Low development risk {res['risk']:.0%} (stable pathway)")

            # Squad quality nuance
            # Compare squad_avg_value if available in steps
            try:
                # take t+1 synthetic row
                t1_row = res.get("steps_pred", [{}])[0].get("synthetic_row", {}) if "steps_pred" in res else {}
                base_t1 = baseline.get("steps_pred", [{}])[0].get("synthetic_row", {}) if "steps_pred" in baseline else {}
                if t1_row and base_t1:
                    sq_delta = t1_row.get("squad_avg_value", 0) - base_t1.get("squad_avg_value", 0)
                    if sq_delta > 5_000_000:
                        drivers.append(f"Squad quality +€{sq_delta/1e6:.0f}M avg teammate value — stronger platform")
                    elif sq_delta < -3_000_000:
                        risks.append(f"Squad quality -€{abs(sq_delta)/1e6:.0f}M — weaker platform")
            except Exception:
                pass

            # Score attribution
            comp = res["components"]
            summary = f"Score {res['score']:.3f} = {comp['value_contrib']:.3f}(value) + {comp['p100_contrib']:.3f}(P100) + {comp['minutes_contrib']:.3f}(mins) - {comp['risk_penalty']:.3f}(risk)"

            if not drivers:
                drivers.append("No clear positive driver vs baseline")
            if not risks:
                risks.append("No major risk flagged")

            explanations[key] = {
                "positive_drivers": drivers,
                "negative_risks": risks,
                "summary": summary,
                "value_delta_m": float(val_delta_m),
                "minutes_delta": float(ms_delta),
                "p100_delta": float(p_delta),
            }
        return explanations

    def rank_and_explain(self, pathway_results: dict, baseline_key: str = "A") -> tuple:
        scored = self.score(pathway_results)
        explanations = self.explain(scored, baseline_key=baseline_key)
        return scored, explanations

    def to_comparison_table(self, scored: dict, explanations: dict = None) -> pd.DataFrame:
        """Build DataFrame for Pathway Name | Minutes | Value | P | Risk | Rank (+ drivers if available)."""
        rows = []
        # sort by rank
        ordered = sorted(scored.items(), key=lambda kv: kv[1]["rank"])
        for key, res in ordered:
            exp = (explanations or {}).get(key, {})
            rows.append({
                "Key": key,
                "Pathway Name": res["name"],
                "Projected Minutes Share": f"{res['avg_minutes_share']:.2%}",
                "Min_t1": f"{res.get('projected_minutes_t1', res['avg_minutes_share']):.0%}",
                "Min_t2": f"{res.get('projected_minutes_t2', res['avg_minutes_share']):.0%}",
                "Projected 2YR Value (€M)": round(res["predicted_value_2yr"] / 1e6, 2),
                "Value Growth Mult": round(res["value_growth_multiplier"], 2),
                "P(€100M+)": f"{res['p100']:.2%}",
                "Development Risk": f"{res['risk']:.0%}",
                "Score": round(res["score"], 3),
                "Rank": res["rank"],
                "Top Positive Driver": (exp.get("positive_drivers", [""])[0] if exp else ""),
                "Top Risk": (exp.get("negative_risks", [""])[0] if exp else ""),
            })
        return pd.DataFrame(rows)
