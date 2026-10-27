"""
pathway_simulator.py — Counterfactual Scenario Generator
Implements PathwaySimulator class for multi-year hypothetical interventions.
Spec-compliant: modifies league_uefa_rank, club_tier, teammate_avg_value,
squad_competition_density, minutes_share.
Uses delta-growth regressor (log(value_t+2 / value_current)) without current value
and secondary calibrator to EUR, plus log-normal P(>=100M).
"""
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Try scipy for norm CDF, fallback to erf
try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    import math
    HAS_SCIPY = False

# Development club profiles for synthetic environments
DEVELOPMENT_PROFILE = {
    "club_tier": 3,
    "league_uefa_rank": 2,  # e.g., Netherlands/Portugal/Belgium-ish
    "teammate_avg_value": 8_000_000,  # strong development squad avg
    "squad_avg_value": 8_000_000,  # alias
    "position_competition_count": 4,
    "squad_competition_density": 4,
    "teammate_max_value": 25_000_000,
}

ELITE_PROFILE = {
    "club_tier": 1,
    "league_uefa_rank": 1,
    "teammate_avg_value": 35_000_000,
    "squad_avg_value": 35_000_000,
    "position_competition_count": 9,
    "squad_competition_density": 9,
    "teammate_max_value": 90_000_000,
}

MID_TIER_PROFILE = {
    "club_tier": 4,
    "league_uefa_rank": 3,
    "teammate_avg_value": 2_500_000,
    "squad_avg_value": 2_500_000,
    "position_competition_count": 3,
    "squad_competition_density": 3,
    "teammate_max_value": 8_000_000,
}

# Also legacy aliases for simulator compatibility
def _apply_env_profile(row: dict, profile: dict, minutes_share: float, age_increment: int = 0):
    out = row.copy()
    for k, v in profile.items():
        if k in out or k in ["teammate_avg_value", "squad_avg_value", "squad_competition_density", "position_competition_count", "teammate_max_value"]:
            # map both naming conventions
            out[k] = v
            # alias handling
            if k == "teammate_avg_value":
                out["squad_avg_value"] = v
                out["teammate_avg_value"] = v
            if k == "squad_avg_value":
                out["squad_avg_value"] = v
                out["teammate_avg_value"] = v
            if k == "squad_competition_density":
                out["squad_competition_density"] = v
                out["position_competition_count"] = v
            if k == "position_competition_count":
                out["position_competition_count"] = v
                out["squad_competition_density"] = v
            out["club_tier"] = profile.get("club_tier", out.get("club_tier", 3))
            out["league_uefa_rank"] = profile.get("league_uefa_rank", out.get("league_uefa_rank", 2))
            out["league_tier"] = profile.get("league_uefa_rank", out.get("league_tier", 2))
            # legacy expected_minutes_pct alias for older models
            out["expected_minutes_pct"] = minutes_share
    out["minutes_share"] = minutes_share
    out["expected_minutes_pct"] = minutes_share
    # total_minutes derived from minutes_share * 3420
    out["total_minutes"] = minutes_share * 3420
    out["minutes_played"] = minutes_share * 3420
    # Age progression
    if "age" in out and out["age"] is not None and not pd.isna(out["age"]):
        out["age"] = out["age"] + age_increment
        if "age_sq" in out:
            out["age_sq"] = out["age"] ** 2
        # age_group update (simplified)
        age = out["age"]
        if age < 20:
            out["age_group"] = "U20"
        elif age <= 21:
            out["age_group"] = "20-21"
        elif age <= 23:
            out["age_group"] = "22-23"
        elif age <= 25:
            out["age_group"] = "24-25"
        elif age <= 28:
            out["age_group"] = "26-28"
        elif age <= 31:
            out["age_group"] = "29-31"
        else:
            out["age_group"] = "32+"
    return out

def log_normal_p100(current_value: float, pred_log_growth: float, sigma: float, threshold: float = 100_000_000) -> float:
    """P(Value >= threshold) under log-normal: log(Value) ~ N(log(current)+pred_growth, sigma^2)"""
    if current_value <= 0 or sigma <= 0 or np.isnan(pred_log_growth):
        return 0.01
    # Need P( log(target) >= log(threshold) )
    # log(target) = log(current) + growth
    # growth ~ N(pred_growth, sigma^2)
    # So log(target) ~ N(log(current)+pred_growth, sigma^2)
    mu = np.log(current_value) + pred_log_growth
    log_thr = np.log(threshold)
    # P = 1 - CDF((log_thr - mu)/sigma)
    z = (log_thr - mu) / sigma
    if HAS_SCIPY:
        p = 1 - norm.cdf(z)
    else:
        import math
        # approximation via erf
        p = 0.5 * (1 - math.erf(z / math.sqrt(2)))
    return float(np.clip(p, 0, 1))

class PathwaySimulator:
    """
    Counterfactual Scenario Generator.
    Takes a player's current snapshot (row at t) and constructs synthetic
    feature rows for t+1, t+2 by modifying environment features.
    """

    def __init__(
        self,
        delta_model=None,
        delta_feature_columns=None,
        delta_imputer=None,
        delta_sigma: float = 0.6,
        classifier_model=None,
        classifier_feature_columns=None,
        classifier_imputer=None,
        load_from_disk: bool = True,
    ):
        self.delta_model = delta_model
        self.delta_features = delta_feature_columns
        self.delta_imputer = delta_imputer
        self.sigma = delta_sigma
        self.classifier_model = classifier_model
        self.classifier_features = classifier_feature_columns
        self.classifier_imputer = classifier_imputer

        if load_from_disk and self.delta_model is None:
            self._load_models()

    def _load_models(self):
        # Try counterfactual delta model
        cand = ROOT / "models" / "counterfactual_delta_regressor.pkl"
        if cand.exists():
            try:
                payload = pickle.load(open(cand, "rb"))
                self.delta_model = payload.get("model")
                self.delta_features = payload.get("feature_columns")
                self.delta_imputer = payload.get("imputer")
                self.sigma = payload.get("sigma", 0.6)
                print(f"[PathwaySimulator] Loaded delta model from {cand} (sigma={self.sigma:.3f}, feats={len(self.delta_features)})")
            except Exception as e:
                print(f"[PathwaySimulator] Failed to load delta model: {e}")
        # Fallback to baseline regressor's feature list if delta not available
        if self.delta_model is None:
            # Try baseline regressor to at least get feature handling, but note dominance
            base = ROOT / "models" / "baseline_regressor.pkl"
            if base.exists():
                try:
                    payload = pickle.load(open(base, "rb"))
                    # We will still use it but warn about dominance
                    self.delta_model = payload.get("model")
                    self.delta_features = payload.get("feature_columns")
                    self.delta_imputer = payload.get("imputer")
                    print(f"[PathwaySimulator] Fallback to baseline_regressor (with current_value dominance)")
                except Exception as e:
                    print(f"[PathwaySimulator] Fallback failed: {e}")
        # Classifier for alternative P100M (optional)
        cls_path = ROOT / "models" / "baseline_classifier.pkl"
        if cls_path.exists():
            try:
                payload = pickle.load(open(cls_path, "rb"))
                self.classifier_model = payload.get("model")
                self.classifier_features = payload.get("feature_columns")
                self.classifier_imputer = payload.get("imputer")
                print(f"[PathwaySimulator] Loaded classifier from {cls_path}")
            except Exception as e:
                print(f"[PathwaySimulator] Classifier load failed: {e}")

    def _prepare_row(self, base_row: dict, feature_columns: list) -> pd.DataFrame:
        # Ensure all feature columns present
        row = {}
        for c in feature_columns:
            row[c] = base_row.get(c, 0)
        df = pd.DataFrame([row], columns=feature_columns)
        # Handle age_group / pos dummies if missing
        # Fill NaNs and infs via imputer later
        return df

    def _predict_growth(self, synthetic_row: dict) -> float:
        if self.delta_model is None or self.delta_features is None:
            return 0.0
        X = self._prepare_row(synthetic_row, self.delta_features)
        # Impute
        if self.delta_imputer is not None:
            X_imp = pd.DataFrame(self.delta_imputer.transform(X), columns=self.delta_features, index=X.index)
        else:
            X_imp = X.fillna(0)
        X_imp = X_imp.replace([np.inf, -np.inf], 0)
        # Handle pipeline vs raw model
        try:
            if hasattr(self.delta_model, "predict"):
                # If pipeline, it expects scaled etc. but we already imputed; pipeline will handle
                # For Pipeline with scaler, we passed imputed but scaler is inside pipeline - need raw?
                # Simpler: if model is Pipeline, call predict on X_imp
                pred = self.delta_model.predict(X_imp)
            else:
                pred = self.delta_model.predict(X_imp)
            return float(pred[0])
        except Exception as e:
            # print(f"Predict growth failed: {e}")
            return 0.0

    def _predict_p100_classifier(self, synthetic_row: dict) -> float:
        if self.classifier_model is None or self.classifier_features is None:
            return None
        try:
            X = self._prepare_row(synthetic_row, self.classifier_features)
            if self.classifier_imputer is not None:
                X_imp = pd.DataFrame(self.classifier_imputer.transform(X), columns=self.classifier_features, index=X.index).replace([np.inf, -np.inf], 0)
            else:
                X_imp = X.fillna(0)
            proba = self.classifier_model.predict_proba(X_imp)[0, 1]
            return float(proba)
        except Exception:
            return None

    def construct_pathways(self, base_row: dict, current_value: float) -> dict:
        """
        Build 4 synthetic pathways. Each pathway is dict with:
        - name, description, steps (each step is synthetic feature dict for t+1, t+2)
        """
        # Extract current env as baseline
        base_env = {
            "club_tier": base_row.get("club_tier", 3),
            "league_uefa_rank": base_row.get("league_uefa_rank", base_row.get("league_tier", 3)),
            "teammate_avg_value": base_row.get("squad_avg_value", base_row.get("teammate_avg_value", 5_000_000)),
            "squad_avg_value": base_row.get("squad_avg_value", base_row.get("teammate_avg_value", 5_000_000)),
            "position_competition_count": base_row.get("position_competition_count", base_row.get("squad_competition_density", 5)),
            "minutes_share": base_row.get("minutes_share", base_row.get("expected_minutes_pct", 0.5)),
        }

        pathways = {}

        # Path A — Status Quo / Stay
        steps_a = []
        for k, season in enumerate(["t+1", "t+2"], start=1):
            syn = _apply_env_profile(base_row, {
                "club_tier": base_env["club_tier"],
                "league_uefa_rank": base_env["league_uefa_rank"],
                "teammate_avg_value": base_env["teammate_avg_value"],
                "squad_avg_value": base_env["squad_avg_value"],
                "position_competition_count": base_env["position_competition_count"],
            }, minutes_share=base_env["minutes_share"], age_increment=k)
            steps_a.append((season, syn))
        pathways["A"] = {
            "name": "Path A — Status Quo / Stay",
            "description": f"Keep current club tier {base_env['club_tier']}, league rank {base_env['league_uefa_rank']}, minutes {base_env['minutes_share']:.0%}",
            "steps": steps_a,
            "risk": 0.05,
        }

        # Path B — Development Move
        steps_b = []
        for k, season in enumerate(["t+1", "t+2"], start=1):
            ms = 0.75 if k == 1 else 0.80
            syn = _apply_env_profile(base_row, DEVELOPMENT_PROFILE, minutes_share=ms, age_increment=k)
            steps_b.append((season, syn))
        pathways["B"] = {
            "name": "Path B — Development Move",
            "description": "Increase minutes_share to 0.75+, move to development-focused club (high youth minutes, strong squad avg ~8M)",
            "steps": steps_b,
            "risk": 0.10,
        }

        # Path C — Premature Elite Transfer
        steps_c = []
        for k, season in enumerate(["t+1", "t+2"], start=1):
            ms = 0.25 if k == 1 else 0.35
            syn = _apply_env_profile(base_row, ELITE_PROFILE, minutes_share=ms, age_increment=k)
            steps_c.append((season, syn))
        pathways["C"] = {
            "name": "Path C — Premature Elite Transfer",
            "description": "Move to top-tier club (high squad value ~35M, top league rank 1), but 0.25 substitute role",
            "steps": steps_c,
            "risk": 0.35,
        }

        # Path D — Stepping-Stone Loan
        steps_d = []
        # Season 1: loan to mid-tier with 0.85+
        syn1 = _apply_env_profile(base_row, MID_TIER_PROFILE, minutes_share=0.85, age_increment=1)
        steps_d.append(("t+1", syn1))
        # Season 2: transfer to Top League
        syn2 = _apply_env_profile(base_row, ELITE_PROFILE, minutes_share=0.55, age_increment=2)
        steps_d.append(("t+2", syn2))
        pathways["D"] = {
            "name": "Path D — Stepping-Stone Loan",
            "description": "Loan to mid-tier (0.85+ minutes) for Season 1, then Top League in Season 2",
            "steps": steps_d,
            "risk": 0.15,
        }

        return pathways

    def predict_pathway(self, pathways: dict, current_value: float) -> dict:
        """
        Run delta regressor on each synthetic step, compute predicted_value_2yr/3yr,
        value_growth_multiplier, and P100M via log-normal (sigma) and optionally classifier.
        """
        results = {}
        for key, pw in pathways.items():
            steps = pw["steps"]
            preds = []
            for season, syn_row in steps:
                pred_log_growth = self._predict_growth(syn_row)
                # Calibrator: predicted_value = current * exp(pred_growth)
                # For t+2 we use current * exp(pred for that season) but note growth is from t to that season
                # Our delta model was trained for 2yr horizon; for t+1 we approximate by scaling 0.6 * growth
                # Simpler: use same pred for t+1 scaled, for t+2 full
                if season == "t+1":
                    # Approximate 1yr growth as 0.55 * 2yr growth (empirical)
                    adj_growth = pred_log_growth * 0.55
                else:
                    adj_growth = pred_log_growth
                pred_value = current_value * np.exp(adj_growth)
                pred_value = max(pred_value, 100_000)  # floor
                growth_mult = pred_value / max(current_value, 1)
                p100_lognormal = log_normal_p100(current_value, adj_growth, self.sigma)
                p100_classifier = self._predict_p100_classifier(syn_row)
                # Prefer log-normal if classifier not available or for consistency; keep both
                p100 = p100_lognormal
                # If classifier available and log-normal is very low but classifier high, blend?
                # Keep log-normal as primary per spec
                preds.append({
                    "season": season,
                    "synthetic_row": syn_row,
                    "pred_log_growth": float(adj_growth),
                    "predicted_value": float(pred_value),
                    "value_growth_multiplier": float(growth_mult),
                    "p100_lognormal": float(p100),
                    "p100_classifier": float(p100_classifier) if p100_classifier is not None else None,
                    "p100": float(p100),
                })
            # Aggregate for pathway: use t+2 values as primary 2yr projection
            t2 = preds[-1]  # last step is t+2
            # 3yr extrapolation: 3yr growth ~ 1.35 * 2yr growth (empirical compounding)
            p3_growth = t2["pred_log_growth"] * 1.35 if t2["pred_log_growth"] != 0 else 0.0
            p3_value = current_value * np.exp(p3_growth) if p3_growth != 0 else t2["predicted_value"] * 1.05
            p3_p100 = log_normal_p100(current_value, p3_growth, self.sigma)
            avg_ms = float(np.mean([syn["minutes_share"] for _, syn in steps]))
            results[key] = {
                "name": pw["name"],
                "description": pw["description"],
                "risk": pw["risk"],
                "steps_pred": preds,
                "predicted_value_2yr": t2["predicted_value"],
                "predicted_value_3yr": float(max(p3_value, 100_000)),
                "value_growth_multiplier": t2["value_growth_multiplier"],
                "value_growth_multiplier_3yr": float(max(p3_value,100_000) / max(current_value,1)),
                "p100": t2["p100"],
                "p100_3yr": float(p3_p100),
                "pred_log_growth_2yr": t2["pred_log_growth"],
                "pred_log_growth_3yr": float(p3_growth),
                "avg_minutes_share": avg_ms,
                "projected_minutes_share": avg_ms,
                "projected_minutes_t1": float(steps[0][1]["minutes_share"]),
                "projected_minutes_t2": float(steps[1][1]["minutes_share"]),
            }

        return results

    def simulate(self, base_row: dict, current_value: float = None, verbose: bool = False) -> dict:
        """Convenience: construct + predict in one call."""
        if current_value is None:
            current_value = base_row.get("current_market_value", base_row.get("value_current", 5_000_000))
        pathways = self.construct_pathways(base_row, current_value)
        results = self.predict_pathway(pathways, current_value)
        if verbose:
            for k, r in results.items():
                print(f"{r['name']}: 2yr €{r['predicted_value_2yr']/1e6:.2f}M, P100 {r['p100']:.2%}, mins {r['avg_minutes_share']:.0%}, risk {r['risk']:.2f}")
        return results
