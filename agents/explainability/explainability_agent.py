import os
import joblib
import pandas as pd
import math
import psycopg2
import sys

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATABASE_URL

class ExplainabilityAgent:
    def __init__(self):
        self.model_path = "data/prioritization/models/prioritizer.pkl"
        self.encoder_path = "data/prioritization/models/encoders.pkl"
        self.model = None
        self.encoder = None

        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.encoder = joblib.load(self.encoder_path)
                print("ExplainabilityAgent: Loaded trained XGBoost prioritizer model.")
            except Exception as e:
                print(f"Warning: Failed to load prioritizer model in ExplainabilityAgent: {e}")
        else:
            print("Warning: Trained prioritizer model not found for ExplainabilityAgent.")

        # Compute or fetch baseline values from database
        self.baseline = self._fetch_medians_from_db()

    def _fetch_medians_from_db(self) -> dict:
        """Fetch median parameters dynamically from the PostgreSQL database."""
        medians = {
            "case_type": 0,                # civil
            "num_ipc_sections": 0,
            "num_cpc_sections": 0,
            "num_precedents": 2,
            "total_words": 1000,
            "case_age_days": 365.0,
            "max_severity_score": 3.0,
            "immediate_threat_flag": 0,
            "societal_impact_score": 1
        }
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            # Fetch actual median values from cases table
            columns_to_query = [
                "num_precedents", "total_words", "case_age_days", 
                "max_severity_score", "immediate_threat_flag", "societal_impact_score"
            ]
            for col in columns_to_query:
                cur.execute(f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) FROM cases")
                val = cur.fetchone()[0]
                if val is not None:
                    # Map immediate threat and societal impact to integers, others to float
                    if col in ["immediate_threat_flag", "societal_impact_score"]:
                        medians[col] = int(val)
                    else:
                        medians[col] = float(val)
            cur.close()
            conn.close()
            print("ExplainabilityAgent: Successfully calculated baseline medians from PostgreSQL.")
        except Exception as e:
            print(f"Warning: Could not fetch database medians in ExplainabilityAgent: {e}. Using defaults.")
        return medians

    def compute_marginal_attributions(self, case_features: dict) -> dict:
        """
        Calculates local feature attributions (SHAP-like contributions) for a given case.
        Compares case features against the baseline, computes marginal contributions,
        and normalizes them to sum exactly to (Predicted Score - Baseline Score).
        Uses the SHAP package directly if installed; falls back to marginal method otherwise.
        """
        if self.model is None or self.encoder is None:
            # Fallback to rule-based estimations if model is missing
            return self._fallback_attributions(case_features)

        # Get case type encoding
        case_type = case_features.get("case_type", "civil")
        try:
            case_type_encoded = self.encoder.transform([case_type])[0]
        except Exception:
            case_type_encoded = 1 if case_type == "criminal" else 0

        # Construct actual case features dictionary
        actual = {
            "case_type": case_type_encoded,
            "num_ipc_sections": int(case_features.get("num_ipc_sections", 0)),
            "num_cpc_sections": int(case_features.get("num_cpc_sections", 0)),
            "num_precedents": int(case_features.get("num_precedents", 0)),
            "total_words": int(case_features.get("total_words", 1000)),
            "case_age_days": float(case_features.get("case_age_days", 365.0)),
            "max_severity_score": float(case_features.get("max_severity_score", 3.0)),
            "immediate_threat_flag": int(case_features.get("immediate_threat_flag", 0)),
            "societal_impact_score": int(case_features.get("societal_impact_score", 1))
        }

        # Handle NaNs in case features
        for k in ["case_age_days", "max_severity_score"]:
            if math.isnan(actual[k]) or math.isinf(actual[k]):
                actual[k] = self.baseline[k]

        df_actual = pd.DataFrame({k: [v] for k, v in actual.items()})

        # 1. Primary path: Use native SHAP if installed
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(df_actual)
            # shap_values[0] holds the attributions array for the sample
            attributions = dict(zip(df_actual.columns, shap_values[0]))
            
            # Format and round
            formatted = {}
            for col, val in attributions.items():
                formatted[col] = round(float(val), 2)
            return formatted
        except ImportError:
            # SHAP not installed, fall back to marginal attribution
            pass
        except Exception as e:
            print(f"Warning: Primary SHAP calculation failed: {e}. Falling back to marginal attributions.")

        # 2. Secondary path: Marginal attribution calculation
        try:
            df_baseline = pd.DataFrame({k: [v] for k, v in self.baseline.items()})
            baseline_score = float(self.model.predict(df_baseline)[0])
            predicted_score = float(self.model.predict(df_actual)[0])

            diff = predicted_score - baseline_score
            raw_impacts = {}

            # Compute marginal contributions (masking one feature at a time)
            for col in self.baseline.keys():
                test_features = self.baseline.copy()
                test_features[col] = actual[col]

                df_test = pd.DataFrame({k: [v] for k, v in test_features.items()})
                test_score = float(self.model.predict(df_test)[0])

                raw_impacts[col] = test_score - baseline_score

            sum_raw = sum(raw_impacts.values())
            attributions = {}

            if abs(sum_raw) > 0.01:
                scale = diff / sum_raw
                for col, val in raw_impacts.items():
                    attributions[col] = round(val * scale, 2)
            else:
                # Sum is near-zero; preserve raw impacts directly to protect relative ordering
                for col, val in raw_impacts.items():
                    attributions[col] = round(val, 2)

            return attributions

        except Exception as e:
            print(f"Error computing explainability attributions: {e}")
            return self._fallback_attributions(case_features)

    def _fallback_attributions(self, case_features: dict) -> dict:
        """Return the exact weighted components used by compute_priority_score."""
        severity = float(case_features.get("max_severity_score", 3.0))
        societal_impact = float(case_features.get("societal_impact_score", 1))
        immediate_threat = float(case_features.get("immediate_threat_flag", 0))
        case_age_days = case_features.get("case_age_days")
        case_type = case_features.get("case_type", "civil")

        severity_norm = severity / 10.0
        if case_type == "civil":
            severity_norm = min(severity_norm, 0.4)

        impact_norm = societal_impact / 5.0
        threat_norm = float(bool(immediate_threat))
        try:
            age = float(case_age_days)
            valid_age = not math.isnan(age) and not math.isinf(age)
        except (ValueError, TypeError):
            valid_age = False
            age = 0.0
        recency = max(0.0, 1.0 - age / 3650.0) if valid_age else 0.5

        components = {
            "severity_contribution": severity_norm * 3.5,
            "societal_impact_contribution": impact_norm * 2.5,
            "immediate_threat_contribution": threat_norm * 3.0,
            "recency_contribution": recency,
        }
        total = sum(components.values())
        if total > 10.0:
            components = {key: value * (10.0 / total) for key, value in components.items()}
        rounded = {key: round(value, 2) for key, value in components.items()}
        displayed_total = round(min(total, 10.0), 2)
        rounded["recency_contribution"] = round(
            rounded["recency_contribution"] + displayed_total - sum(rounded.values()), 2
        )
        return rounded

if __name__ == "__main__":
    # Test stub
    agent = ExplainabilityAgent()
    sample_features = {
        "case_type": "criminal",
        "num_ipc_sections": 3,
        "num_cpc_sections": 0,
        "num_precedents": 8,
        "total_words": 1500,
        "case_age_days": 1825.0,
        "max_severity_score": 8.0,
        "immediate_threat_flag": 1,
        "societal_impact_score": 3
    }
    explanation = agent.compute_marginal_attributions(sample_features)
    print("Test local feature attributions:", explanation)
