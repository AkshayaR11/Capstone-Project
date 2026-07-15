import os
import joblib
import pandas as pd
import math

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

        # Baseline reference features (median/typical values from historical cases)
        self.baseline = {
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

    def explain_priority(self, case_features: dict) -> dict:
        """
        Calculates local feature attributions (SHAP-like contributions) for a given case.
        Compares case features against the baseline, computes marginal contributions,
        and normalizes them to sum exactly to (Predicted Score - Baseline Score).
        """
        if self.model is None or self.encoder is None:
            # Fallback to rule-based estimations if model is missing
            return self._fallback_attributions(case_features)

        try:
            # 1. Prepare baseline features DataFrame
            df_baseline = pd.DataFrame({k: [v] for k, v in self.baseline.items()})
            baseline_score = float(self.model.predict(df_baseline)[0])

            # 2. Get case type encoding
            case_type = case_features.get("case_type", "civil")
            try:
                case_type_encoded = self.encoder.transform([case_type])[0]
            except Exception:
                case_type_encoded = 1 if case_type == "criminal" else 0

            # 3. Construct actual case features
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

            # Handle NaNs in case age
            if math.isnan(actual["case_age_days"]) or math.isinf(actual["case_age_days"]):
                actual["case_age_days"] = 365.0

            df_actual = pd.DataFrame({k: [v] for k, v in actual.items()})
            predicted_score = float(self.model.predict(df_actual)[0])

            diff = predicted_score - baseline_score
            raw_impacts = {}

            # 4. Compute marginal contributions (masking one feature at a time)
            for col in self.baseline.keys():
                # Start with baseline, replace ONLY column 'col' with actual value
                test_features = self.baseline.copy()
                test_features[col] = actual[col]

                df_test = pd.DataFrame({k: [v] for k, v in test_features.items()})
                test_score = float(self.model.predict(df_test)[0])

                # Marginal impact of feature 'col'
                raw_impacts[col] = test_score - baseline_score

            # 5. Normalize raw impacts to sum exactly to the prediction difference
            sum_raw = sum(raw_impacts.values())
            attributions = {}

            if abs(sum_raw) > 0.01:
                scale = diff / sum_raw
                for col, val in raw_impacts.items():
                    attributions[col] = round(val * scale, 2)
            else:
                # If sum is zero, distribute difference evenly based on raw impacts
                for col in self.baseline.keys():
                    attributions[col] = 0.0

            return attributions

        except Exception as e:
            print(f"Error computing explainability attributions: {e}")
            return self._fallback_attributions(case_features)

    def _fallback_attributions(self, case_features: dict) -> dict:
        """Fallback attribution rules aligned with rule-based priority scoring."""
        attributions = {col: 0.0 for col in self.baseline.keys()}
        
        # Estimate coarse contributions based on rule weighting
        case_type = case_features.get("case_type", "civil")
        attributions["case_type"] = 2.0 if case_type == "criminal" else 0.0

        ipc = int(case_features.get("num_ipc_sections", 0))
        if ipc >= 3:
            attributions["num_ipc_sections"] = 2.0
        elif ipc >= 1:
            attributions["num_ipc_sections"] = 1.0

        cpc = int(case_features.get("num_cpc_sections", 0))
        if cpc >= 3:
            attributions["num_cpc_sections"] = 1.5
        elif cpc >= 1:
            attributions["num_cpc_sections"] = 1.0

        age = float(case_features.get("case_age_days", 0.0))
        if not math.isnan(age) and age > 365:
            attributions["case_age_days"] = round(min(age / 3650.0, 3.0), 2)

        prec = int(case_features.get("num_precedents", 0))
        if prec > 10:
            attributions["num_precedents"] = 1.0
        elif prec > 5:
            attributions["num_precedents"] = 0.5

        sev = float(case_features.get("max_severity_score", 3.0))
        if not math.isnan(sev):
            attributions["max_severity_score"] = round(min((sev - 3.0) / 2.0, 3.5), 2)

        threat = int(case_features.get("immediate_threat_flag", 0))
        attributions["immediate_threat_flag"] = 3.0 if threat > 0 else 0.0

        impact = int(case_features.get("societal_impact_score", 1))
        attributions["societal_impact_score"] = round(max((impact - 1.0) * 0.5, 0.0), 2)

        return attributions

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
    explanation = agent.explain_priority(sample_features)
    print("Test local feature attributions:", explanation)
