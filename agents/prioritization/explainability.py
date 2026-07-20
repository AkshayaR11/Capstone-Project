import os
import joblib
import pandas as pd
import math
import shap

class ExplainabilityAgent:
    def __init__(self):
        self.model_path = "data/prioritization/models/prioritizer.pkl"
        self.encoder_path = "data/prioritization/models/encoders.pkl"
        self.model = None
        self.encoder = None
        self.explainer = None

        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.encoder = joblib.load(self.encoder_path)
                # Initialize actual SHAP TreeExplainer on the trained XGBoost model
                self.explainer = shap.TreeExplainer(self.model)
                print("ExplainabilityAgent: Loaded trained XGBoost prioritizer model and initialized SHAP TreeExplainer.")
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
        Calculates local feature attributions using the official SHAP TreeExplainer.
        If the model or explainer is missing, it falls back to a rule-based attribution estimation.
        """
        if self.model is None or self.explainer is None:
            return self._fallback_attributions(case_features)

        try:
            # 1. Get case type encoding
            case_type = case_features.get("case_type", "civil")
            try:
                case_type_encoded = self.encoder.transform([case_type])[0]
            except Exception:
                case_type_encoded = 1 if case_type == "criminal" else 0

            # 2. Construct actual case features DataFrame
            actual = {
                "case_type_enc":          case_type_encoded,
                "num_ipc_sections":        int(case_features.get("num_ipc_sections", 0)) + int(case_features.get("num_bns_sections", 0)),
                "num_cpc_sections":        int(case_features.get("num_cpc_sections", 0)),
                "num_precedents":          int(case_features.get("num_precedents", 0)),
                "total_words":             int(case_features.get("total_words", 1000)),
                "case_age_days":           float(case_features.get("case_age_days", 365.0)),
                "max_severity_score":      float(case_features.get("max_severity_score", 3.0)),
                "immediate_threat_flag":   int(case_features.get("immediate_threat_flag", 0)),
                "societal_impact_score":   int(case_features.get("societal_impact_score", 1))
            }

            # Handle NaNs in case age
            if math.isnan(actual["case_age_days"]) or math.isinf(actual["case_age_days"]):
                actual["case_age_days"] = 365.0

            df_actual = pd.DataFrame([actual])

            # 3. Compute SHAP Values using the TreeExplainer
            shap_values = self.explainer(df_actual)
            
            # 4. Format contributions dictionary
            attributions = {}
            feature_names = df_actual.columns.tolist()
            # Extract raw shap values for first sample
            values = shap_values.values[0]

            for name, val in zip(feature_names, values):
                attributions[name] = round(float(val), 2)

            return attributions

        except Exception as e:
            print(f"Error computing SHAP values: {e}. Using fallback attributions.")
            return self._fallback_attributions(case_features)

    def _fallback_attributions(self, case_features: dict) -> dict:
        """Fallback attribution rules aligned with rule-based priority scoring."""
        attributions = {
            "case_type_enc":          0.0,
            "num_ipc_sections":        0.0,
            "num_cpc_sections":        0.0,
            "num_precedents":          0.0,
            "total_words":             0.0,
            "case_age_days":           0.0,
            "max_severity_score":      0.0,
            "immediate_threat_flag":   0.0,
            "societal_impact_score":   0.0
        }
        
        case_type = case_features.get("case_type", "civil").lower()
        attributions["case_type_enc"] = 1.0 if case_type == "criminal" else -0.5

        ipc = int(case_features.get("num_ipc_sections", 0)) + int(case_features.get("num_bns_sections", 0))
        if ipc >= 3:
            attributions["num_ipc_sections"] = 1.5
        elif ipc >= 1:
            attributions["num_ipc_sections"] = 0.5

        cpc = int(case_features.get("num_cpc_sections", 0))
        if cpc >= 3:
            attributions["num_cpc_sections"] = 1.0
        elif cpc >= 1:
            attributions["num_cpc_sections"] = 0.5

        age = float(case_features.get("case_age_days", 0.0))
        if not math.isnan(age) and age > 365:
            attributions["case_age_days"] = round(min((age / 365.0) * 0.1, 1.5), 2)

        prec = int(case_features.get("num_precedents", 0))
        if prec > 5:
            attributions["num_precedents"] = 0.3

        sev = float(case_features.get("max_severity_score", 3.0))
        if not math.isnan(sev):
            attributions["max_severity_score"] = round(min((sev - 3.0) * 0.2, 2.0), 2)

        threat = int(case_features.get("immediate_threat_flag", 0))
        attributions["immediate_threat_flag"] = 2.0 if threat > 0 else 0.0

        impact = int(case_features.get("societal_impact_score", 1))
        attributions["societal_impact_score"] = round(max((impact - 1.0) * 0.4, 0.0), 2)

        return attributions
