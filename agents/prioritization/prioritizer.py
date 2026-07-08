import re
import os
import joblib
import math
import pandas as pd
from datetime import datetime

class PrioritizationAgent:
    def __init__(self):
        self.BNS_CUTOFF = datetime(2024, 7, 1)
        self.model_path = "data/prioritization/models/prioritizer.pkl"
        self.encoder_path = "data/prioritization/models/encoders.pkl"
        self.model = None
        self.encoder = None

        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model = joblib.load(self.model_path)
                self.encoder = joblib.load(self.encoder_path)
                print("🎯 PrioritizationAgent: Loaded XGBoost ML prioritizer model.")
            except Exception as e:
                print(f"⚠️ Warning: Failed to load ML model: {e}. Falling back to Rule-Based.")
        else:
            print("⚠️ Warning: XGBoost model not found at data/prioritization/models/. Using Rule-Based fallback.")

    def is_valid_age(self, age) -> bool:
        """Helper to determine if a case age value is numeric and not NaN/Inf."""
        if age is None:
            return False
        try:
            val = float(age)
            if math.isnan(val) or math.isinf(val):
                return False
            return True
        except (ValueError, TypeError):
            return False

    def check_immediate_threat(self, text: str) -> int:
        """Any case involving bail, custody, restraint, or threat to life = immediate threat."""
        urgent_patterns = [
            r'\bbail petition\b', r'\banticipatory bail\b',
            r'\bhabeas corpus\b',
            r'\bdomestic violence\b',
            r'\bcustody\b',
            r'\binjunction\b', r'\bstay order\b', r'\bex parte\b',
            r'\bthreat to life\b', r'\blife in danger\b',
            r'\bprotection order\b',
        ]
        text_lower = text.lower()
        return int(any(re.search(p, text_lower) for p in urgent_patterns))

    def calculate_societal_impact(self, text: str) -> int:
        """Societal impact score from 1-5 based on PIL, constitutional rights, or corruption."""
        text_lower = text.lower()
        score = 1

        if re.search(r'\bpublic interest litigation\b|\bpil\b', text_lower):
            score += 3
        if re.search(r'\benvironmental\b|\bpollution\b|\bforest\b|\bwater\b', text_lower):
            score += 2
        if re.search(r'\bconstitutional\b|\bfundamental rights?\b|\barticle\s+\d+\b', text_lower):
            score += 2
        if re.search(r'\bcorruption\b|\bcbi\b|\bscam\b|\briot\b|\bterror\b', text_lower):
            score += 2
        if re.search(r'\bclass action\b|\bmass\b|\bmultiple accused\b|\bgang\b', text_lower):
            score += 1

        return min(score, 5)

    def compute_priority_score(
        self,
        severity: int,
        societal_impact: int,
        immediate_threat: int,
        case_age_days: float | None,
        case_type: str
    ) -> float:
        """
        Calculate composite priority score (0-10) using legal weights:
        - Severity of sections: 35%
        - Societal impact: 25%
        - Immediate threat: 30%
        - Recency: 10%
        """
        sev_norm = severity / 10.0
        impact_norm = societal_impact / 5.0
        threat_norm = float(immediate_threat)

        if self.is_valid_age(case_age_days):
            recency = max(0.0, 1.0 - (float(case_age_days) / 3650))
        else:
            recency = 0.5

        if case_type == "civil":
            sev_norm = min(sev_norm, 0.4)

        raw = (sev_norm * 3.5) + (impact_norm * 2.5) + (threat_norm * 3.0) + (recency * 1.0)
        return round(min(raw, 10.0), 2)

    def predict_ml_priority(
        self,
        case_type: str,
        num_ipc_sections: int,
        num_cpc_sections: int,
        num_precedents: int,
        total_words: int,
        case_age_days: float | None,
        max_severity_score: float,
        immediate_threat_flag: int,
        societal_impact_score: int
    ) -> float:
        """
        Predict priority score using trained XGBoost Regressor model if available.
        Otherwise falls back to default rule-based score.
        """
        if self.model is not None and self.encoder is not None:
            try:
                # Prepare single-row DataFrame matching training features
                try:
                    case_type_encoded = self.encoder.transform([case_type])[0]
                except Exception:
                    case_type_encoded = 1 if case_type == "criminal" else 0

                age_days = float(case_age_days) if self.is_valid_age(case_age_days) else 365.0

                feature_dict = {
                    "case_type": [case_type_encoded],
                    "num_ipc_sections": [num_ipc_sections],
                    "num_cpc_sections": [num_cpc_sections],
                    "num_precedents": [num_precedents],
                    "total_words": [total_words],
                    "case_age_days": [age_days],
                    "max_severity_score": [max_severity_score],
                    "immediate_threat_flag": [immediate_threat_flag],
                    "societal_impact_score": [societal_impact_score]
                }

                df_features = pd.DataFrame(feature_dict)
                pred_score = float(self.model.predict(df_features)[0])

                # Constrain within bounds 0-10
                return round(min(max(pred_score, 0.0), 10.0), 2)
            except Exception as e:
                print(f"⚠️ Predict ML Priority Exception: {e}. Falling back to Rule-Based.")

        # Fallback to rule-based score
        return self.compute_priority_score(
            severity=int(max_severity_score),
            societal_impact=int(societal_impact_score),
            immediate_threat=int(immediate_threat_flag),
            case_age_days=case_age_days,
            case_type=case_type
        )

    def assign_category(self, score: float | None) -> str:
        """Assign category badge based on priority score."""
        if score is None or math.isnan(score):
            return "Unknown"
        if score >= 8.0:
            return "Critical"
        elif score >= 6.0:
            return "High"
        elif score >= 4.0:
            return "Medium"
        elif score >= 2.0:
            return "Low"
        else:
            return "Very Low"

    def generate_explanation(
        self,
        case_type: str,
        num_ipc_sections: int,
        num_cpc_sections: int,
        case_age_days: float | None,
        num_precedents: int
    ) -> str:
        """Formulate a quick justification of priority score assignment."""
        reasons = []
        if case_type == 'criminal':
            reasons.append("Criminal Case")
            if num_ipc_sections > 0:
                reasons.append(f"{num_ipc_sections} Penal Sections implicated")
        else:
            reasons.append("Civil Case")
            if num_cpc_sections > 0:
                reasons.append(f"{num_cpc_sections} Civil Procedure Codes cited")

        if self.is_valid_age(case_age_days):
            years = int(float(case_age_days) / 365)
            if years > 5:
                reasons.append(f"Pending/Active litigation for {years} years")
            elif years > 0:
                reasons.append(f"Case age: {years} years")

        if num_precedents > 5:
            reasons.append(f"High Precedential Complexity ({num_precedents} citations)")

        if self.model is not None:
            reasons.append("Evaluated using XGBoost Machine Learning model")
        else:
            reasons.append("Evaluated using baseline Rule-Based framework")

        return "; ".join(reasons) if reasons else "Standard priority profile"
