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
                print("PrioritizationAgent: Loaded XGBoost ML prioritizer model.")
            except Exception as e:
                print(f"Warning: Failed to load ML model: {e}. Falling back to Rule-Based.")
        else:
            print("Warning: XGBoost model not found at data/prioritization/models/. Using Rule-Based fallback.")

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
        """Any case involving bail, custody, or threat to life = immediate threat."""
        urgent_patterns = [
            r'\bbail petition\b', r'\banticipatory bail\b',
            r'\bhabeas corpus\b',
            r'\bdomestic violence\b',
            r'\bcustody\b',
            r'\bthreat to life\b', r'\blife in danger\b',
            r'\bprotection order\b',
        ]
        text_lower = text.lower()
        return int(any(re.search(p, text_lower) for p in urgent_patterns))

    def calculate_societal_impact(self, text: str) -> int:
        """Societal impact score from 1-5 based on PIL, constitutional rights, or corruption."""
        text_lower = text.lower()
        score = 1

        if re.search(r'\bpublic interest litigation\b|\b(pil)\b', text_lower):
            score += 3
        if re.search(r'\benvironmental impact\b|\benvironmental protection\b|\bpollution control\b|\becological damage\b', text_lower):
            score += 2
        if re.search(r'\bconstitutional validity\b|\bconstitutionality\b|\bconstitutional bench\b|\bviolation of fundamental rights?\b', text_lower):
            score += 2
        if re.search(r'\bcorruption prevention\b|\bprevention of corruption\b|\bcbi investigation\b|\bterrorist act\b|\bfinancial scam\b', text_lower):
            score += 2
        if re.search(r'\bclass action lawsuit\b|\bmassive protest\b|\bgang rape\b|\borganized crime syndicates?\b', text_lower):
            score += 1

        return min(score, 5)

    def compute_priority_score(
        self,
        severity: int | float | None = None,
        societal_impact: int | None = None,
        immediate_threat: int | None = None,
        case_age_days: float | None = None,
        case_type: str = "civil",
        bns_sections: str = "",
        ipc_sections: str = "",
        text: str = ""
    ) -> float:
        """
        Calculate composite priority score (0-10) using legal weights:
        - Severity of sections: 35%
        - Societal impact: 25%
        - Immediate threat: 30%
        - Recency: 10%
        """
        if immediate_threat is None:
            immediate_threat = self.check_immediate_threat(text)
            
        if societal_impact is None:
            societal_impact = self.calculate_societal_impact(text)

        if severity is None:
            case_type_clean = (case_type or "civil").lower()
            if case_type_clean == "criminal":
                all_secs = f"{ipc_sections} {bns_sections}".lower()
                if any(s in all_secs for s in ["302", "376", "396", "364a", "103"]):
                    severity = 10
                elif any(s in all_secs for s in ["304", "304b", "366a"]):
                    severity = 9
                elif any(s in all_secs for s in ["307", "395", "120b", "201", "109"]):
                    severity = 8
                elif any(s in all_secs for s in ["326", "392", "354", "498a", "85"]):
                    severity = 6
                elif any(s in all_secs for s in ["420", "406", "409", "316", "318"]):
                    severity = 5
                elif any(s in all_secs for s in ["379", "323", "427", "504", "303"]):
                    severity = 3
                else:
                    severity = 4
            else:
                severity = 3

        sev_norm = float(severity) / 10.0
        impact_norm = float(societal_impact) / 5.0
        threat_norm = float(immediate_threat)

        if self.is_valid_age(case_age_days):
            recency = max(0.0, 1.0 - (float(case_age_days) / 3650))
        else:
            recency = 0.5

        if (case_type or "civil").lower() == "civil":
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
                # Standardize casing to match encoder classes ('civil' / 'criminal')
                case_type_clean = case_type.strip().lower()
                try:
                    case_type_encoded = self.encoder.transform([case_type_clean])[0]
                except Exception:
                    case_type_encoded = 1 if case_type_clean == "criminal" else 0

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
        num_precedents: int,
        severity: float | None = None,
        immediate_threat: int | None = None,
        societal_impact: int | None = None,
        case_date: str | None = None,
        scoring_method: str | None = None
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
            
            # Determine if it's a closed/historical case or an active/pending case.
            is_closed = False
            if case_date:
                try:
                    from scripts.extract_features import parse_date
                    p_date = parse_date(case_date)
                    if p_date and p_date < datetime.now():
                        is_closed = True
                except Exception:
                    pass
            
            if years > 5:
                if is_closed:
                    reasons.append(f"Total litigation duration: {years} years")
                else:
                    reasons.append(f"Pending/Active litigation for {years} years")
            elif years > 0:
                if is_closed:
                    reasons.append(f"Litigation duration: {years} years")
                else:
                    reasons.append(f"Case pending: {years} years")

        if num_precedents > 5:
            reasons.append(f"High Precedential Complexity ({num_precedents} citations)")

        if immediate_threat and immediate_threat > 0:
            reasons.append("Immediate threat to life or liberty detected")
        if severity and severity >= 8:
            reasons.append(f"High severity offence (score {severity}/10)")
        if societal_impact and societal_impact >= 3:
            reasons.append("Significant societal impact identified")

        if scoring_method:
            reasons.append(f"Evaluated using {scoring_method}")
        elif self.model is not None:
            reasons.append("Evaluated using XGBoost Machine Learning model")
        else:
            reasons.append("Evaluated using baseline Rule-Based framework")

        return "; ".join(reasons) if reasons else "Standard priority profile"
