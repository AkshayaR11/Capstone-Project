import re
import math
from datetime import datetime
from agents.prioritization.scoring import (
    calculate_societal_impact as _calculate_societal_impact,
    check_immediate_threat as _check_immediate_threat,
    compute_priority_score as _compute_priority_score,
)

class PrioritizationAgent:
    def __init__(self):
        self.BNS_CUTOFF = datetime(2024, 7, 1)
        # Keep the ML attributes for callers that inspect agent capability, but
        # never load/use the experimental regressor for docket priority scores.
        self.model = None
        self.encoder = None
        print("PrioritizationAgent: Using deterministic lawyer-weighted priority rules.")

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
        return _check_immediate_threat(text)

    def calculate_societal_impact(self, text: str) -> int:
        return _calculate_societal_impact(text)

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
                section_tokens = set(re.findall(r"(?<![a-z0-9])\d{1,3}[a-z]?(?![a-z0-9])", all_secs))
                severity_map = {
                    "302": 10, "376": 10, "376a": 10, "396": 10, "364a": 10,
                    "103": 10, "64": 10, "65": 10, "66": 10, "70": 10,
                    "304": 9, "304b": 9, "366a": 9, "105": 9, "80": 9,
                    "307": 8, "395": 8, "120b": 8, "201": 8, "109": 8, "61": 8, "310": 8,
                    "326": 6, "392": 6, "354": 6, "498a": 6, "85": 6, "117": 6, "118": 6, "309": 6,
                    "420": 5, "406": 5, "409": 5, "316": 5, "318": 5,
                    "379": 3, "323": 3, "427": 3, "504": 3, "303": 3, "115": 3,
                }
                severity = max((severity_map.get(section, 4) for section in section_tokens), default=3)
            else:
                severity = 3

        return _compute_priority_score(
            severity, societal_impact, immediate_threat, case_age_days, case_type
        )

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
        Backward-compatible entry point; priority policy is the deterministic rule formula.
        """
        return self.compute_priority_score(
            severity=max_severity_score,
            societal_impact=societal_impact_score,
            immediate_threat=immediate_threat_flag,
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
        else:
            return "Low"

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
