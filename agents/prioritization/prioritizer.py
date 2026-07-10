import re
import os
import math
import joblib
from datetime import datetime

# ============================================================
# JUDICIAL CASE PRIORITIZATION ENGINE
# Source: Legal Expert Meeting (Phase 2)
# Approved by: Dr. Saritha P
# Formula: FINAL PRIORITY = min(10, BASE × AGE_MULTIPLIER + URGENCY_BOOST)
# ============================================================

BASE_PRIORITY_TABLE = {
    "criminal": {
        "CRITICAL": 8.5,  # Murder (101-103), Sexual Assault (63-67)
        "HIGH":     7.5,  # Violent crimes (108, 111, 328, 384, 397), Domestic Violence
        "MEDIUM":   5.0,  # Fraud, cheating (406, 420, 337, 304, 358), Economic crimes
        "LOW":      3.0,  # Theft (379, 427, 447, 336), minor offenses
    },
    "civil": {
        "CRITICAL": 8.0,  # Family + minors, Maintenance/Alimony
        "HIGH":     7.0,  # Property/Commercial > 50L, Inheritance
        "MEDIUM":   5.5,  # Property 5L-50L, Contracts 1L-10L
        "LOW":      5.0,  # Small claims < 1L, routine matters
    }
}

# BNS Section → Severity mapping (criminal)
BNS_SEVERITY_MAP = {
    # CRITICAL - Murder
    "101": "CRITICAL", "102": "CRITICAL", "103": "CRITICAL",
    # CRITICAL - Sexual Assault
    "63": "CRITICAL", "64": "CRITICAL", "65": "CRITICAL", "66": "CRITICAL", "67": "CRITICAL",
    # HIGH - Violent crimes / Dacoity
    "108": "HIGH", "111": "HIGH", "328": "HIGH", "384": "HIGH", "397": "HIGH",
    # MEDIUM - Fraud, cheating
    "406": "MEDIUM", "420": "MEDIUM", "337": "MEDIUM", "304": "MEDIUM", "358": "MEDIUM",
    # LOW - Theft, mischief
    "379": "LOW", "427": "LOW", "447": "LOW", "336": "LOW",
}

# IPC Section → Severity mapping (legacy / pre-BNS regime)
IPC_SEVERITY_MAP = {
    # CRITICAL - Murder
    "302": "CRITICAL", "307": "CRITICAL", "300": "CRITICAL",
    # CRITICAL - Sexual Assault
    "376": "CRITICAL", "354": "CRITICAL", "375": "CRITICAL",
    # HIGH - Violent crimes
    "395": "HIGH", "396": "HIGH", "397": "HIGH", "307": "HIGH",
    "326": "HIGH", "325": "HIGH",
    # HIGH - Dacoity / Robbery
    "390": "HIGH", "391": "HIGH", "392": "HIGH", "393": "HIGH",
    # MEDIUM - Fraud, cheating
    "420": "MEDIUM", "406": "MEDIUM", "409": "MEDIUM",
    "467": "MEDIUM", "468": "MEDIUM", "471": "MEDIUM",
    # LOW - Theft, mischief
    "379": "LOW", "380": "LOW", "381": "LOW",
    "425": "LOW", "426": "LOW", "427": "LOW",
    "441": "LOW", "447": "LOW",
}


class PrioritizationAgent:
    def __init__(self):
        self.model_path   = "data/prioritization/models/prioritizer.pkl"
        self.encoder_path = "data/prioritization/models/encoders.pkl"
        self.model   = None
        self.encoder = None

        if os.path.exists(self.model_path) and os.path.exists(self.encoder_path):
            try:
                self.model   = joblib.load(self.model_path)
                self.encoder = joblib.load(self.encoder_path)
                print("PrioritizationAgent: Loaded XGBoost v2 model (trained on Dr. Saritha P legal rules).")
            except Exception as e:
                print(f"Warning: Could not load XGBoost model ({e}). Falling back to rule engine.")
        else:
            print("PrioritizationAgent: No model file found. Using rule-based engine.")

    # ------------------------------------------------------------------ #
    #  HELPER: BNS / IPC section → severity level
    # ------------------------------------------------------------------ #
    def _sections_to_severity(self, sections_str: str, regime: str = "BNS") -> str:
        """Return highest severity level from a comma-separated section string."""
        if not sections_str:
            return None
        severity_map = BNS_SEVERITY_MAP if regime.upper() != "IPC" else IPC_SEVERITY_MAP
        order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        best = None
        for sec in sections_str.split(","):
            sec = sec.strip()
            sev = severity_map.get(sec)
            if sev:
                if best is None or order.index(sev) < order.index(best):
                    best = sev
        return best

    # ------------------------------------------------------------------ #
    #  PART 2: AGE MULTIPLIER
    # ------------------------------------------------------------------ #
    def _get_age_multiplier(self, case_age_days: float) -> float:
        """Return multiplier based on how long case has been pending."""
        if case_age_days is None or math.isnan(case_age_days) or math.isinf(case_age_days):
            return 1.0
        years = case_age_days / 365.0
        if years < 0.5:
            return 1.0
        elif years < 1.0:
            return 1.1
        elif years < 2.0:
            return 1.3
        elif years < 3.0:
            return 1.5
        elif years < 5.0:
            return 1.8
        else:
            return 2.0

    # ------------------------------------------------------------------ #
    #  PART 3: URGENCY BOOST
    # ------------------------------------------------------------------ #
    def _get_urgency_boost(self, text: str) -> tuple[int, list[str]]:
        """
        Scan text for urgency keywords.
        Returns (total_boost, list_of_reasons).
        """
        text_lower = text.lower()
        boost = 0
        reasons = []

        # +5: Bail / In-Custody
        custody_patterns = [
            r'\bin custody\b', r'\bin jail\b', r'\bdetention\b',
            r'\baccused is detained\b', r'\bremand\b', r'\bbail petition\b',
            r'\banticipatory bail\b', r'\bhabeas corpus\b',
        ]
        if any(re.search(p, text_lower) for p in custody_patterns):
            boost += 5
            reasons.append("+5 (accused in custody / bail petition)")

        # +3: Constitutional / Fundamental Rights
        const_patterns = [
            r'\bfundamental rights?\b', r'\bconstitutional\b',
            r'\bfreedom of\b', r'\bright to life\b', r'\bdignity\b',
        ]
        if any(re.search(p, text_lower) for p in const_patterns):
            boost += 3
            reasons.append("+3 (constitutional / fundamental rights)")

        # +2: Time-Sensitive
        time_patterns = [
            r'\bmaintenance\b', r'\balimony\b', r'\bchild custody\b',
            r'\bchild support\b', r'\bguardianship\b', r'\bmedical\b',
            r'\bemergency\b',
        ]
        if any(re.search(p, text_lower) for p in time_patterns):
            boost += 2
            reasons.append("+2 (time-sensitive: maintenance / child / medical)")

        # +1: Precedent-Setting
        prec_patterns = [
            r'\blandmark\b', r'\bprecedent\b', r'\bfirst time\b',
            r'\bnovel issue\b',
        ]
        if any(re.search(p, text_lower) for p in prec_patterns):
            boost += 1
            reasons.append("+1 (precedent-setting / landmark case)")

        return boost, reasons

    # ------------------------------------------------------------------ #
    #  PART 1: BASE PRIORITY
    # ------------------------------------------------------------------ #
    def _get_base_priority(
        self,
        case_type: str,
        bns_sections: str = "",
        ipc_sections: str = "",
        text: str = "",
    ) -> tuple[float, str]:
        """
        Determine base priority from case type + section severity.
        Returns (base_priority, severity_label).
        """
        ct = (case_type or "").lower()

        if ct == "criminal":
            # Try BNS first, then IPC
            severity = self._sections_to_severity(bns_sections, "BNS")
            if not severity:
                severity = self._sections_to_severity(ipc_sections, "IPC")

            # Domestic violence keyword override → HIGH
            if not severity or severity == "LOW":
                text_lower = (text or "").lower()
                if re.search(r'\bdomestic violence\b|\bwife assault\b|\bchild abuse\b', text_lower):
                    severity = "HIGH"

            severity = severity or "MEDIUM"  # default
            base = BASE_PRIORITY_TABLE["criminal"].get(severity, 5.0)
            return base, severity

        else:  # civil
            text_lower = (text or "").lower()

            # CRITICAL: Family / maintenance / minors
            if re.search(r'\bguardianship\b|\bcustody\b|\bmaintenance\b|\balimony\b|\bminor\b', text_lower):
                return BASE_PRIORITY_TABLE["civil"]["CRITICAL"], "CRITICAL"

            # HIGH: Inheritance
            if re.search(r'\binheritance\b|\bwill dispute\b|\bsuccession\b', text_lower):
                return BASE_PRIORITY_TABLE["civil"]["HIGH"], "HIGH"

            # Try to detect monetary amounts (in lakhs)
            amount_match = re.search(
                r'(?:rs\.?|rupees?)\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lakhs|crore|crores)?',
                text_lower
            )
            if amount_match:
                try:
                    raw = float(amount_match.group(1).replace(",", ""))
                    suffix = amount_match.group(0).lower()
                    amount_lakhs = raw * 100 if "crore" in suffix else raw
                    if amount_lakhs > 50:
                        return BASE_PRIORITY_TABLE["civil"]["HIGH"], "HIGH"
                    elif amount_lakhs >= 5:
                        return BASE_PRIORITY_TABLE["civil"]["MEDIUM"], "MEDIUM"
                    else:
                        return BASE_PRIORITY_TABLE["civil"]["LOW"], "LOW"
                except ValueError:
                    pass

            return BASE_PRIORITY_TABLE["civil"]["LOW"], "LOW"

    # ------------------------------------------------------------------ #
    #  MAIN: compute_priority_score  (fully rule-based, legally approved)
    # ------------------------------------------------------------------ #
    def _predict_raw_ml(
        self,
        case_type: str,
        bns_sections: str = "",
        ipc_sections: str = "",
        case_age_days: float = None,
        text: str = "",
    ) -> float | None:
        if self.model is None or self.encoder is None:
            return None
        try:
            import pandas as pd
            age = float(case_age_days) if self.is_valid_age(case_age_days) else 365.0
            base, sev_label = self._get_base_priority(case_type, bns_sections, ipc_sections, text)
            sev_num = {"CRITICAL": 9, "HIGH": 7, "MEDIUM": 5, "LOW": 3}.get(sev_label, 5)

            try:
                ct_enc = self.encoder.transform([case_type.lower() if case_type else "civil"])[0]
            except Exception:
                ct_enc = 1 if (case_type or "").lower() == "criminal" else 0

            row = pd.DataFrame([{
                "case_type_enc":          ct_enc,
                "num_ipc_sections":        len([s for s in bns_sections.split(",") if s]) + len([s for s in ipc_sections.split(",") if s]),
                "num_cpc_sections":        0,
                "num_precedents":          0,
                "total_words":             len(text.split()) if text else 0,
                "case_age_days":           age,
                "max_severity_score":      float(sev_num),
                "immediate_threat_flag":   self.check_immediate_threat(text),
                "societal_impact_score":   self.calculate_societal_impact(text),
            }])
            pred = float(self.model.predict(row)[0])
            return round(min(10.0, max(0.0, pred)), 2)
        except Exception as e:
            print(f"XGBoost raw prediction failed: {e}")
            return None

    def compute_priority_score(
        self,
        case_type: str,
        bns_sections: str = "",
        ipc_sections: str = "",
        case_age_days: float = None,
        text: str = "",
        severity: int = None,
    ) -> float:
        """
        Calculates constrained hybrid priority score.
        Rules set FLOORS/CEILINGS. ML refines within safe bounds.
        """
        # Calculate raw rule score
        base, severity_label = self._get_base_priority(case_type, bns_sections, ipc_sections, text)
        multiplier = self._get_age_multiplier(case_age_days)
        boost, _ = self._get_urgency_boost(text)
        rule_score = base * multiplier + boost

        # Fetch ML score if active
        ml_score = self._predict_raw_ml(case_type, bns_sections, ipc_sections, case_age_days, text)

        if ml_score is not None:
            # Constrained Hybrid Logic
            FLOOR_MAP = {
                "CRITICAL": 8.0,
                "HIGH":     6.5,
                "MEDIUM":   4.5,
                "LOW":      2.0
            }
            floor = FLOOR_MAP.get(severity_label, 0.0)
            
            # Floor constraint
            final = max(floor, ml_score)
            # Ceiling constraint (don't exceed rule_score + 1.0)
            final = min(final, rule_score + 1.0)
            
            return round(min(10.0, max(0.0, final)), 2)

        # Fallback to pure rules
        return round(min(10.0, max(0.0, rule_score)), 2)

    # ------------------------------------------------------------------ #
    #  Backward-compatible aliases used by api.py
    # ------------------------------------------------------------------ #
    def check_immediate_threat(self, text: str) -> int:
        """Returns 1 if any custody/bail/threat keyword found, else 0."""
        patterns = [
            r'\bbail petition\b', r'\banticipatory bail\b', r'\bhabeas corpus\b',
            r'\bdomestic violence\b', r'\bcustody\b', r'\binjunction\b',
            r'\bstay order\b', r'\bthreat to life\b', r'\bprotection order\b',
        ]
        return int(any(re.search(p, text.lower()) for p in patterns))

    def calculate_societal_impact(self, text: str) -> int:
        text_lower = text.lower()
        score = 1
        if re.search(r'\bpublic interest litigation\b|\bpil\b', text_lower): score += 3
        if re.search(r'\benvironmental\b|\bpollution\b', text_lower): score += 2
        if re.search(r'\bconstitutional\b|\bfundamental rights?\b', text_lower): score += 2
        if re.search(r'\bcorruption\b|\bcbi\b|\bscam\b|\bterror\b', text_lower): score += 2
        return min(score, 5)

    def predict_ml_priority(self, **kwargs) -> float:
        """Deprecated ML path — delegates to rule-based compute."""
        return self.compute_priority_score(
            case_type=kwargs.get("case_type", "civil"),
            case_age_days=kwargs.get("case_age_days"),
        )

    def is_valid_age(self, age) -> bool:
        if age is None:
            return False
        try:
            val = float(age)
            return not (math.isnan(val) or math.isinf(val))
        except (ValueError, TypeError):
            return False

    def assign_category(self, score: float) -> str:
        if score is None or math.isnan(score): return "Unknown"
        if score >= 8.5: return "Critical"
        if score >= 7.0: return "High"
        if score >= 5.0: return "Medium"
        return "Low"

    # ------------------------------------------------------------------ #
    #  EXPLAINABILITY: generate_explanation
    # ------------------------------------------------------------------ #
    def generate_explanation(
        self,
        case_type: str,
        num_ipc_sections: int = 0,
        num_cpc_sections: int = 0,
        case_age_days: float = None,
        num_precedents: int = 0,
        bns_sections: str = "",
        ipc_sections: str = "",
        text: str = "",
    ) -> str:
        """
        Returns a step-by-step human-readable explanation of the priority score.
        """
        base, severity = self._get_base_priority(case_type, bns_sections, ipc_sections, text)
        multiplier = self._get_age_multiplier(case_age_days)
        boost, boost_reasons = self._get_urgency_boost(text)

        age_adjusted = base * multiplier
        rule_score = round(min(10.0, max(0.0, age_adjusted + boost)), 2)
        
        # Get actual final score (which runs through XGBoost if loaded)
        final = self.compute_priority_score(
            case_type=case_type,
            bns_sections=bns_sections,
            ipc_sections=ipc_sections,
            case_age_days=case_age_days,
            text=text
        )
        category = self.assign_category(final)

        if self.is_valid_age(case_age_days):
            years = round(float(case_age_days) / 365, 1)
            age_str = f"{years} years pending"
        else:
            age_str = "0.0 years pending"

        lines = [
            f"Base priority: {base} ({severity} - {(case_type or 'unknown').capitalize()} case)",
            f"Age adjustment: ×{multiplier} ({age_str})",
            f"Final score: {final} / 10.0",
        ]
        if boost_reasons:
            lines.append("Urgency factors: " + ", ".join(boost_reasons))
        else:
            lines.append("Urgency factors: None")

        lines.append(f"Baseline rule score: {rule_score}")

        if self.model is not None:
            # We fetch the raw ML prediction first to show it
            ml_score = self._predict_raw_ml(case_type, bns_sections, ipc_sections, case_age_days, text)
            lines.append(f"XGBoost predicted score: {ml_score}")
            
            # Show safety floor if applied
            floor_map = {"CRITICAL": 8.0, "HIGH": 6.5, "MEDIUM": 4.5, "LOW": 2.0}
            floor = floor_map.get(severity, 0.0)
            if ml_score < floor:
                lines.append(f"Safety Floor Applied: {floor} ({severity} cases cannot drop below {floor})")

        lines.append(f"Category: {category}")

        action_map = {
            "Critical": "Schedule IMMEDIATELY within 1-2 weeks",
            "High":     "Schedule urgently within 2-4 weeks",
            "Medium":   "Standard scheduling within 1-3 months",
            "Low":      "Routine queue"
        }
        lines.append(f"Action: {action_map.get(category, 'Routine queue')}")

        return " | ".join(lines)
