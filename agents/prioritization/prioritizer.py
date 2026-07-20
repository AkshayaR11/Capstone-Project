import re
import math
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================
# REFINED JUDICIAL CASE PRIORITIZATION ENGINE (MODULE 3)
# Deterministic, explainable, and legally defensible
# ============================================================

class PrioritizationAgent:
    # Base priority by case type and severity (Expert-informed baseline heuristics)
    BASE_PRIORITIES = {
        "CRIMINAL": {
            "CRITICAL": 5.0,  # Life imprisonment, capital offenses, bail
            "HIGH": 4.0,      # Serious offenses against body/high-value property
            "MEDIUM": 3.0,    # General theft, moderate cheating/forgery
            "LOW": 2.0        # Petty offenses, public nuisance
        },
        "CIVIL": {
            "CRITICAL": 4.5,  # Custody disputes, urgent maintenance
            "HIGH": 3.5,      # High-value property, partition, commercial disputes
            "MEDIUM": 2.5,    # General contract, rent, land tenancy
            "LOW": 1.5        # Routine civil declarations, minor appeals
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
        "395": "HIGH", "396": "HIGH", "397": "HIGH", "326": "HIGH", "325": "HIGH",
        # HIGH - Dacoity / Robbery
        "390": "HIGH", "391": "HIGH", "392": "HIGH", "393": "HIGH",
        # MEDIUM - Fraud, cheating
        "420": "MEDIUM", "406": "MEDIUM", "409": "MEDIUM", "467": "MEDIUM", "468": "MEDIUM", "471": "MEDIUM",
        # LOW - Theft, mischief
        "379": "LOW", "380": "LOW", "381": "LOW", "425": "LOW", "426": "LOW", "427": "LOW", "441": "LOW", "447": "LOW",
    }

    def __init__(self):
        # We enforce rule-based logic to preserve explainability for capstone viva.
        # Deep learning / black-box models are disabled.
        self.model = None
        self.encoder = None

    def _sections_to_severity(self, sections_str: str, regime: str = "BNS") -> str:
        """Return highest severity level from a comma-separated section string."""
        if not sections_str:
            return None
        severity_map = self.BNS_SEVERITY_MAP if regime.upper() != "IPC" else self.IPC_SEVERITY_MAP
        order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        best = None
        for sec in sections_str.split(","):
            sec = sec.strip()
            # Extract numerical digits (e.g. "IPC 302" -> "302")
            num_part = re.sub(r'\D', '', sec)
            sev = severity_map.get(num_part)
            if sev:
                if best is None or order.index(sev) < order.index(best):
                    best = sev
        return best

    def _get_base_priority(
        self,
        case_type: str,
        bns_sections: str = "",
        ipc_sections: str = "",
        text: str = "",
    ) -> Tuple[float, str]:
        """
        Determine base priority from case type + section severity.
        Returns (base_priority, severity_label).
        """
        ct = (case_type or "CIVIL").upper()
        
        # 1. Primary: Statutory mapping
        severity = self._sections_to_severity(bns_sections, "BNS")
        if not severity:
            severity = self._sections_to_severity(ipc_sections, "IPC")

        # 2. Secondary: Fallback Keyword Heuristics
        if not severity:
            text_lower = (text or "").lower()
            if ct == "CRIMINAL":
                if any(w in text_lower for w in ["murder", "rape", "death penalty", "capital offence", "life imprisonment"]):
                    severity = "CRITICAL"
                elif any(w in text_lower for w in ["kidnapping", "abduction", "grievous hurt", "robbery", "dacoity", "extortion"]):
                    severity = "HIGH"
                elif any(w in text_lower for w in ["theft", "cheating", "forgery", "fraud"]):
                    severity = "MEDIUM"
                else:
                    severity = "MEDIUM"
            else:  # CIVIL
                if any(w in text_lower for w in ["child custody", "guardianship", "interim maintenance", "alimony", "domestic violence safety"]):
                    severity = "CRITICAL"
                elif any(w in text_lower for w in ["high-value property", "significant financial", "commercial dispute", "land acquisition"]):
                    severity = "HIGH"
                elif any(w in text_lower for w in ["partition suit", "tenancy", "rent dispute", "specific performance", "contract breach"]):
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

        base = self.BASE_PRIORITIES.get(ct, self.BASE_PRIORITIES["CIVIL"]).get(severity, 2.5)
        return base, severity

    def _get_age_multiplier(self, case_age_days: float) -> float:
        """
        Computes bounded logarithmic delay factor:
        f_delay(t) = 1.0 + 0.18 * ln(1 + t)
        capped at 1.50. Suppressed to 1.0 if t < 0.5 years.
        """
        if case_age_days is None or math.isnan(case_age_days) or math.isinf(case_age_days) or case_age_days < 0:
            return 1.0
            
        t = case_age_days / 365.25
        if t < 0.5:
            return 1.0
            
        factor = 1.0 + 0.18 * math.log1p(t)
        return min(1.50, factor)

    def _get_urgency_boost(self, text: str) -> Tuple[float, List[str]]:
        """
        Scan text for refined legal urgency boosts.
        - Personal Liberty / Bail: +2.5
        - Urgent Protective Orders / Personal Safety: +2.0
        - Interim Injunctions / Irreparable Harm: +1.0
        """
        text_lower = (text or "").lower()
        boost = 0.0
        reasons = []

        # 1. Bail & Personal Liberty (+2.5)
        liberty_patterns = ["bail application", "anticipatory bail", "habeas corpus", "unlawful detention", "custody", "jail", "detention"]
        if any(w in text_lower for w in liberty_patterns):
            boost += 2.5
            reasons.append("+2.50 (Bail / Personal Liberty)")

        # 2. Urgent Protective Orders & Personal Safety (+2.0)
        safety_patterns = ["domestic violence", "protection order", "child custody emergency", "medical treatment override", "senior citizen welfare"]
        if any(w in text_lower for w in safety_patterns):
            boost += 2.0
            reasons.append("+2.00 (Safety / Protective Orders)")

        # 3. Interim Injunctions & Irreparable Harm (+1.0)
        harm_patterns = ["interim injunction", "stay order", "irreparable loss", "status quo", "temporary injunction"]
        if any(w in text_lower for w in harm_patterns):
            boost += 1.0
            reasons.append("+1.00 (Interim Injunctions / Irreparable Harm)")

        return boost, reasons

    def _get_societal_impact_boost(self, text: str) -> Tuple[float, List[str]]:
        """
        Calculates flat boosts for cases affecting the public interest (capped at +2.0).
        - Crimes Against Women & Children: +1.0
        - Large-Scale Public Harms / Financial Frauds: +1.0
        - Public Health / Environmental Safety: +1.0
        - Class Actions & Representative Disputes: +0.5
        - Administrative Policy Interpretation: +0.5
        """
        text_lower = (text or "").lower()
        boosts = []
        reasons = []

        if any(w in text_lower for w in ["domestic abuse", "dowry harassment", "sexual harassment", "pocso", "child labor", "child trafficking"]):
            boosts.append(1.0)
            reasons.append("+1.00 (Crimes Against Women & Children)")

        if any(w in text_lower for w in ["ponzi scheme", "mass fraud", "multi-victim", "consumer scam", "defrauding public"]):
            boosts.append(1.0)
            reasons.append("+1.00 (Large-Scale Financial Fraud / Scams)")

        if any(w in text_lower for w in ["pollution control", "hazardous waste", "industrial disaster", "public health safety", "environmental hazard"]):
            boosts.append(1.0)
            reasons.append("+1.00 (Public Health & Environmental Safety)")

        if any(w in text_lower for w in ["representative suit", "class action", "union dispute", "consumer class", "mass wage"]):
            boosts.append(0.5)
            reasons.append("+0.50 (Class Actions / Representative Disputes)")

        if any(w in text_lower for w in ["ultra vires", "constitutional validity", "statutory rules", "administrative policy"]):
            boosts.append(0.5)
            reasons.append("+0.50 (Administrative Policy Interpretation)")

        total = sum(boosts)
        final_boost = min(2.0, total)
        return final_boost, reasons

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
        Calculates the finalized priority score based on the refined formula.
        Capped between 0.0 and 10.0.
        """
        base, _ = self._get_base_priority(case_type, bns_sections, ipc_sections, text)
        multiplier = self._get_age_multiplier(case_age_days)
        urgency, _ = self._get_urgency_boost(text)
        societal, _ = self._get_societal_impact_boost(text)

        score = base * multiplier + urgency + societal
        return round(min(10.0, max(0.0, score)), 2)

    def assign_category(self, score: float) -> str:
        if score is None or math.isnan(score): return "Unknown"
        if score >= 8.5: return "Critical"
        if score >= 7.0: return "High"
        if score >= 5.0: return "Medium"
        return "Low"

    def is_valid_age(self, age) -> bool:
        if age is None:
            return False
        try:
            val = float(age)
            return not (math.isnan(val) or math.isinf(val))
        except (ValueError, TypeError):
            return False

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
        Formatted specifically to remain compatible with your React frontend's parser.
        """
        base, severity = self._get_base_priority(case_type, bns_sections, ipc_sections, text)
        multiplier = self._get_age_multiplier(case_age_days)
        urgency, urgency_reasons = self._get_urgency_boost(text)
        societal, societal_reasons = self._get_societal_impact_boost(text)

        age_adjusted = base * multiplier
        final = self.compute_priority_score(case_type, bns_sections, ipc_sections, case_age_days, text)
        category = self.assign_category(final)

        if self.is_valid_age(case_age_days):
            years = round(float(case_age_days) / 365.25, 2)
            age_str = f"{years} years pending"
        else:
            age_str = "0.0 years pending"

        lines = [
            f"Base priority: {base} ({severity} - {(case_type or 'unknown').capitalize()} case)",
            f"Age adjustment: ×{multiplier:.2f} ({age_str})",
            f"Final score: {final:.2f} / 10.0",
        ]
        if urgency_reasons:
            lines.append("Urgency factors: " + ", ".join(urgency_reasons))
        else:
            lines.append("Urgency factors: None")

        lines.append(f"Baseline rule score: {age_adjusted + urgency:.2f}")
        
        if societal_reasons:
            lines.append("Societal impact factors: " + ", ".join(societal_reasons))
        else:
            lines.append("Societal impact factors: None")

        lines.append(f"Category: {category}")

        action_map = {
            "Critical": "Schedule IMMEDIATELY within 1-2 weeks",
            "High":     "Schedule urgently within 2-4 weeks",
            "Medium":   "Standard scheduling within 1-3 months",
            "Low":      "Routine queue"
        }
        lines.append(f"Action: {action_map.get(category, 'Routine queue')}")

        return " | ".join(lines)
