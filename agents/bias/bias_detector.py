from typing import Dict, List, Any

class BiasDetector:
    def audit_case_priority(self, case_features: Dict[str, Any], priority_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Audits a single case priority score for anomalies and logical inconsistency.
        Returns a dictionary with specific bias warning flags, a fairness score, and human-in-the-loop recommendations.
        """
        bias_flags = []
        
        # 1. Severity Consistency Check
        severity = priority_result.get("severity", "MEDIUM")
        final_score = priority_result.get("final_priority", 5.0)
        
        expected_ranges = {
            "CRITICAL": (7.5, 10.0),
            "HIGH": (6.0, 8.5),
            "MEDIUM": (4.5, 7.0),
            "LOW": (0.0, 5.5)
        }
        
        min_val, max_val = expected_ranges.get(severity, (0.0, 10.0))
        severity_flagged = not (min_val <= final_score <= max_val)
        
        bias_flags.append({
            "type": "severity_consistency",
            "flagged": severity_flagged,
            "message": f"Case severity is '{severity}' but final priority score is {final_score:.2f} (expected range: {min_val}-{max_val})." if severity_flagged else "OK"
        })
        
        # 2. Case Type Skew Check (Criminal vs Civil)
        case_type = priority_result.get("case_type", "CIVIL").upper()
        case_type_flagged = False
        case_type_message = "OK"
        
        if case_type == "CRIMINAL" and final_score < 4.0:
            case_type_flagged = True
            case_type_message = "Criminal case is assigned unusually low priority (< 4.0). Verify if personal liberty/bail was overlooked."
        elif case_type == "CIVIL" and final_score > 8.5:
            case_type_flagged = True
            case_type_message = "Civil case is prioritized exceptionally high (> 8.5). Confirm if extreme safety or custody urgency is present."
            
        bias_flags.append({
            "type": "case_type_skew",
            "flagged": case_type_flagged,
            "message": case_type_message
        })
        
        # 3. Extreme Delay Influence Check
        delay_factor = priority_result.get("delay_factor", 1.0)
        delay_flagged = delay_factor > 1.45
        
        bias_flags.append({
            "type": "extreme_delay_skew",
            "flagged": delay_flagged,
            "message": f"Delay factor ({delay_factor:.2f}) is approaching the absolute limit (1.50). Ensure severity is not being excessively distorted by backlog aging." if delay_flagged else "OK"
        })
        
        # Calculate individual case fairness score (0.0 to 1.0)
        total_checks = len(bias_flags)
        flagged_count = sum(1 for flag in bias_flags if flag["flagged"])
        fairness_score = 1.0 - (flagged_count / total_checks) if total_checks > 0 else 1.0
        
        # Human-in-the-loop recommendation
        flagged_warnings = [f for f in bias_flags if f["flagged"]]
        if not flagged_warnings:
            recommendation = "No prioritization bias or anomaly detected. The priority score appears consistent with legal guidelines."
        elif len(flagged_warnings) == 1:
            recommendation = f"Minor inconsistency identified: {flagged_warnings[0]['message']} Review case details before scheduling."
        else:
            recommendation = f"Multiple anomalies detected ({len(flagged_warnings)}/3). High risk of bias or scoring distortion. Presiding judge / court registrar override recommended."
            
        return {
            "case_id": priority_result.get("case_id", "unknown"),
            "final_priority": final_score,
            "fairness_score": round(fairness_score * 100, 1),
            "bias_flags": bias_flags,
            "recommendation": recommendation
        }

    def compute_systemic_fairness(self, cases_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes system-level fairness metrics (Demographic Parity) across a list/dataset of audited cases.
        Uses case_type as the demographic attribute.
        """
        if not cases_data:
            return {"demographic_parity_difference": 0.0, "status": "No data available"}
            
        civil_cases = [c for c in cases_data if str(c.get("case_type")).upper() == "CIVIL"]
        criminal_cases = [c for c in cases_data if str(c.get("case_type")).upper() == "CRIMINAL"]
        
        threshold = 7.0
        
        def get_selection_rate(group):
            if not group:
                return 0.0
            high_priority_count = sum(1 for c in group if c.get("final_priority", 0.0) >= threshold)
            return high_priority_count / len(group)
            
        civil_rate = get_selection_rate(civil_cases)
        criminal_rate = get_selection_rate(criminal_cases)
        
        dp_diff = abs(civil_rate - criminal_rate)
        bias_detected = dp_diff > 0.20
        
        return {
            "total_cases_analyzed": len(cases_data),
            "civil_high_priority_rate": round(civil_rate * 100, 2),
            "criminal_high_priority_rate": round(criminal_rate * 100, 2),
            "demographic_parity_difference": round(dp_diff * 100, 2),
            "systemic_skew_detected": bias_detected,
            "recommendation": (
                "Systemic distribution is balanced." if not bias_detected else
                "Noticeable systemic skew detected between Civil and Criminal priority distributions. Evaluate baseline heuristic calibration."
            )
        }
