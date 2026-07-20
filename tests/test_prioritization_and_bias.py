import unittest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.prioritization.prioritizer import PrioritizationAgent
from agents.bias.bias_detector import BiasDetector

class TestPrioritizationAndBias(unittest.TestCase):
    def setUp(self):
        self.calc = PrioritizationAgent()
        self.detector = BiasDetector()

    def test_severity_classification(self):
        # 1. Statutory Criminal Critical (Section 302 - Murder)
        base1, sev1 = self.calc._get_base_priority("CRIMINAL", ipc_sections="IPC 302", text="The accused committed robbery.")
        self.assertEqual(sev1, "CRITICAL")
        self.assertEqual(base1, 5.0)
        
        # 2. Statutory Criminal Low (Section 379 - Theft)
        base2, sev2 = self.calc._get_base_priority("CRIMINAL", ipc_sections="IPC 379", text="The accused stole a bicycle.")
        self.assertEqual(sev2, "LOW")
        self.assertEqual(base2, 2.0)

        # 3. Fallback Keyword Criminal Critical
        base3, sev3 = self.calc._get_base_priority("CRIMINAL", text="Accused sentenced to life imprisonment for violent assault.")
        self.assertEqual(sev3, "CRITICAL")
        self.assertEqual(base3, 5.0)

        # 4. Fallback Keyword Civil High
        base4, sev4 = self.calc._get_base_priority("CIVIL", text="High-value property disputes regarding significant financial transactions.")
        self.assertEqual(sev4, "HIGH")
        self.assertEqual(base4, 3.5)

    def test_priority_score_calculation(self):
        # Test recent, low-severity case (No age escalation, no boosts)
        res_recent = self.calc.compute_priority_score(
            case_type="CIVIL",
            case_age_days=10.0,  # very recent
            text="Routine partition dispute of property."
        )
        self.assertEqual(res_recent, 1.5)  # Low Civil = 1.5

        # Test delayed case with urgency and societal boosts
        res_complex = self.calc.compute_priority_score(
            case_type="CRIMINAL",
            ipc_sections="302", # Critical = 5.0
            case_age_days=1825.0,  # ~5 years old
            text="Murder case. Accused in custody for 5 years. Severe domestic abuse of woman involved."
        )
        # base (5.0) * delay (>1.0) + urgency (2.5) + societal (1.0)
        self.assertTrue(res_complex > 8.5)

    def test_bias_auditor(self):
        # Check severity consistency warning (Critical case with low score)
        case_mock = {
            "case_id": "mock001",
            "case_type": "CRIMINAL",
            "sections": ["IPC 302"],
            "summary": "Murder case.",
            "filing_date": "2026-07-01"
        }
        # Artificially modify priority result to trigger severity warning
        p_res = {
            "case_id": "mock001",
            "case_type": "CRIMINAL",
            "severity": "CRITICAL",
            "final_priority": 3.0
        }
        
        audit = self.detector.audit_case_priority(case_mock, p_res)
        self.assertTrue(audit["fairness_score"] < 100.0)
        severity_warnings = [f for f in audit["bias_flags"] if f["type"] == "severity_consistency"]
        self.assertEqual(len(severity_warnings), 1)
        self.assertTrue(severity_warnings[0]["flagged"])

    def test_systemic_fairness(self):
        mock_priorities = [
            {"case_type": "CRIMINAL", "final_priority": 8.5},
            {"case_type": "CRIMINAL", "final_priority": 7.5},
            {"case_type": "CRIMINAL", "final_priority": 9.0},
            {"case_type": "CRIMINAL", "final_priority": 8.0},
            {"case_type": "CRIMINAL", "final_priority": 4.5}, # 80% high priority
            
            {"case_type": "CIVIL", "final_priority": 8.0},
            {"case_type": "CIVIL", "final_priority": 4.0},
            {"case_type": "CIVIL", "final_priority": 3.5},
            {"case_type": "CIVIL", "final_priority": 2.0},
            {"case_type": "CIVIL", "final_priority": 1.5}, # 20% high priority
        ]
        
        sys_report = self.detector.compute_systemic_fairness(mock_priorities)
        self.assertEqual(sys_report["demographic_parity_difference"], 60.0)
        self.assertTrue(sys_report["systemic_skew_detected"])

if __name__ == "__main__":
    unittest.main()
