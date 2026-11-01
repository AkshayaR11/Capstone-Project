import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.prioritization.prioritizer import PrioritizationAgent
from agents.fairness.bias_agent import BiasAgent
from agents.reconciliation.reconciliation_agent import JudicialReconciliationAgent

class TestJudicialDecisionSupportSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prioritizer = PrioritizationAgent()
        cls.bias_agent = BiasAgent()
        cls.reconciliation_agent = JudicialReconciliationAgent()

    def test_priority_score_calculation_civil_routine(self):
        """Test a routine civil case: low severity, no threat, standard pendency."""
        score = self.prioritizer.compute_priority_score(
            severity=3,
            societal_impact=1,
            immediate_threat=0,
            case_age_days=365.0,
            case_type="civil"
        )
        # Civil severity capped at 0.45; severity 3 gives 0.3 * 3.5 = 1.05.
        # impact = 1/5 * 2.5 = 0.50
        # threat = 0.0
        # One-year age adjustment = 0.15 + (1 * 0.065) = 0.215.
        # Total = 1.05 + 0.50 + 0.0 + 0.215 = 1.765 -> 1.77.
        self.assertAlmostEqual(score, 1.77, places=2)
        self.assertEqual(self.prioritizer.assign_category(score), "Low")

    def test_priority_score_calculation_criminal_critical(self):
        """Test a critical criminal case (murder + bail/threat to life)."""
        score = self.prioritizer.compute_priority_score(
            severity=10,
            societal_impact=3,
            immediate_threat=1,
            case_age_days=180.0,
            case_type="criminal"
        )
        # sev_norm = 1.0 -> 3.5
        # impact = 3/5 * 2.5 = 1.5
        # threat = 1 * 3.0 = 3.0
        # Age adjustment = 0.15 + ((180/365) * 0.065) ~= 0.182.
        # Total ~= 3.5 + 1.5 + 3.0 + 0.182 = 8.18.
        self.assertAlmostEqual(score, 8.18, places=2)
        self.assertEqual(self.prioritizer.assign_category(score), "Critical")

    def test_immediate_threat_detection(self):
        """Verify threat detection catches bail petitions and habeas corpus."""
        urgent_text = "The petitioner has filed an urgent application for anticipatory bail under Article 21."
        routine_text = "Suit for declaration of title and partition of joint agricultural land."
        self.assertEqual(self.prioritizer.check_immediate_threat(urgent_text), 1)
        self.assertEqual(self.prioritizer.check_immediate_threat(routine_text), 0)

    def test_societal_impact_detection(self):
        """Verify societal impact score detects PIL and fundamental rights."""
        pil_text = "This Public Interest Litigation (PIL) challenges environmental pollution and constitutional validity."
        routine_text = "Dispute regarding simple money recovery of promissory note."
        self.assertTrue(self.prioritizer.calculate_societal_impact(pil_text) >= 4)
        self.assertEqual(self.prioritizer.calculate_societal_impact(routine_text), 1)

    def test_explainability_trace(self):
        """Verify natural language explanation highlights key drivers."""
        trace = self.prioritizer.generate_explanation(
            case_type="criminal",
            num_ipc_sections=2,
            num_cpc_sections=0,
            case_age_days=2500.0,
            num_precedents=7,
            severity=9.0,
            immediate_threat=1,
            societal_impact=4
        )
        self.assertIn("Criminal Case", trace)
        self.assertIn("Immediate threat to life or liberty", trace)
        self.assertIn("High severity offence", trace)

    def test_bias_lexical_audit(self):
        """Verify lexical bias auditor catches discriminatory markers."""
        biased_text = "The accused belongs to a suspicious background and suspect community."
        neutral_text = "The appellant challenged the decree passed by the trial court in accordance with law."

        biased_res = self.bias_agent._lexical_audit(biased_text)
        self.assertIsNotNone(biased_res["bias_score"])
        self.assertTrue(len(biased_res["flags"]) > 0)
        self.assertTrue(biased_res["bias_score"] < 0.98)

        neutral_res = self.bias_agent._lexical_audit(neutral_text)
        self.assertIsNone(neutral_res["bias_score"])  # Unflagged/neutral returns None (automated audit unavailable)

    def test_langgraph_reconciliation_workflow(self):
        """Verify LangGraph reconciliation detects custody urgency discrepancy."""
        dossier = {
            "case_id": "TEST_RECONCILE_001",
            "case_type": "criminal",
            "legal_regime": "ipc",
            "summary": "Urgent habeas corpus petition alleging illegal detention and unlawful custody without trial.",
            "priority_score": 4.2,  # Low priority erroneously assigned
            "severity": 4.0,
            "contributions": {"immediate_threat_flag": 2.5},
            "bias_score": 0.95,
            "bias_details": "No bias found.",
            "bias_flags": [],
            "precedents": []
        }
        res = self.reconciliation_agent.reconcile(dossier)
        self.assertIn("status", res)
        self.assertIn("recommendation", res)
        self.assertTrue(res["conflict_detected"])  # Should flag liberty exception!
        self.assertIn("memo", res)
    def test_priority_uses_deterministic_policy(self):
        self.assertIsNone(self.prioritizer.model)
        self.assertIsNone(self.prioritizer.encoder)
        rule_score = self.prioritizer.compute_priority_score(
            severity=8,
            societal_impact=3,
            immediate_threat=1,
            case_age_days=730,
            case_type="criminal"
        )
        compatibility_score = self.prioritizer.predict_ml_priority(
            "criminal", 2, 0, 5, 2000, 730, 8, 1, 3
        )
        self.assertEqual(compatibility_score, rule_score)

    def test_ml_priority_bounded(self):
        s = self.prioritizer.predict_ml_priority("criminal", 3, 0, 5, 2000, None, 10, 1, 5)
        self.assertTrue(0.0 <= s <= 10.0)

    def test_ipc_regex(self):
        from scripts.extract_features import extract_ipc_sections
        self.assertIn("302", extract_ipc_sections("convicted u/s 302/34 IPC"))


if __name__ == "__main__":
    unittest.main()
