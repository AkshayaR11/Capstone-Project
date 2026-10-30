import unittest
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.prioritization.prioritizer import PrioritizationAgent
from agents.explainability.explainability_agent import ExplainabilityAgent
from agents.fairness.bias_agent import BiasAgent
from agents.bias.bias_detector import BiasDetector
from agents.reconciliation.reconciliation_agent import JudicialReconciliationAgent

class TestJudicialDecisionSupportSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prioritizer = PrioritizationAgent()
        cls.bias_agent = BiasAgent()
        cls.bias_detector = BiasDetector()
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
        # Civil severity capped at 0.4.
        # sev_norm = min(3/10, 0.4) = 0.3 -> 0.3 * 3.5 = 1.05
        # impact = 1/5 * 2.5 = 0.50
        # threat = 0.0
        # recency = (1 - 365/3650) = 0.9 -> 0.9 * 1.0 = 0.90
        # Total = 1.05 + 0.50 + 0.0 + 0.90 = 2.45
        self.assertAlmostEqual(score, 2.45, places=1)
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
        # recency = (1 - 180/3650) * 1.0 ~= 0.95
        # Total = 3.5 + 1.5 + 3.0 + 0.95 = 8.95
        self.assertTrue(score >= 8.0)
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

if __name__ == "__main__":
    unittest.main()
