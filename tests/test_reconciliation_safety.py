import json
import unittest
from unittest.mock import patch

import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.reconciliation.reconciliation_agent import (
    JudicialReconciliationAgent,
    synthesize_judicial_memo_node,
)


class FakeGemini:
    def __init__(self, score, notes="bounded review"):
        self.score = score
        self.notes = notes

    def generate_content(self, prompt):
        return type("Response", (), {
            "text": json.dumps({
                "reconciled_score": self.score,
                "reconciliation_status": "ADJUSTED",
                "deliberation_notes": self.notes,
            })
        })()


class TestReconciliationSafety(unittest.TestCase):
    def setUp(self):
        self.agent = JudicialReconciliationAgent()

    @patch("agents.reconciliation.reconciliation_agent.GEMINI_MODEL", None)
    def test_unaudited_case_never_claims_neutrality(self):
        result = self.agent.reconcile({
            "case_id": "routine_case",
            "summary": "Routine civil suit.",
            "priority_score": 3.0,
            "bias_score": None,
        })
        self.assertIn("no demographic-language screen score/result was recorded", result["memo"].lower())
        self.assertNotIn("verified", result["memo"].lower())
        self.assertNotIn("acceptable threshold", result["memo"].lower())
        self.assertIn("not an approved court listing policy", result["memo"].lower())

    @patch("agents.reconciliation.reconciliation_agent.GEMINI_MODEL", None)
    def test_liberty_floor_is_applied_without_llm(self):
        result = self.agent.reconcile({
            "case_id": "habeas_case",
            "summary": "Habeas corpus petition concerning unlawful custody.",
            "priority_score": 4.2,
            "bias_score": 0.95,
        })
        self.assertTrue(result["conflict_detected"])
        self.assertEqual(result["reconciled_score"], 6.0)
        self.assertEqual(result["original_priority_score"], 4.2)
        self.assertTrue(result["score_changed"])
        self.assertTrue(any("Deterministic triage guardrail" in reason for reason in result["conflict_reasons"]))
        self.assertNotIn("Demographic-language caution", result["recommendation"])

    @patch("agents.reconciliation.reconciliation_agent.GEMINI_MODEL", None)
    def test_age_contribution_alone_does_not_create_skew_conflict(self):
        result = self.agent.reconcile({
            "case_id": "age_skew_case",
            "summary": "Routine appeal.",
            "priority_score": 8.0,
            "bias_score": None,
            "contributions": {"case_age_days": 3.0, "max_severity_score": 0.0},
        })
        self.assertFalse(result["conflict_detected"])
        self.assertEqual(result["original_priority_score"], 8.0)
        self.assertFalse(result["score_changed"])
        self.assertNotIn("Demographic-language caution", result["recommendation"])
        self.assertIn("not a pass or proof of neutrality", result["memo"])

    @patch("agents.reconciliation.reconciliation_agent.GEMINI_MODEL", FakeGemini(99.0))
    def test_llm_score_cannot_break_liberty_floor_or_upper_bound(self):
        result = self.agent.reconcile({
            "case_id": "habeas_case",
            "summary": "Habeas corpus petition concerning unlawful custody.",
            "priority_score": 4.2,
            "bias_score": None,
        })
        self.assertEqual(result["reconciled_score"], 6.2)

    @patch("agents.reconciliation.reconciliation_agent.GEMINI_MODEL", FakeGemini(-99.0))
    def test_llm_score_change_is_bounded_without_liberty_signal(self):
        result = self.agent.reconcile({
            "case_id": "threat_signal_case",
            "summary": "Ordinary filing.",
            "priority_score": 5.0,
            "bias_score": None,
            "contributions": {"immediate_threat_flag": 2.5},
        })
        self.assertEqual(result["reconciled_score"], 4.0)

    def test_precedents_use_rank_not_percentage_match(self):
        result = synthesize_judicial_memo_node({
            "case_id": "memo_case",
            "case_type": "civil",
            "priority_score": 4.0,
            "reconciled_score": 4.0,
            "reconciliation_status": "APPROVED",
            "bias_score": None,
            "bias_flags": [],
            "bias_details": "",
            "precedents": [
                {"case_id": "lower_score", "score": 0.82},
                {"case_id": "higher_similarity", "similarity": 0.99},
            ],
            "conflict_reasons": [],
            "demographic_context_flagged": False,
            "case_summary": "A supplied summary.",
        })
        memo = result["reconciliation_memo"]
        self.assertIn("Rank 1: higher similarity", memo)
        self.assertIn("Rank 2: lower score", memo)
        self.assertNotIn("% match", memo.lower())
        self.assertNotIn("0.99", memo)


if __name__ == "__main__":
    unittest.main()
