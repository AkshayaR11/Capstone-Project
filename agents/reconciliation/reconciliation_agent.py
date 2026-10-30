import os
import sys
import json
import time
import math
import re
from typing import TypedDict, List, Dict, Optional, Any
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATABASE_URL

# Load Gemini API
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
import google.generativeai as genai
if api_key:
    genai.configure(api_key=api_key)
    GEMINI_MODEL = genai.GenerativeModel('gemini-2.5-flash')
else:
    GEMINI_MODEL = None


# ==========================================
# 1. State Definition
# ==========================================
class CaseDossierState(TypedDict):
    # Dossier inputs; precedent retrieval is displayed but not relevance-audited.
    case_id: str
    case_type: str
    legal_regime: str
    case_summary: str
    priority_score: float
    priority_category: str
    severity: float
    contributions: Dict[str, float]
    bias_score: Optional[float]
    bias_details: str
    bias_flags: List[str]
    precedents: List[Dict[str, Any]]
    
    # Internal agent deliberation flags
    conflict_detected: bool
    conflict_reasons: List[str]
    demographic_context_flagged: bool
    liberty_exception_detected: bool
    deliberation_notes: str
    
    # Output reconciliation verdict
    reconciliation_status: str  # "APPROVED" | "ADJUSTED" | "FLAGGED_FOR_BENCH"
    reconciled_score: float
    reconciliation_memo: str
    action_recommendation: str


# ==========================================
# 2. Node Implementations
# ==========================================

LIBERTY_PRIORITY_FLOOR = 6.0
SCORE_ADJUSTMENT_LIMIT = 1.0
LIBERTY_SCORE_ADJUSTMENT_LIMIT = 2.0


def _bounded_score(value: Any, fallback: float = 5.0) -> float:
    try:
        score = float(value)
        if not math.isfinite(score):
            return fallback
        return round(min(10.0, max(0.0, score)), 2)
    except (TypeError, ValueError):
        return fallback


def _bounded_unit_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
        if not math.isfinite(score):
            return None
        return min(1.0, max(0.0, score))
    except (TypeError, ValueError):
        return None


def audit_dossier_node(state: CaseDossierState) -> Dict[str, Any]:
    """
    Apply deterministic priority, explanation, synopsis, and demographic-language checks.
    Retrieved precedent candidates are not evaluated for legal relevance here.
    """
    conflicts = []
    priority = _bounded_score(state.get("priority_score", 5.0))
    bias_score = state.get("bias_score")
    bias_flags = state.get("bias_flags", []) or []
    if not isinstance(bias_flags, list):
        bias_flags = [str(bias_flags)]
    summary = (state.get("case_summary") or "").lower()
    contributions = state.get("contributions", {}) or {}

    # This is a language-screen signal, not an evaluation of ranking fairness.
    numeric_bias_score = _bounded_unit_score(bias_score)
    demographic_context_flagged = bool(bias_flags) or (
        numeric_bias_score is not None and numeric_bias_score < 0.85
    )
    if demographic_context_flagged:
        flag_text = ", ".join(str(flag) for flag in bias_flags) if bias_flags else "low language-screen score"
        conflicts.append(
            f"Demographic-language review: automated text screening flagged {flag_text}. "
            "This is a caution about wording in the document; it does not measure fairness of docket ranking."
        )

    # Check 2: Under-Prioritization of Fundamental Liberties
    # Keyword/attribution signals prompt human review; the shortlist is a heuristic.
    urgent_keywords = ["habeas corpus", "unlawful custody", "illegal detention", "death threat", "custodial violence", "juvenile"]
    matched_kw = [kw for kw in urgent_keywords if kw in summary]
    has_liberty_issue = bool(matched_kw)
    explicit_liberty_keywords = ("habeas corpus", "unlawful custody", "illegal detention", "custodial violence", "death threat")
    deterministic_liberty_signal = any(kw in summary for kw in explicit_liberty_keywords)
    try:
        threat_contrib = float(contributions.get("immediate_threat_flag", 0.0) or 0.0)
        if not math.isfinite(threat_contrib):
            threat_contrib = 0.0
    except (TypeError, ValueError):
        threat_contrib = 0.0

    # An attribution alone is a review signal, not proof of a liberty restraint.
    liberty_exception_detected = deterministic_liberty_signal
    if priority < 6.0 and (has_liberty_issue or threat_contrib > 2.0):
        matched_str = ", ".join(matched_kw) if matched_kw else "high immediate-threat attribution; verify the underlying facts"
        conflicts.append(
            f"Potential liberty/urgency review: model assigned {priority:.1f}/10; review signal: {matched_str}."
        )
    # Explicit custody/liberty language gets a deterministic minimum score before
    # optional LLM deliberation, so the guardrail does not depend on API access.
    needs_liberty_floor = deterministic_liberty_signal and priority < LIBERTY_PRIORITY_FLOOR
    reconciled_score = max(priority, LIBERTY_PRIORITY_FLOOR) if needs_liberty_floor else priority
    if needs_liberty_floor:
        conflicts.append(
            f"Deterministic triage guardrail: score raised to at least {LIBERTY_PRIORITY_FLOOR:.1f}/10 for human review."
        )

    return {
        "conflict_detected": len(conflicts) > 0,
        "conflict_reasons": conflicts,
        "demographic_context_flagged": demographic_context_flagged,
        "liberty_exception_detected": liberty_exception_detected,
        "reconciliation_status": "FLAGGED_FOR_BENCH" if len(conflicts) > 0 else "APPROVED",
        "reconciled_score": reconciled_score
    }


def route_deliberation(state: CaseDossierState) -> str:
    """
    Conditional Edge: Routes to LLM Deliberation if conflicts/exceptions exist;
    otherwise proceeds directly to final memo synthesis.
    """
    if state.get("conflict_detected", False):
        return "deliberate_and_reconcile"
    return "synthesize_judicial_memo"


def deliberate_and_reconcile_node(state: CaseDossierState) -> Dict[str, Any]:
    """
    Optional LLM deliberation over review signals with strict score bounds.
    This step is advisory and does not make a legal finding.
    """
    reasons = "\n- ".join(state.get("conflict_reasons", []))
    original_priority = _bounded_score(state.get("priority_score", 5.0))
    priority = _bounded_score(state.get("reconciled_score", original_priority), original_priority)
    liberty_exception = bool(state.get("liberty_exception_detected", False))
    if liberty_exception:
        lower_bound = max(LIBERTY_PRIORITY_FLOOR, original_priority - SCORE_ADJUSTMENT_LIMIT)
        allowed_increase = LIBERTY_SCORE_ADJUSTMENT_LIMIT if original_priority < LIBERTY_PRIORITY_FLOOR else SCORE_ADJUSTMENT_LIMIT
        upper_bound = min(10.0, max(LIBERTY_PRIORITY_FLOOR, original_priority + allowed_increase))
    else:
        lower_bound = max(0.0, original_priority - SCORE_ADJUSTMENT_LIMIT)
        upper_bound = min(10.0, original_priority + SCORE_ADJUSTMENT_LIMIT)
    summary = state.get("case_summary", "")[:2500]
    bias_score = state.get("bias_score")
    bias_details = state.get("bias_details", "") or ""
    bias_flags = state.get("bias_flags", []) or []
    if isinstance(bias_flags, str):
        bias_flags = [bias_flags]
    elif not isinstance(bias_flags, list):
        bias_flags = [str(bias_flags)]
    bounded_bias_score = _bounded_unit_score(bias_score)
    if bounded_bias_score is None and bias_flags:
        bias_summary = (
            f"No numeric result; text screen supplied these flags: {bias_flags}. "
            "This screen does not evaluate fairness of docket ranking."
        )
    elif bounded_bias_score is None:
        bias_summary = (
            "NO SCORE / RESULT RECORDED (not a pass). Do not infer neutrality or ranking fairness. "
            "Manual review is recommended."
        )
    else:
        bias_summary = (
            f"Automated demographic-language screen score: {bounded_bias_score:.2f}; flags: {bias_flags}; "
            f"details: {bias_details}. This screen does not evaluate fairness of docket ranking."
        )

    prompt = f"""
You are the Judicial Review & Reconciliation Agent presiding over court docket triage.
A potential discrepancy has been detected in this case filing across our AI modules.

DETECTOR REVIEW ALERTS:
- {reasons}

CASE SYNOPSIS:
{summary}

ORIGINAL ML PRIORITY SCORE: {original_priority:.2f} / 10.0
DETERMINISTIC SCORE AFTER GUARDRAILS: {priority:.2f} / 10.0
DEMOGRAPHIC LANGUAGE SCREEN: {bias_summary}
ALLOWED SCORE RANGE: {lower_bound:.2f} to {upper_bound:.2f} / 10.0.
The deterministic liberty/custody guardrail, if triggered, must not be reduced below {LIBERTY_PRIORITY_FLOOR:.1f}.

YOUR TASK:
1. Explain the review alert. Treat demographic-language flags only as wording cautions; they do not establish ranking unfairness.
2. Recommend a score within the allowed range. Preserve the deterministic liberty/custody floor if it was triggered.
3. Formulate a concise Judicial Deliberation Note (3-4 sentences) explaining the reconciliation to the bench judge.

Output strictly valid JSON (no markdown formatting, no code fences):
{{
  "reconciled_score": <float between 0.0 and 10.0>,
  "reconciliation_status": "<'ADJUSTED' or 'FLAGGED_FOR_BENCH'>",
  "deliberation_notes": "<concise deliberation rationale>"
}}
"""
    deliberation_notes = "Automated review signal detected; human review required. No LLM adjustment was applied."
    reconciled_score = priority
    status = "FLAGGED_FOR_BENCH"

    if GEMINI_MODEL:
        for attempt in range(2):
            try:
                res = GEMINI_MODEL.generate_content(prompt).text.strip()
                # Clean code blocks if present
                if res.startswith("```"):
                    lines = res.split("\n")
                    res = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
                parsed = json.loads(res.strip())
                proposed_score = _bounded_score(parsed.get("reconciled_score", priority), priority)
                reconciled_score = round(min(upper_bound, max(lower_bound, proposed_score)), 2)
                status = "FLAGGED_FOR_BENCH"
                proposed_notes = str(parsed.get("deliberation_notes", deliberation_notes))
                if re.search(r"\b(fairness|neutrality|neutral|fair)\b", proposed_notes, flags=re.IGNORECASE):
                    deliberation_notes = (
                        "LLM deliberation completed within score bounds. See review alerts; "
                        "this output makes no demographic or ranking-fairness finding."
                    )
                else:
                    deliberation_notes = proposed_notes
                break
            except Exception as e:
                time.sleep(1)
                continue

    return {
        "reconciled_score": reconciled_score,
        "reconciliation_status": status,
        "deliberation_notes": deliberation_notes
    }


def synthesize_judicial_memo_node(state: CaseDossierState) -> Dict[str, Any]:
    """
    Compose a deterministic triage draft with explicit screen and retrieval limits.
    """
    case_id = str(state.get("case_id", "Case"))
    case_type = str(state.get("case_type", "General") or "General").capitalize()
    priority = _bounded_score(state.get("reconciled_score", state.get("priority_score", 5.0)))
    bias_score = state.get("bias_score")
    bias_details = state.get("bias_details", "") or ""
    bias_flags = state.get("bias_flags", []) or []
    if isinstance(bias_flags, str):
        bias_flags = [bias_flags]
    elif not isinstance(bias_flags, list):
        bias_flags = [str(bias_flags)]
    precedents = state.get("precedents", []) or []
    conflict_reasons = state.get("conflict_reasons", []) or []
    demographic_context_flagged = bool(state.get("demographic_context_flagged", False))
    summary = str(state.get("case_summary", "") or "")[:1200]

    # Scores from semantic retrieval are not calibrated match probabilities.
    # The memo therefore reports retrieval order only, not a percentage match.
    def retrieval_score(precedent):
        for key in ("score", "similarity", "similarity_score"):
            try:
                score = float(precedent.get(key))
                if math.isfinite(score):
                    return score
            except (TypeError, ValueError):
                pass
        return None

    valid_precedents = [p for p in precedents if isinstance(p, dict)]
    valid_precedents.sort(key=lambda p: retrieval_score(p) if retrieval_score(p) is not None else float("-inf"), reverse=True)
    ranked_precedents = []
    for rank, precedent in enumerate(valid_precedents[:3], start=1):
        precedent_case_id = str(precedent.get("case_id", "Precedent record")).replace("_", " ")
        ranked_precedents.append(f"Rank {rank}: {precedent_case_id}")
    precedent_text = (
        "Retrieved precedent candidates (ranked by search system): " + "; ".join(ranked_precedents) + "."
        if ranked_precedents else "No precedent candidates were supplied to reconciliation."
    )

    # Illustrative score bands only; the project has no lawyer-approved listing policy.
    if priority >= 8.0:
        action = "Illustrative suggestion: review for urgent listing (human decision required)"
    elif priority >= 6.0:
        action = "Illustrative suggestion: review for expedited listing (human decision required)"
    elif priority >= 4.0:
        action = "Illustrative suggestion: standard docket review (human decision required)"
    else:
        action = "Illustrative suggestion: routine docket review (human decision required)"

    review_text = (
        "Demographic-language caution: the automated text screen flagged "
        f"{', '.join(str(flag) for flag in bias_flags) if bias_flags else 'a low screen score'}. "
        "This is not an assessment of ranking fairness; manual review is recommended."
        if demographic_context_flagged else
        "No demographic-language flag was supplied. This does not establish fairness of docket ranking."
    )
    bounded_bias_score = _bounded_unit_score(bias_score)
    if bounded_bias_score is None and bias_flags:
        review_text = (
            f"No numeric screen score is available; flagged wording: {', '.join(str(flag) for flag in bias_flags)}. "
            "This screen does not evaluate ranking fairness; manual review is recommended."
        )
    elif bounded_bias_score is None:
        review_text = "No demographic-language screen score/result was recorded; this is not a pass or proof of neutrality. Manual review is recommended."
    else:
        review_text += f" Screen score: {bounded_bias_score:.2f}/1.00. {bias_details}"

    reasons_text = " ".join(str(reason) for reason in conflict_reasons) or "No automated review signal was detected."
    memo = (
        "ILLUSTRATIVE AUTOMATED TRIAGE DRAFT - HUMAN REVIEW REQUIRED.\n"
        f"Case: {case_id} ({case_type}). The supplied synopsis is: {summary or 'No synopsis supplied.'}\n"
        f"Composite priority score: {priority:.2f}/10.0. {action}. This score band is illustrative and is not an approved court listing policy.\n"
        f"Automated review notes: {reasons_text}\n"
        f"Demographic-language screen: {review_text}\n"
        f"{precedent_text} Retrieval rank is not a relevance probability; precedent relevance has not been independently reviewed."
    )

    return {
        "reconciliation_memo": memo,
        "action_recommendation": action
    }


# ==========================================
# 3. Graph Assembly & Class Wrapper
# ==========================================

class JudicialReconciliationAgent:
    """
    LangGraph workflow with a conditional LLM deliberation step.
    It checks priority, explainability signals, and demographic-language flags;
    retrieved precedent candidates are listed by rank but their relevance is not audited.
    """
    def __init__(self):
        print("[INFO] Initializing LangGraph Judicial Reconciliation Agent...")
        self.workflow = StateGraph(CaseDossierState)

        # Register nodes
        self.workflow.add_node("audit_dossier", audit_dossier_node)
        self.workflow.add_node("deliberate_and_reconcile", deliberate_and_reconcile_node)
        self.workflow.add_node("synthesize_judicial_memo", synthesize_judicial_memo_node)

        # Set entry point
        self.workflow.set_entry_point("audit_dossier")

        # Add conditional edges
        self.workflow.add_conditional_edges(
            "audit_dossier",
            route_deliberation,
            {
                "deliberate_and_reconcile": "deliberate_and_reconcile",
                "synthesize_judicial_memo": "synthesize_judicial_memo"
            }
        )

        # Join paths
        self.workflow.add_edge("deliberate_and_reconcile", "synthesize_judicial_memo")
        self.workflow.add_edge("synthesize_judicial_memo", END)

        # Compile graph
        self.app = self.workflow.compile()
        print("[SUCCESS] LangGraph Judicial Reconciliation Agent Compiled Successfully.\n")

    def reconcile(self, dossier: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs deterministic review checks and, when needed, bounded optional LLM deliberation.
        Guaranteed to return a structured reconciliation result without throwing unhandled errors.
        """
        initial_priority = _bounded_score(dossier.get("priority_score", 5.0))
        supplied_bias_flags = dossier.get("bias_flags", []) or []
        if isinstance(supplied_bias_flags, str):
            supplied_bias_flags = [supplied_bias_flags]
        elif not isinstance(supplied_bias_flags, list):
            supplied_bias_flags = [str(supplied_bias_flags)]
        supplied_precedents = dossier.get("precedents", []) or dossier.get("similar_cases", []) or []
        if not isinstance(supplied_precedents, list):
            supplied_precedents = []
        # Populate initial state with defaults
        initial_state: CaseDossierState = {
            "case_id": dossier.get("case_id", "Unknown_Case"),
            "case_type": dossier.get("case_type", "criminal"),
            "legal_regime": dossier.get("legal_regime", "ipc"),
            "case_summary": dossier.get("case_summary") or dossier.get("summary", ""),
            "priority_score": initial_priority,
            "priority_category": dossier.get("priority_category", "Medium"),
            "severity": _bounded_score(dossier.get("severity", 3.0), 3.0),
            "contributions": dossier.get("contributions", {}) or {},
            "bias_score": dossier.get("bias_score"),
            "bias_details": dossier.get("bias_details", ""),
            "bias_flags": supplied_bias_flags,
            "precedents": supplied_precedents,
            "conflict_detected": False,
            "conflict_reasons": [],
            "demographic_context_flagged": False,
            "liberty_exception_detected": False,
            "deliberation_notes": "",
            "reconciliation_status": "APPROVED",
            "reconciled_score": initial_priority,
            "reconciliation_memo": "",
            "action_recommendation": "Standard Docket Scheduling"
        }

        try:
            final_state = self.app.invoke(initial_state)
            return {
                "status": final_state.get("reconciliation_status", "APPROVED"),
                "reconciled_score": final_state.get("reconciled_score", initial_state["priority_score"]),
                "original_priority_score": initial_priority,
                "score_changed": final_state.get("reconciled_score", initial_priority) != initial_priority,
                "conflict_detected": final_state.get("conflict_detected", False),
                "conflict_reasons": final_state.get("conflict_reasons", []),
                "deliberation_notes": final_state.get("deliberation_notes", ""),
                "memo": final_state.get("reconciliation_memo", ""),
                "recommendation": final_state.get("action_recommendation", "Standard Docket Scheduling")
            }
        except Exception as e:
            print(f"[WARNING] LangGraph Reconciliation Exception: {e}. Returning safe fallback.")
            return {
                "status": "FLAGGED_FOR_BENCH",
                "reconciled_score": initial_state["priority_score"],
                "original_priority_score": initial_priority,
                "score_changed": False,
                "conflict_detected": True,
                "conflict_reasons": ["Reconciliation workflow failed; manual review is required."],
                "deliberation_notes": "No reconciliation result is available; no fairness conclusion was made.",
                "memo": "ILLUSTRATIVE AUTOMATED TRIAGE DRAFT — HUMAN REVIEW REQUIRED. The reconciliation workflow failed. No demographic-neutrality or docket-fairness result is available.",
                "recommendation": "Illustrative suggestion only; human review required."
            }


if __name__ == "__main__":
    # Test Stub
    agent = JudicialReconciliationAgent()
    sample_dossier = {
        "case_id": "2024_SC_BAIL_091",
        "case_type": "criminal",
        "legal_regime": "ipc",
        "summary": "Urgent petition seeking habeas corpus relief. Accused has been detained for 6 months without charge.",
        "priority_score": 4.5,
        "priority_category": "Medium",
        "severity": 4.0,
        "contributions": {"immediate_threat_flag": 2.5, "case_age_days": 0.5},
        "bias_score": 0.95,
        "bias_details": "No demographic profiling detected.",
        "bias_flags": [],
        "similar_cases": [{"case_id": "State_vs_Sunil_1998", "score": 0.88}]
    }

    result = agent.reconcile(sample_dossier)
    print("\n--- LangGraph Execution Result ---")
    print(json.dumps(result, indent=2))
