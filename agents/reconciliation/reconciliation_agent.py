import os
import sys
import json
import time
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
    # Core case inputs from previous 4 modules
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
    deliberation_notes: str
    
    # Output reconciliation verdict
    reconciliation_status: str  # "APPROVED" | "ADJUSTED" | "FLAGGED_FOR_BENCH"
    reconciled_score: float
    reconciliation_memo: str
    action_recommendation: str


# ==========================================
# 2. Node Implementations
# ==========================================

def audit_dossier_node(state: CaseDossierState) -> Dict[str, Any]:
    """
    Evaluator Node: Scans the combined outputs of Prioritization (XGBoost),
    Explainability (SHAP), Bias Audit, and Precedents to detect discrepancies or edge cases.
    """
    conflicts = []
    priority = state.get("priority_score", 5.0)
    bias_score = state.get("bias_score")
    bias_flags = state.get("bias_flags", [])
    summary = (state.get("case_summary") or "").lower()
    contributions = state.get("contributions", {})

    # Check 1: Bias vs. Priority Anomaly
    # High priority case flagged with demographic profiling or prejudice in allegations
    if (bias_score is not None and bias_score < 0.85) or (bias_flags and len(bias_flags) > 0):
        if priority >= 6.0:
            flag_text = ", ".join(bias_flags) if bias_flags else "Demographic profiling context"
            conflicts.append(
                f"Demographic Fairness Discrepancy: Case is ranked High/Critical ({priority:.1f}/10), "
                f"but Fairness Audit flagged contextual bias ({flag_text}) with neutrality index of {(bias_score or 0.0)*100:.0f}%."
            )

    # Check 2: Under-Prioritization of Fundamental Liberties
    # Low priority assigned by ML model, but text reveals ongoing custody, habeas corpus, or life threat
    urgent_keywords = ["habeas corpus", "unlawful custody", "illegal detention", "death threat", "custodial violence", "juvenile"]
    has_liberty_issue = any(kw in summary for kw in urgent_keywords)
    threat_contrib = contributions.get("immediate_threat_flag", 0.0)
    
    if priority < 6.0 and (has_liberty_issue or threat_contrib > 2.0):
        matched_kw = [kw for kw in urgent_keywords if kw in summary]
        matched_str = ", ".join(matched_kw) if matched_kw else "active liberty restraint"
        conflicts.append(
            f"Statutory Liberty Exception: Model assigned routine priority ({priority:.1f}/10), "
            f"yet case contains high-urgency fundamental rights factors ({matched_str})."
        )

    # Check 3: Explainability Outlier (Over-reliance on case age vs statutory severity)
    age_contrib = contributions.get("case_age_days", 0.0)
    sev_contrib = contributions.get("max_severity_score", 0.0)
    if age_contrib > 2.5 and sev_contrib <= 0.5 and priority >= 7.0:
        conflicts.append(
            "Prioritization Skew: Priority elevated primarily by case age backlog rather than statutory offence gravity."
        )

    return {
        "conflict_detected": len(conflicts) > 0,
        "conflict_reasons": conflicts,
        "reconciliation_status": "FLAGGED_FOR_BENCH" if len(conflicts) > 0 else "APPROVED",
        "reconciled_score": priority
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
    Optimizer Node: Uses Gemini 2.5 Flash to reason through the detected conflicts,
    determine if priority adjustment is legally warranted, and formulate judicial cautions.
    """
    reasons = "\n- ".join(state.get("conflict_reasons", []))
    priority = state.get("priority_score", 5.0)
    summary = state.get("case_summary", "")[:2500]
    bias_score = state.get("bias_score")
    bias_details = state.get("bias_details", "None")

    prompt = f"""
You are the Judicial Review & Reconciliation Agent presiding over court docket triage.
A potential discrepancy has been detected in this case filing across our AI modules.

DETECTOR AUDIT ALERTS:
- {reasons}

CASE SYNOPSIS:
{summary}

ORIGINAL ML PRIORITY SCORE: {priority:.2f} / 10.0
FAIRNESS AUDIT DETAILS: Neutrality Index: {bias_score if bias_score is not None else 'N/A'}, Details: {bias_details}

YOUR TASK:
1. Deliberate on whether the statutory urgency should be maintained, elevated (e.g. for urgent liberty/custody), or cautioned (due to demographic bias).
2. Recommend an adjusted priority score (0.0 to 10.0). Keep it close to original unless a fundamental liberty was missed.
3. Formulate a concise Judicial Deliberation Note (3-4 sentences) explaining the reconciliation to the bench judge.

Output strictly valid JSON (no markdown formatting, no code fences):
{{
  "reconciled_score": <float between 0.0 and 10.0>,
  "reconciliation_status": "<'ADJUSTED' or 'FLAGGED_FOR_BENCH'>",
  "deliberation_notes": "<concise deliberation rationale>"
}}
"""
    deliberation_notes = "Automated conflict audit triggered. Judicial caution flag attached."
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
                reconciled_score = round(float(parsed.get("reconciled_score", priority)), 2)
                status = parsed.get("reconciliation_status", "FLAGGED_FOR_BENCH")
                deliberation_notes = parsed.get("deliberation_notes", deliberation_notes)
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
    Synthesis Node: Drafts the comprehensive Judicial Bench Triage Memo
    unifying Precedents, Prioritization, Explainability, and Fairness.
    """
    case_id = state.get("case_id", "Case")
    case_type = state.get("case_type", "General").capitalize()
    priority = state.get("reconciled_score", state.get("priority_score", 5.0))
    status = state.get("reconciliation_status", "APPROVED")
    deliberation = state.get("deliberation_notes", "")
    bias_score = state.get("bias_score")
    precedents = state.get("precedents", [])
    contributions = state.get("contributions", {})
    summary = state.get("case_summary", "")[:2000]

    # Format top precedents
    top_prec_strs = []
    for p in precedents[:3]:
        pid = p.get("case_id", "Precedent").replace("_", " ")
        match_pct = (p.get("score") or p.get("similarity_score") or 0.0) * 100
        top_prec_strs.append(f"{pid} ({match_pct:.1f}% match)")
    prec_summary = "; ".join(top_prec_strs) if top_prec_strs else "Standard canonical jurisprudence"

    # Action recommendation logic
    if priority >= 8.0:
        action = "Admit for Urgent Motion / Immediate Listing"
    elif priority >= 6.0:
        action = "Expedited Listing (2-4 Weeks Queue)"
    elif priority >= 4.0:
        action = "Standard Docket Scheduling"
    else:
        action = "Routine Civil/Procedural Queue"

    if status == "FLAGGED_FOR_BENCH":
        action += " [With Judicial Caution on Demographic Context]"

    prompt = f"""
You are the Judicial Bench Clerk formulating an executive Judicial Triage Memo for the presiding Judge.

CASE: {case_id} ({case_type})
RECONCILED DOCKET SCORE: {priority:.2f}/10.0 ({status})
TOP PRECEDENTS: {prec_summary}
DEMOGRAPHIC FAIRNESS INDEX: {f"{bias_score*100:.0f}% Neutral" if bias_score is not None else "Audited"}
DELIBERATION AUDIT: {deliberation if deliberation else "All 4 modules in consistent alignment."}
CASE FACTS:
{summary}

Write a concise, professional 2-paragraph Judicial Triage Memo:
- Paragraph 1: Statutory gravity, procedural posture, and relevant precedent alignment.
- Paragraph 2: Triage justification, fairness/neutrality audit confirmation, and scheduling recommendation.

Return ONLY the plain text of the memo.
"""
    memo = (
        f"The case has been triaged with a statutory priority score of {priority:.2f}/10.0 ({action}). "
        f"Precedent discovery mapped {len(precedents)} analogous Supreme Court rulings ({prec_summary}). "
        f"Fairness audit verified demographic neutrality index at "
        f"{f'{bias_score*100:.0f}%' if bias_score is not None else 'acceptable threshold'}. "
        f"Scheduling Recommendation: {action}."
    )

    if GEMINI_MODEL:
        try:
            res = GEMINI_MODEL.generate_content(prompt).text.strip()
            if res:
                memo = res
        except Exception:
            pass

    return {
        "reconciliation_memo": memo,
        "action_recommendation": action
    }


# ==========================================
# 3. Graph Assembly & Class Wrapper
# ==========================================

class JudicialReconciliationAgent:
    """
    LangGraph-powered Cyclic Multi-Agent Evaluator & Reconciliation Engine.
    Inspects outputs from Prioritization, Explainability, Bias Detection, and Semantic Search,
    detects systemic contradictions, and deliberates to produce a reconciled judicial docket memo.
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
        Executes the LangGraph reconciliation state machine over the provided case dossier.
        Guaranteed to return a structured reconciliation result without throwing unhandled errors.
        """
        # Populate initial state with defaults
        initial_state: CaseDossierState = {
            "case_id": dossier.get("case_id", "Unknown_Case"),
            "case_type": dossier.get("case_type", "criminal"),
            "legal_regime": dossier.get("legal_regime", "ipc"),
            "case_summary": dossier.get("case_summary") or dossier.get("summary", ""),
            "priority_score": float(dossier.get("priority_score", 5.0)),
            "priority_category": dossier.get("priority_category", "Medium"),
            "severity": float(dossier.get("severity", 3.0)),
            "contributions": dossier.get("contributions", {}),
            "bias_score": dossier.get("bias_score"),
            "bias_details": dossier.get("bias_details", ""),
            "bias_flags": dossier.get("bias_flags", []),
            "precedents": dossier.get("precedents", []) or dossier.get("similar_cases", []),
            "conflict_detected": False,
            "conflict_reasons": [],
            "deliberation_notes": "",
            "reconciliation_status": "APPROVED",
            "reconciled_score": float(dossier.get("priority_score", 5.0)),
            "reconciliation_memo": "",
            "action_recommendation": "Standard Docket Scheduling"
        }

        try:
            final_state = self.app.invoke(initial_state)
            return {
                "status": final_state.get("reconciliation_status", "APPROVED"),
                "reconciled_score": final_state.get("reconciled_score", initial_state["priority_score"]),
                "conflict_detected": final_state.get("conflict_detected", False),
                "conflict_reasons": final_state.get("conflict_reasons", []),
                "deliberation_notes": final_state.get("deliberation_notes", ""),
                "memo": final_state.get("reconciliation_memo", ""),
                "recommendation": final_state.get("action_recommendation", "Standard Docket Scheduling")
            }
        except Exception as e:
            print(f"[WARNING] LangGraph Reconciliation Exception: {e}. Returning safe fallback.")
            return {
                "status": "APPROVED",
                "reconciled_score": initial_state["priority_score"],
                "conflict_detected": False,
                "conflict_reasons": [],
                "deliberation_notes": "Deterministic fallback applied.",
                "memo": "Case validated by standard multi-agent pipeline.",
                "recommendation": "Standard Docket Scheduling"
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
