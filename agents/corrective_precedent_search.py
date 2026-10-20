"""Bounded corrective retrieval workflow for precedent search."""

import json
import os
import re
from typing import Any, Callable, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from agents.gemini_client import create_gemini_client, generate_gemini_content


class CorrectiveSearchState(TypedDict, total=False):
    query: str
    current_query: str
    candidates: List[Dict[str, Any]]
    first_pass_candidates: List[Dict[str, Any]]
    final_candidates: List[Dict[str, Any]]
    retrieve_fn: Callable[[str], List[Dict[str, Any]]]
    top_k: int
    did_retry: bool
    rewritten_query: str
    rewrite_status: str
    retry_retrieval_status: str
    grading_method: str


class CorrectivePrecedentSearch:
    """Grade first-pass results and allow at most one Gemini-guided retry."""

    STRONG_RELEVANCE = 0.65

    def __init__(self):
        load_dotenv()
        self.model = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                self.model = create_gemini_client()
            except Exception as exc:
                print(f"Corrective search grader unavailable: {exc}")

        graph = StateGraph(CorrectiveSearchState)
        graph.add_node("grade_first_pass", self._grade_first_pass)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("retrieve_rewritten", self._retrieve_rewritten)
        graph.add_node("grade_retry", self._grade_retry)
        graph.add_node("finish_first_pass", self._finish_first_pass)
        graph.add_node("merge_and_finish", self._merge_and_finish)
        graph.set_entry_point("grade_first_pass")
        graph.add_conditional_edges(
            "grade_first_pass",
            self._should_retry,
            {"retry": "rewrite_query", "finish": "finish_first_pass"},
        )
        graph.add_edge("rewrite_query", "retrieve_rewritten")
        graph.add_edge("retrieve_rewritten", "grade_retry")
        graph.add_edge("grade_retry", "merge_and_finish")
        graph.add_edge("finish_first_pass", END)
        graph.add_edge("merge_and_finish", END)
        self.app = graph.compile()

    def search(
        self,
        query: str,
        initial_candidates: List[Dict[str, Any]],
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        top_k: int,
    ) -> Dict[str, Any]:
        state = self.app.invoke({
            "query": query,
            "current_query": query,
            "candidates": initial_candidates,
            "retrieve_fn": retrieve_fn,
            "top_k": top_k,
            "did_retry": False,
            "rewritten_query": "",
            "rewrite_status": "not_attempted",
            "retry_retrieval_status": "not_attempted",
            "grading_method": "Gemini relevance grader" if self.model else "Lexical fallback grader",
        })
        return {
            "results": state.get("final_candidates", initial_candidates),
            "retried": state.get("did_retry", False),
            "retry_recommended": self._needs_retry(initial_candidates),
            "candidate_count": len(initial_candidates),
            "rewritten_query": state.get("rewritten_query", ""),
            "rewrite_status": state.get("rewrite_status", "not_attempted"),
            "retry_retrieval_status": state.get("retry_retrieval_status", "not_attempted"),
            "grading_method": state.get("grading_method", "Lexical fallback grader"),
            "returned_count": len(state.get("final_candidates", [])),
        }

    def _needs_retry(self, candidates: List[Dict[str, Any]]) -> bool:
        required = min(2, len(candidates))
        return bool(candidates) and sum(
            1 for item in candidates if item.get("relevance_score", 0.0) >= self.STRONG_RELEVANCE
        ) < required

    def _grade_candidates(self, query: str, candidates: List[Dict[str, Any]]):
        if not candidates:
            return [], False
        if self.model:
            try:
                candidate_context = [
                    {
                        "case_id": item.get("case_id", ""),
                        "case_type": item.get("case_type", ""),
                        "legal_regime": item.get("legal_regime", ""),
                        "ipc_sections": item.get("ipc_sections", ""),
                        "bns_sections": item.get("bns_sections", ""),
                        "summary": (item.get("summary") or "")[:800],
                    }
                    for item in candidates[:15]
                ]
                prompt = f"""You are grading Indian legal precedent search results. Assess how useful each case is for the user's legal research query based only on the supplied metadata and summary.
Consider overlap in legal issue, material facts, procedural posture, and applicable or equivalent statutory provisions. Do not require identical section numbers. Do not infer that a case is still good law; the data does not provide subsequent-history status. Do not reward mere keyword overlap when the legal issue differs.

Query: {query}
Candidates (JSON): {json.dumps(candidate_context, ensure_ascii=False)}

Return only JSON in this shape: {{"grades":[{{"case_id":"exact supplied id","score":0.0,"reason":"one concise sentence grounded in supplied information"}}]}}. Score is 0 to 1. Include one grade per supplied candidate."""
                response = generate_gemini_content(self.model, prompt)
                raw = response.text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
                parsed = json.loads(raw)
                grades = {
                    str(row.get("case_id")): row
                    for row in parsed.get("grades", [])
                    if isinstance(row, dict) and row.get("case_id") is not None
                }
                candidate_ids = {str(item.get("case_id")) for item in candidates}
                if candidate_ids.intersection(grades):
                    self._attach_grades(candidates, grades)
                    return candidates, True
            except Exception as exc:
                print(f"Corrective search Gemini grading failed; using lexical fallback: {exc}")

        self._lexical_grade(query, candidates)
        return candidates, False

    @staticmethod
    def _attach_grades(candidates: List[Dict[str, Any]], grades: Dict[str, Dict[str, Any]]) -> None:
        for item in candidates:
            grade = grades.get(str(item.get("case_id")))
            if not grade:
                item["relevance_score"] = 0.0
                item["relevance_reason"] = "The grader returned no assessment for this result."
            else:
                try:
                    score = float(grade.get("score", 0.0))
                except (ValueError, TypeError):
                    score = 0.0
                item["relevance_score"] = min(max(score, 0.0), 1.0)
                item["relevance_reason"] = str(grade.get("reason") or "No relevance explanation was returned.")[:400]
            item["relevance_label"] = CorrectivePrecedentSearch._label(item["relevance_score"])

    @staticmethod
    def _lexical_grade(query: str, candidates: List[Dict[str, Any]]) -> None:
        stop_words = {"about", "after", "against", "and", "before", "from", "into", "that", "the", "their", "this", "under", "with"}
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2 and t not in stop_words}
        for item in candidates:
            context = " ".join(str(item.get(k) or "") for k in ("case_id", "summary", "ipc_sections", "bns_sections")).lower()
            matched = sorted(term for term in terms if term in context)
            score = len(matched) / max(len(terms), 1)
            item["relevance_score"] = round(score, 2)
            item["relevance_label"] = CorrectivePrecedentSearch._label(score)
            item["relevance_reason"] = (
                f"Lexical fallback found query terms: {', '.join(matched[:5])}."
                if matched else "Lexical fallback found no distinctive query terms in the case metadata or summary."
            )

    @staticmethod
    def _label(score: float) -> str:
        if score >= 0.75:
            return "Strong"
        if score >= CorrectivePrecedentSearch.STRONG_RELEVANCE:
            return "Partial"
        return "Weak"

    def _grade_first_pass(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        candidates, used_gemini = self._grade_candidates(state["query"], state.get("candidates", []))
        return {
            "candidates": candidates,
            "first_pass_candidates": candidates,
            "grading_method": "Gemini relevance grader" if used_gemini else "Lexical fallback grader",
        }

    def _should_retry(self, state: CorrectiveSearchState) -> str:
        candidates = state.get("candidates", [])
        required = min(2, len(candidates))
        adequate = sum(1 for item in candidates if item.get("relevance_score", 0.0) >= self.STRONG_RELEVANCE) >= required
        return "retry" if self.model and candidates and not adequate else "finish"

    def _rewrite_query(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        prompt = f"""Rewrite this Indian legal research query once to improve precedent retrieval. Preserve the user's issue and facts. Add only closely related legal synonyms or equivalent statutory terminology when justified by the original wording. Do not invent facts, statutes, or citations. Keep it concise and return only JSON: {{"rewritten_query":"..."}}

Original query: {state['query']}"""
        try:
            response = generate_gemini_content(self.model, prompt)
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            rewritten = str(json.loads(raw).get("rewritten_query", "")).strip()
        except Exception as exc:
            print(f"Corrective search query rewrite failed: {exc}")
            return {"rewritten_query": "", "rewrite_status": "failed"}
        rewritten = rewritten[:500]
        if not rewritten:
            status = "failed"
        elif rewritten.lower() == state["query"].strip().lower():
            status = "unchanged"
        else:
            status = "ready"
        return {"rewritten_query": rewritten, "rewrite_status": status}

    def _retrieve_rewritten(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        query = state.get("rewritten_query", "")
        if not query or query.lower() == state["query"].lower():
            return {"candidates": [], "did_retry": False, "retry_retrieval_status": "not_attempted"}
        try:
            candidates = state["retrieve_fn"](query)
        except Exception as exc:
            print(f"Corrective search retry retrieval failed: {exc}")
            return {
                "candidates": [],
                "current_query": query,
                "did_retry": False,
                "retry_retrieval_status": "failed",
            }
        return {
            "candidates": candidates,
            "current_query": query,
            "did_retry": True,
            "retry_retrieval_status": "completed",
        }

    def _grade_retry(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        if not state.get("candidates"):
            return {"candidates": []}
        candidates, used_gemini = self._grade_candidates(state.get("rewritten_query", ""), state.get("candidates", []))
        first_method = state.get("grading_method", "")
        retry_method = "Gemini relevance grader" if used_gemini else "Lexical fallback grader"
        if retry_method == first_method:
            method = retry_method
        else:
            method = f"{first_method} + {retry_method}"
        return {"candidates": candidates, "grading_method": method}

    @staticmethod
    def _finish_first_pass(state: CorrectiveSearchState) -> Dict[str, Any]:
        candidates = state.get("first_pass_candidates", state.get("candidates", []))
        top_k = state.get("top_k", 10)
        grading_method = state.get("grading_method", "")
        if "Gemini relevance grader" not in grading_method:
            # Lexical matching is only a fallback signal, not a safe reason to
            # discard vector-retrieved precedents when Gemini is unavailable.
            ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
            return {"final_candidates": ranked[:top_k]}
        qualified = [item for item in candidates if item.get("relevance_score", 0.0) >= CorrectivePrecedentSearch.STRONG_RELEVANCE]
        qualified.sort(key=lambda item: (item.get("relevance_score", 0.0), item.get("score", 0.0)), reverse=True)
        return {"final_candidates": qualified[:top_k]}

    @staticmethod
    def _merge_and_finish(state: CorrectiveSearchState) -> Dict[str, Any]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in state.get("first_pass_candidates", []) + state.get("candidates", []):
            case_id = str(item.get("case_id", ""))
            existing = merged.get(case_id)
            if existing is None or item.get("relevance_score", 0.0) > existing.get("relevance_score", 0.0):
                merged[case_id] = item
        ranked = sorted(
            merged.values(),
            key=lambda item: (item.get("relevance_score", 0.0), item.get("score", 0.0)),
            reverse=True,
        )
        fallback_used = (
            "Lexical fallback grader" in state.get("grading_method", "")
            or state.get("rewrite_status") == "failed"
            or state.get("retry_retrieval_status") == "failed"
        )
        if fallback_used:
            # Keep original retrieval results if an LLM-dependent step fails.
            ranked_by_retrieval = sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)
            return {"final_candidates": ranked_by_retrieval[:state.get("top_k", 10)]}
        qualified = [item for item in ranked if item.get("relevance_score", 0.0) >= CorrectivePrecedentSearch.STRONG_RELEVANCE]
        return {"final_candidates": qualified[:state.get("top_k", 10)]}
