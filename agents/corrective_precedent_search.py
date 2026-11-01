"""Bounded, tool-using LangGraph workflow for precedent retrieval."""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableConfig

from agents.gemini_client import create_gemini_client, generate_gemini_content


class CorrectiveSearchState(TypedDict, total=False):
    query: str
    current_query: str
    expanded_query: str
    candidates: List[Dict[str, Any]]
    final_candidates: List[Dict[str, Any]]
    top_k: int
    action: Dict[str, Any]
    action_history: List[Dict[str, Any]]
    steps: int
    retrieval_calls: int
    statute_expansion_attempted: bool
    statute_expanded: bool
    did_retry: bool
    rewritten_query: str
    rewrite_status: str
    retry_retrieval_status: str
    grading_method: str
    low_grade_fallback: bool
    requested_count: int
    available_candidate_count: int


class CorrectivePrecedentSearch:
    """Plan/act/observe loop with fixed step limits and an allow-listed tool set."""

    MAX_STEPS = 4
    MAX_RETRIEVAL_CALLS = 2
    GRADE_BATCH_SIZE = 15
    STRONG_RELEVANCE = 2 / 3
    ALLOWED_TOOLS = {"expand_statutes", "retrieve", "done"}

    def __init__(self):
        load_dotenv()
        self.model = None
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.ipc_to_bns: Dict[str, str] = {}
        try:
            mapping_path = Path(__file__).resolve().parents[1] / "data" / "mappings.json"
            with mapping_path.open("r", encoding="utf-8") as mapping_file:
                mappings = json.load(mapping_file)
            self.ipc_to_bns = {
                str(ipc).lower(): str(bns)
                for ipc, bns in mappings.get("IPC_TO_BNS", {}).items()
                if isinstance(bns, (str, int))
            }
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Statute cross-reference table unavailable: {exc}")

        if os.getenv("GEMINI_API_KEY"):
            try:
                self.model = create_gemini_client()
            except Exception as exc:
                print(f"Corrective search planner unavailable: {exc}")

        graph = StateGraph(CorrectiveSearchState)
        graph.add_node("plan", self._plan)
        graph.add_node("act", self._act)
        graph.add_node("grade_and_finish", self._grade_and_finish)
        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan", self._route_plan,
            {"act": "act", "finish": "grade_and_finish"},
        )
        graph.add_edge("act", "plan")
        graph.add_edge("grade_and_finish", END)
        self.app = graph.compile()

    def search(
        self,
        query: str,
        initial_candidates: List[Dict[str, Any]],
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        top_k: int,
    ) -> Dict[str, Any]:
        state = self.app.invoke(
            {
                "query": query,
                "current_query": query,
                "expanded_query": query,
                "candidates": initial_candidates,
                "top_k": top_k,
                "action": {"tool": "done", "args": {}},
                "action_history": [],
                "steps": 0,
                "retrieval_calls": 0,
                "statute_expansion_attempted": False,
                "statute_expanded": False,
                "did_retry": False,
                "rewritten_query": "",
                "rewrite_status": "not_attempted",
                "retry_retrieval_status": "not_attempted",
                "grading_method": "Gemini ordinal grader" if self.model else "Lexical fallback grader",
            },
            config={"configurable": {"retrieve_fn": retrieve_fn}},
        )
        return {
            "results": state.get("final_candidates", initial_candidates),
            "retried": state.get("did_retry", False),
            "retry_recommended": state.get("did_retry", False),
            "candidate_count": len(initial_candidates),
            "requested_count": top_k,
            "rewritten_query": state.get("rewritten_query", ""),
            "expanded_query": state.get("expanded_query", query),
            "rewrite_status": state.get("rewrite_status", "not_attempted"),
            "retry_retrieval_status": state.get("retry_retrieval_status", "not_attempted"),
            "grading_method": state.get("grading_method", "Lexical fallback grader"),
            "low_grade_fallback": state.get("low_grade_fallback", False),
            "available_candidate_count": state.get("available_candidate_count", len(initial_candidates)),
            "returned_count": len(state.get("final_candidates", [])),
            "steps": state.get("steps", 0),
            "statute_expanded": state.get("statute_expanded", False),
            "planner_model": self.model_name if self.model else "none",
            "actions": state.get("action_history", []),
        }

    def _plan(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        if state.get("steps", 0) >= self.MAX_STEPS:
            return {"action": {"tool": "done", "args": {}}}
        if state.get("retrieval_calls", 0) >= self.MAX_RETRIEVAL_CALLS and state.get("statute_expansion_attempted"):
            return {"action": {"tool": "done", "args": {}}}
        can_expand = not state.get("statute_expansion_attempted", False) and bool(
            self._expand_statute_terms(state["query"]) != state["query"]
        )
        if can_expand:
            return {"action": {"tool": "expand_statutes", "args": {}}}
        if state.get("expanded_query", state["query"]) != state["query"] and state.get("retrieval_calls", 0) == 0:
            return {"action": {"tool": "retrieve", "args": {"synonyms": ""}}}
        if not self.model:
            return {"action": {"tool": "done", "args": {}}}

        candidate_context = [
            {
                "case_id": item.get("case_id", ""),
                "case_type": item.get("case_type", ""),
                "ipc_sections": item.get("ipc_sections", ""),
                "bns_sections": item.get("bns_sections", ""),
                "summary": str(item.get("summary") or "")[:350],
            }
            for item in state.get("candidates", [])[:5]
        ]
        prompt = f"""You are planning a small number of precedent-search actions. Choose only one tool from the allow-list.
Tools:
- expand_statutes: add cross-references from the project's mapping table. Use only when available.
- retrieve: request one more retrieval using issue-level synonyms. Provide synonyms only; never add statute numbers, facts, citations, or authorities.
- done: stop searching.
Prefer done when the current candidates already look useful. At most {self.MAX_RETRIEVAL_CALLS} additional retrievals are allowed.
Return JSON only: {{"tool":"expand_statutes|retrieve|done","args":{{"synonyms":"..."}}}}. expand_statutes and done must have empty args.

Original query: {state['query']}
Current query: {state.get('current_query', state['query'])}
Project-table statute expansion available: {str(can_expand).lower()}
Statute expansion already attempted: {str(state.get('statute_expansion_attempted', False)).lower()}
Additional retrievals used: {state.get('retrieval_calls', 0)}
Candidates so far: {json.dumps(candidate_context, ensure_ascii=False)}"""
        try:
            from google.genai import types
            response = generate_gemini_content(
                self.model,
                prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            action = json.loads(response.text.strip())
            if not isinstance(action, dict) or action.get("tool") not in self.ALLOWED_TOOLS:
                raise ValueError("Planner selected a tool outside the allow-list")
            args = action.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("Planner arguments must be a JSON object")
            tool = action["tool"]
            if tool in {"expand_statutes", "done"} and args:
                raise ValueError(f"{tool} does not accept arguments")
            if tool == "expand_statutes" and not can_expand:
                return {"action": {"tool": "done", "args": {}}}
            if tool == "retrieve":
                if set(args) != {"synonyms"} or not isinstance(args["synonyms"], str):
                    raise ValueError("retrieve accepts only a synonyms string")
                synonyms = self._sanitize_synonyms(args["synonyms"])
                if (not synonyms and not (state.get("expanded_query") != state.get("query"))) or state.get("retrieval_calls", 0) >= self.MAX_RETRIEVAL_CALLS:
                    return {"action": {"tool": "done", "args": {}}}
                action = {"tool": "retrieve", "args": {"synonyms": synonyms}}
            return {"action": action}
        except Exception as exc:
            print(f"Corrective search planner failed; finishing with current candidates: {exc}")
            return {"action": {"tool": "done", "args": {}}}

    @staticmethod
    def _route_plan(state: CorrectiveSearchState) -> str:
        action = state.get("action", {})
        if state.get("steps", 0) >= CorrectivePrecedentSearch.MAX_STEPS:
            return "finish"
        return "act" if action.get("tool") in CorrectivePrecedentSearch.ALLOWED_TOOLS - {"done"} else "finish"

    def _act(self, state: CorrectiveSearchState, config: RunnableConfig) -> Dict[str, Any]:
        action = state.get("action", {})
        tool = action.get("tool")
        args = action.get("args", {})
        history = list(state.get("action_history", []))
        update: Dict[str, Any] = {"steps": state.get("steps", 0) + 1}

        if tool == "expand_statutes":
            expanded = self._expand_statute_terms(state["query"])
            history.append({"tool": tool, "status": "completed" if expanded != state["query"] else "no_mapping"})
            update.update({
                "expanded_query": expanded,
                "statute_expansion_attempted": True,
                "statute_expanded": expanded != state["query"],
                "rewrite_status": "ready" if expanded != state["query"] else state.get("rewrite_status", "not_attempted"),
                "rewritten_query": expanded if expanded != state["query"] else state.get("rewritten_query", ""),
                "action_history": history,
                "action": {"tool": "done", "args": {}},
            })
            return update

        if tool != "retrieve" or set(args) != {"synonyms"}:
            history.append({"tool": str(tool), "status": "rejected"})
            update.update({"action_history": history, "action": {"tool": "done", "args": {}}})
            return update

        synonyms = self._sanitize_synonyms(args.get("synonyms", ""))
        base_query = state.get("expanded_query", state["query"])
        query = " ".join(part for part in (base_query, synonyms) if part).strip()[:500]
        if (not synonyms and base_query == state["query"]) or (
            query.lower() == state.get("current_query", "").lower()
            and state.get("retrieval_calls", 0) > 0
        ):
            history.append({"tool": "retrieve", "status": "duplicate_or_empty_query"})
            update.update({"action_history": history, "action": {"tool": "done", "args": {}}})
            return update

        try:
            retrieve_fn = config.get("configurable", {}).get("retrieve_fn")
            if not callable(retrieve_fn):
                raise ValueError("Retry retrieval callback is unavailable")
            found = retrieve_fn(query)
            candidates = self._merge_candidates(state.get("candidates", []), found)
            history.append({"tool": "retrieve", "status": "completed", "candidate_count": len(found)})
            update.update({
                "candidates": candidates,
                "current_query": query,
                "rewritten_query": query,
                "rewrite_status": "ready",
                "retry_retrieval_status": "completed",
                "retrieval_calls": state.get("retrieval_calls", 0) + 1,
                "did_retry": True,
                "action_history": history,
                "action": {"tool": "done", "args": {}},
            })
        except Exception as exc:
            print(f"Corrective search retry retrieval failed: {exc}")
            history.append({"tool": "retrieve", "status": "failed"})
            update.update({
                "retry_retrieval_status": "failed",
                "action_history": history,
                "action": {"tool": "done", "args": {}},
            })
        return update

    @staticmethod
    def _merge_candidates(existing: List[Dict[str, Any]], added: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for candidate in existing + added:
            if not isinstance(candidate, dict):
                continue
            case_id = str(candidate.get("case_id", ""))
            if not case_id:
                continue
            if case_id not in by_id:
                by_id[case_id] = candidate
                order.append(case_id)
            else:
                try:
                    if float(candidate.get("score", 0.0)) > float(by_id[case_id].get("score", 0.0)):
                        by_id[case_id] = candidate
                except (TypeError, ValueError):
                    pass
        return [by_id[case_id] for case_id in order]

    def _grade_and_finish(self, state: CorrectiveSearchState) -> Dict[str, Any]:
        candidates, used_gemini = self._grade_candidates(state["query"], state.get("candidates", []))
        method = "Gemini ordinal grader" if used_gemini else "Lexical fallback grader"
        target_count = max(1, state.get("top_k", 10))
        low_grade_fallback = False
        if not used_gemini:
            ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
        else:
            ranked = [
                item for item in candidates
                if item.get("relevance_score") is None
                or item.get("relevance_score", 0.0) >= self.STRONG_RELEVANCE
            ]
            ranked.sort(
                key=lambda item: (
                    item.get("relevance_score") is not None,
                    item.get("relevance_score") or 0.0,
                    item.get("score", 0.0),
                ),
                reverse=True,
            )
            if len(ranked) < target_count:
                selected_ids = {str(item.get("case_id", "")) for item in ranked}
                below_cutoff = sorted(
                    (
                        item for item in candidates
                        if str(item.get("case_id", "")) not in selected_ids
                    ),
                    key=lambda item: item.get("score", 0.0),
                    reverse=True,
                )
                fill_count = min(target_count - len(ranked), len(below_cutoff))
                if fill_count:
                    ranked.extend(below_cutoff[:fill_count])
                    low_grade_fallback = True
        if state.get("retry_retrieval_status") == "failed" and not ranked:
            ranked = sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)
        available_count = len(candidates)
        return {
            "final_candidates": ranked[:target_count],
            "grading_method": method,
            "low_grade_fallback": low_grade_fallback,
            "available_candidate_count": available_count,
        }

    def _grade_candidates(self, query: str, candidates: List[Dict[str, Any]]):
        if not candidates:
            return [], False
        if self.model:
            try:
                from google.genai import types
                grades: Dict[str, Dict[str, Any]] = {}
                for start in range(0, len(candidates), self.GRADE_BATCH_SIZE):
                    batch = candidates[start:start + self.GRADE_BATCH_SIZE]
                    context = [
                        {
                            "case_id": item.get("case_id", ""),
                            "case_type": item.get("case_type", ""),
                            "legal_regime": item.get("legal_regime", ""),
                            "ipc_sections": item.get("ipc_sections", ""),
                            "bns_sections": item.get("bns_sections", ""),
                            "summary": str(item.get("summary") or "")[:800],
                        }
                        for item in batch
                    ]
                    prompt = f"""Grade each Indian legal precedent candidate for usefulness to the ORIGINAL user query using only the supplied metadata and summary. Consider legal issue, material facts, procedural posture, and applicable provisions. Mere keyword overlap is insufficient. Do not infer whether a case remains good law; subsequent treatment is not supplied.

Use this ordinal rubric: 0 = unrelated; 1 = weak/partial issue overlap; 2 = relevant on a material legal issue; 3 = highly relevant on issue and posture. Return JSON only: {{"grades":[{{"case_id":"exact supplied id","score":0,"reason":"brief reason grounded in supplied data"}}]}}. Include one grade per candidate in this batch.

ORIGINAL QUERY: {query}
CANDIDATES: {json.dumps(context, ensure_ascii=False)}"""
                    response = generate_gemini_content(
                        self.model,
                        prompt,
                        config=types.GenerateContentConfig(response_mime_type="application/json"),
                    )
                    parsed = json.loads(response.text.strip())
                    grades.update({
                        str(row.get("case_id")): row
                        for row in parsed.get("grades", [])
                        if isinstance(row, dict) and row.get("case_id") is not None
                    })
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
                item.update({
                    "relevance_score": None,
                    "relevance_grade": None,
                    "relevance_label": "Ungraded",
                    "relevance_reason": "No grader assessment was returned; this result was not treated as irrelevant.",
                })
                continue
            try:
                ordinal_score = int(grade.get("score"))
                if ordinal_score not in (0, 1, 2, 3):
                    raise ValueError("Ordinal grade must be from 0 to 3")
                score = ordinal_score / 3
            except (ValueError, TypeError):
                ordinal_score, score = None, None
            item["relevance_score"] = score
            item["relevance_grade"] = ordinal_score
            item["relevance_label"] = CorrectivePrecedentSearch._label(score)
            item["relevance_reason"] = str(grade.get("reason") or "No relevance explanation was returned.")[:400]

    @staticmethod
    def _lexical_grade(query: str, candidates: List[Dict[str, Any]]) -> None:
        stop_words = {"about", "after", "against", "and", "before", "from", "into", "that", "the", "their", "this", "under", "with"}
        terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2 and t not in stop_words}
        for item in candidates:
            context = " ".join(str(item.get(k) or "") for k in ("case_id", "summary", "ipc_sections", "bns_sections")).lower()
            matched = sorted(term for term in terms if term in context)
            score = len(matched) / max(len(terms), 1)
            item["relevance_score"] = round(score, 2)
            item["relevance_grade"] = None
            item["relevance_label"] = CorrectivePrecedentSearch._label(score)
            item["relevance_reason"] = (
                f"Lexical fallback found query terms: {', '.join(matched[:5])}."
                if matched else "Lexical fallback found no distinctive query terms in the case metadata or summary."
            )

    @staticmethod
    def _label(score: float | None) -> str:
        if score is None:
            return "Ungraded"
        if score >= 1.0:
            return "Strong"
        if score >= CorrectivePrecedentSearch.STRONG_RELEVANCE:
            return "Relevant"
        if score > 0:
            return "Partial"
        return "Weak"

    def _expand_statute_terms(self, query: str) -> str:
        references = []
        section_patterns = [
            ("ipc", re.compile(r"\b(?:ipc|indian penal code)\s*(?:section|sec\.?|s\.?\s*)?(\d+[a-z]?)\b", re.I)),
            ("ipc", re.compile(r"\b(?:section|sec\.?)\s*(\d+[a-z]?)\s*(?:of\s+)?(?:ipc|indian penal code)\b", re.I)),
            ("bns", re.compile(r"\b(?:bns|bharatiya nyaya sanhita)\s*(?:section|sec\.?|s\.?\s*)?(\d+[a-z]?)\b", re.I)),
            ("bns", re.compile(r"\b(?:section|sec\.?)\s*(\d+[a-z]?)\s*(?:of\s+)?(?:bns|bharatiya nyaya sanhita)\b", re.I)),
        ]
        reverse_groups: Dict[str, List[str]] = {}
        for ipc, bns in self.ipc_to_bns.items():
            reverse_groups.setdefault(bns.lower(), []).append(ipc)
        reverse = {bns: sections[0] for bns, sections in reverse_groups.items() if len(sections) == 1}
        for regime, pattern in section_patterns:
            for match in pattern.finditer(query):
                section = match.group(1).lower()
                if regime == "ipc" and section in self.ipc_to_bns:
                    references.append(f"BNS Section {self.ipc_to_bns[section]}")
                elif regime == "bns" and section in reverse:
                    references.append(f"IPC Section {reverse[section]}")
        additions = list(dict.fromkeys(references))
        return query if not additions else f"{query} {' '.join(additions)}"

    @staticmethod
    def _sanitize_synonyms(value: str) -> str:
        synonyms = re.sub(r"[\r\n\x00-\x1f]+", " ", str(value or ""))
        synonyms = re.sub(
            r"\b(?:IPC|BNS|Indian Penal Code|Bharatiya Nyaya Sanhita)\s+(?:(?:section|sec\.?)\s*)?\d+[A-Za-z]?\b",
            " ", synonyms, flags=re.I,
        )
        synonyms = re.sub(r"\b(?:section|sec\.?)\s*\d+[A-Za-z]?\b", " ", synonyms, flags=re.I)
        synonyms = re.sub(r"\s+", " ", synonyms).strip()
        return synonyms[:300]
