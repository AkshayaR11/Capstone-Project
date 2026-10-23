# Judicial Decision Support System (JDSS): Multi-Agent Architecture for Backlog Triage and Precedent Retrieval

**Authors / Capstone Project ID:** Project 97  
**Department:** Computer Science and Engineering  
**Supervisor:** Dr. Saritha P  

---

## Abstract
The Indian judicial system faces a systemic bottleneck with over 40 million pending cases across various tiers of the judiciary. Traditional case listing follows First-In-First-Out (FIFO) or ad-hoc recency-biased (LIFO) scheduling, often causing high-severity, life-and-liberty cases to linger for months alongside low-impact disputes. Additionally, legal research is complicated by India's transition from the colonial **Indian Penal Code (IPC)** to the modern **Bharatiya Nyaya Sanhita (BNS)**. 

In this work, we propose an **Agentic AI-Driven Judicial Decision Support System** powered by a multi-agent orchestration architecture utilizing **LangGraph**, **Legal-BERT**, and explainable machine learning (XGBoost + SHAP). Our system integrates:
1. Bidirectional statute translation bridging IPC and BNS.
2. Corrective Precedent Retrieval (Corrective RAG) using semantic embeddings and self-evaluating LLM agents.
3. Multi-objective prioritization balancing statutory gravity, age backlog, and immediate liberty threats.
4. Algorithmic bias and fairness auditing across civil and criminal dockets.
5. A LangGraph-based Judicial Reconciliation Agent that deliberates upon cross-module conflicts to assist the bench.

Simulated on a dataset of Supreme Court judgements, our proposed AI triage reduces average resolution time for critical/high-severity cases by **74.45%** compared to traditional FIFO scheduling without causing docket starvation.

---

## 1. Introduction
Court pendency severely hampers access to timely justice. A major reason for this delay is unranked scheduling, where a routine procedural boundary dispute receives the same docket scheduling weight as urgent bail hearings or violent criminal matters.

Furthermore, legal practitioners and judges must map 75+ years of jurisprudence decided under the **Indian Penal Code (IPC 1860)** and **Code of Civil Procedure (CPC 1908)** to the recently enacted **Bharatiya Nyaya Sanhita (BNS 2023)**.

To solve both challenges simultaneously, we designed an explainable, human-in-the-loop multi-agent architecture.

---

## 2. System Architecture & Multi-Agent Design

The system is organized into modular agents orchestrated with **LangGraph** and supported by PostgreSQL with `pgvector`:

```
                 [Case PDF / Ingestion]
                            │
                            ▼
              ┌───────────────────────────┐
              │ Module 1: Parser & Bridge │
              │ (IPC <-> BNS Mapping)     │
              └─────────────┬─────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
 ┌──────────────────────┐      ┌────────────────────────┐
 │ Module 2A: Precedent │      │ Module 3: Priority &   │
 │ Retrieval (LangGraph │      │ Fairness Engine        │
 │ Corrective RAG)      │      │ (XGBoost + SHAP + Bias)│
 └───────────┬──────────┘      └───────────┬────────────┘
             │                             │
             └──────────────┬──────────────┘
                            ▼
              ┌───────────────────────────┐
              │ LangGraph Reconciliation  │
              │ Evaluator-Optimizer Agent │
              └─────────────┬─────────────┘
                            ▼
             [Interactive Judicial Bench]
             (React UI & Streamlit App)
```

### 2.1 Agent Roles:
1. **Document Processing & Statute Mapping Agent:** Extracts procedural facts, parses statutory mentions, and maps IPC sections to equivalent BNS counterparts with baseline severity weightings (1–10).
2. **Corrective Precedent Search Agent (LangGraph):**
   - Employs `nlpaueb/legal-bert-base-uncased` to generate 768-dimensional embeddings stored in `pgvector`.
   - Uses a StateGraph with nodes `grade_first_pass`, `rewrite_query`, `retrieve_rewritten`, and `grade_retry` to ensure retrieved precedents meet the strict relevance threshold ($\ge 0.65$).
3. **Intelligent Prioritizer & Explainability Agent:**
   - Multi-factor formulation scoring:
     $$\text{Priority} = w_1 \cdot \text{Severity} + w_2 \cdot \text{Threat} + w_3 \cdot \text{Societal Impact} + w_4 \cdot \text{Normalized Age}$$
   - Augmented by XGBoost regression and SHAP local feature attributions.
4. **Bias & Fairness Auditor:** Evaluates demographic neutrality, checks for demographic parity differences across case types, and alerts against systemic skew.
5. **Reconciliation Agent (LangGraph):**
   - Intercepts dossiers where the priority score clashes with fundamental liberties (e.g., habeas corpus, custody) or where bias flags are active.
   - Executes LLM deliberation to synthesize a final bench recommendation and triage memo (`APPROVED`, `ADJUSTED`, `FLAGGED_FOR_BENCH`).

---

## 3. Simulation & Empirical Evaluation

To evaluate the real-world utility of the proposed triage scheduler, we conducted a courtroom simulation over **860 Supreme Court cases** with **3 Virtual Judges** (daily throughput = 6 cases/day).

### 3.1 Baselines:
- **FIFO (First-In, First-Out):** Strict chronological listing according to case filing date.
- **LIFO (Last-In, First-Out):** Recency-biased listing.
- **Random / Unprioritized:** Simulated unweighted docket assignment.
- **AI Agent Prioritization (Proposed):** Dynamic ordering using the multi-agent priority scores.

### 3.2 Key Evaluation Results:

| Scheduling Strategy | Total Cases | Critical Cases | Avg Wait Days (Critical) | Median Wait Days (Critical) | 90th Percentile Wait (Critical) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FIFO (Traditional)** | 860 | 89 | 63.73 days | 43.0 days | 131.0 days |
| **LIFO (Recency Skew)** | 860 | 89 | 90.91 days | 86.0 days | 131.0 days |
| **Random Docket** | 860 | 89 | 76.04 days | 80.0 days | 129.4 days |
| **AI Agent (Proposed)** | **860** | **89** | **16.28 days** | **8.0 days** | **39.2 days** |

### 3.3 Core Findings:
1. **Critical Case Turnaround Improvement:** The AI agent reduces the average resolution time of critical/high-severity matters from **63.73 days to 16.28 days**—a **74.45% speedup**.
2. **Backlog Threat Decay:** The cumulative severity index of pending cases decays rapidly under AI prioritization, drastically minimizing public harm and prolonging of urgent fundamental rights cases.
3. **Fairness Preservation:** The demographic parity difference between criminal and civil listings remained balanced without starvation of civil matters.

---

## 4. Artifacts & Generated Figures
- `evaluation/simulation_results/critical_case_resolution_time.png`: Bar comparison across scheduling regimes.
- `evaluation/simulation_results/backlog_decay_curve.png`: Trajectory of docket severity decay over time.
- `evaluation/simulation_results/simulation_metrics.json`: Raw simulation parameters and distribution metrics.

---

## 5. Conclusion
The proposed agentic judicial decision support framework demonstrates that combining domain-adapted transformers (Legal-BERT) with stateful multi-agent deliberation (LangGraph) and explainable prioritization can significantly alleviate courtroom congestion while upholding transparency and fairness.
