# Judicial AI Decision Support System - Capstone Design Document

This document outlines the architecture, module breakdown, handling of legal statutes (IPC vs. BNS), and evaluation strategy for the B.Tech Capstone Project. It is designed to act as a foundational reference for your implementation and research paper.

---

## 1. Multi-Agent Architecture

To process cases efficiently, the system is divided into collaborative AI agents. This modular approach is excellent for a capstone paper because it demonstrates distributed AI design.

* **Agent 1: Document Processing & Summarization Agent**
  * **Role:** Parses incoming case PDFs.
  * **Action:** Cleans the text, removes boilerplate, and extracts two main components: **Synopsis of the Case** (facts) and **Judgment Passed**.
* **Agent 2: Legal Statute Mapping Agent (The Bridge)**
  * **Role:** Resolves the IPC vs. BNS conflict and handles CPC.
  * **Action:** Scans documents for sections. If it finds IPC (in historical documents), it maps it to the equivalent BNS section. It also assigns a **Severity Score** based on the referenced crimes.
* **Agent 3: Precedent Retrieval Agent (RAG)**
  * **Role:** Finds similar historical cases.
  * **Action:** Uses Legal-BERT to embed the current case facts and sections, returning the top $K$ most similar historical cases.
* **Agent 4: Intelligent Prioritization Scheduler**
  * **Role:** Ranks the case in the court's docket.
  * **Action:** Computes a priority score based on: 
    * *Immediate Threat / Societal Impact* (derived from the severity of the sections).
    * *Case Age / Recency* (as advised by your lawyers, prioritizing newer and highly severe cases).
* **Agent 5: Explainability Agent (XAI)**
  * **Role:** Provides transparency for the judge.
  * **Action:** Generates a human-readable justification for *why* a case was marked "Critical" and *why* specific precedents were retrieved.

---

## 2. Solving the IPC to BNS Mapping Problem

Since your 75-year-old dataset uses the **Indian Penal Code (IPC)** and modern criminal law uses the **Bharatiya Nyaya Sanhita (BNS)**, you need a "Bidirectional Translation" strategy.

### The Strategy: Data Enrichment via Mapping Dictionary
1. **Create a Mapping Dictionary:** Create a JSON or CSV file mapping IPC sections to BNS sections (e.g., `IPC 302` -> `BNS 103` [Murder]). Include a "Severity Score" (1-10) for each.
2. **Enrich Historical Data:** When your `extract_features.py` script processes the Indian Kanoon dataset, have it find IPC mentions. For every IPC mention, artificially inject the equivalent BNS tag into the metadata and text chunks *before* passing them to Legal-BERT.
3. **Future-Proof Querying:** When a judge inputs a new criminal case today (which will use BNS), the Legal-BERT embeddings will successfully match the modern BNS terms against the enriched historical dataset.
4. **Civil Cases:** For civil cases, the system cleanly branches off and relies on the Code of Civil Procedure (CPC) and other relevant acts. 

---

## 3. Prioritization & Evaluation (The Simulation)

To prove your system works for the research paper, you cannot wait for real judges to use it for years. You must **simulate** a court.

### Simulation Design
* **The Setup:** Imagine a queue of 1,000 cases waiting to be heard. You have 3 "Virtual Judges." Each judge can finish 2 cases a day.
* **The Baselines (Traditional Systems):**
  * **FIFO (First-In-First-Out):** The judges take cases in the exact order they were filed.
  * **LIFO (Last-In-First-Out):** Judges take the newest cases first (sometimes happens in reality when new pressing cases overshadow old ones).
* **The AI Approach (Your Project):**
  * Cases are ordered dynamically by the **Prioritization Agent**. Critical cases (murder, immediate threats to society) are pushed to the top. Low-impact civil disputes are scheduled later.
* **Metrics for Your Paper:**
  * *High-Priority Resolution Time:* Measure how many days it takes on average to resolve a "Critical" case under FIFO vs. AI. (Your AI should drastically reduce this).
  * *Throughput:* Total severity-weighted cases resolved per month. 
  * *Backlog Decay:* Show a graph of the backlog's severity decreasing over time.

---

## 4. Outline for your Research Paper

Use this structure for your B.Tech CSE final paper:

1. **Abstract:** Summary of reducing court pendency using a Multi-Agent LLM system.
2. **Introduction:** The Indian judicial backlog, the shift from IPC to BNS, and the need for AI prioritization.
3. **Literature Review:** Existing NLP in law (Legal-BERT), classical case scheduling vs AI scheduling.
4. **Methodology / Architecture:**
   * Discuss the Indian Kanoon dataset.
   * Diagram the 5-Agent Architecture.
   * Detail the **Semantic Search (RAG)** implementation (`nlpaueb/legal-bert`).
   * Detail the **IPC to BNS Translation Vector Mapping**.
5. **Experimental Setup (Simulation):**
   * Explain the Priority Scoring Formula (Weights for Recency, Severity, Threat).
   * Describe the Courtroom Simulation parameters.
6. **Results & Evaluation:**
   * Graphs comparing FIFO / LIFO vs Priority Scheduling.
   * Precision/Recall for semantic search retrieval.
7. **Conclusion & Future Scope:** How this system assists (doesn't replace) judges.
