"""
Semantic Retrieval Benchmark
Empirically evaluates Legal-BERT, Lexical BM25, and Hybrid Fusion (Legal-BERT + Lexical Section/Title Boost).
Evaluates cross-regime retrieval (modern BNS queries retrieving historical IPC precedents).
Produces ranking-overlap results for exploratory comparison. Without relevance
judgments, this benchmark does not measure retrieval accuracy or recall.
"""

import os
import sys
import json
import psycopg2
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

OUTPUT_CSV = "evaluation/retrieval_benchmark_results.csv"
OUTPUT_TEX = "evaluation/retrieval_benchmark_table.tex"
RRF_K = 60
RRF_POOL_SIZE = 100

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def run_retrieval_benchmark():
    print("[RETRIEVAL] Loading case texts from database and chunk cache...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SELECT case_id, ipc_sections, bns_sections, case_summary FROM cases;")
    db_cases = cur.fetchall()
    cur.execute("SELECT DISTINCT case_id FROM case_chunks;")
    vector_case_ids = {r[0] for r in cur.fetchall()}
    conn.close()

    corpus_ids = []
    bm25_chunk_ids = []
    bm25_chunk_texts = []
    case_metadata = {}

    for case_id, ipc, bns, summary in db_cases:
        if case_id not in vector_case_ids:
            continue
        chunk_file = os.path.join("data", "chunks", f"{case_id}_chunks.json")
        chunk_texts = []
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                chunk_texts = [c["text"] for c in cdata.get("chunks", []) if c.get("text")]
            except Exception:
                pass
        if not chunk_texts and summary:
            chunk_texts = [summary]

        if chunk_texts:
            corpus_ids.append(case_id)
            case_metadata[case_id] = {
                "ipc": ipc or "",
                "bns": bns or ""
            }
            bm25_chunk_ids.extend([case_id] * len(chunk_texts))
            bm25_chunk_texts.extend(chunk_texts)

    print(f"[RETRIEVAL] Loaded {len(corpus_ids)} cases and {len(bm25_chunk_texts)} chunks into evaluation corpus.")

    # BM25 scores the same chunks used by vector search, then pools each case
    # by its best matching chunk so both baselines return case-level rankings.
    print("[RETRIEVAL] Building chunk-level BM25 index...")
    tokenized_corpus = [t.lower().split() for t in bm25_chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # 2. Load Legal-BERT
    print("[RETRIEVAL] Loading Legal-BERT model (nlpaueb/legal-bert-base-uncased)...")
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    query_embedding_cache = {}

    def get_query_embedding(query_text):
        if query_text in query_embedding_cache:
            return query_embedding_cache[query_text]
        encoded = tokenizer([query_text.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pooling(output, encoded['attention_mask'])
        result = F.normalize(emb, p=2, dim=1).numpy()[0]
        query_embedding_cache[query_text] = result
        return result

    def search_bm25(query_text, top_k=None):
        scores = bm25.get_scores(query_text.lower().split())
        case_scores = {}
        for case_id, score in zip(bm25_chunk_ids, scores):
            case_scores[case_id] = max(case_scores.get(case_id, float("-inf")), float(score))
        ranked = sorted(case_scores, key=case_scores.get, reverse=True)
        return ranked if top_k is None else ranked[:top_k]

    def search_legalbert_vector(query_text, top_k=None):
        emb = get_query_embedding(query_text)
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        limit_clause = "LIMIT %s" if top_k is not None else ""
        params = (emb.tolist(), top_k) if top_k is not None else (emb.tolist(),)
        cur.execute("""
            SELECT * FROM (
                SELECT DISTINCT ON (c.case_id)
                    c.case_id,
                    cc.embedding <=> %s::vector AS distance
                FROM case_chunks cc
                JOIN cases c ON cc.case_id = c.case_id
                ORDER BY c.case_id, distance ASC
            ) AS matches
            ORDER BY distance ASC
        """ + limit_clause, params)
        records = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        return records

    def search_bm25_legalbert_rrf(query_text, top_k=10):
        """Fuse BM25 and Legal-BERT case rankings with equal-weight RRF."""
        bm25_ranked = search_bm25(query_text, top_k=RRF_POOL_SIZE)
        bert_ranked = search_legalbert_vector(query_text, top_k=RRF_POOL_SIZE)
        fused_scores = {}
        for ranked in (bm25_ranked, bert_ranked):
            for rank, case_id in enumerate(ranked, start=1):
                fused_scores[case_id] = fused_scores.get(case_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]

    def search_hybrid(query_text, top_k=10):
        emb = get_query_embedding(query_text)
        query_words = [w.lower() for w in query_text.replace("_", " ").split() if len(w) > 2]

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                c.case_id, 
                MIN(cc.embedding <=> %s::vector) AS min_dist,
                c.bns_sections, c.ipc_sections
            FROM case_chunks cc
            JOIN cases c ON cc.case_id = c.case_id
            GROUP BY c.case_id, c.bns_sections, c.ipc_sections;
        """, (emb.tolist(),))
        records = cur.fetchall()
        conn.close()

        scored = []
        for r in records:
            case_id = r[0]
            dist = float(r[1])
            vec_sim = 1.0 - dist
            case_id_lower = case_id.lower().replace("_", " ")
            ipc_lower = (r[3] or "").lower()
            bns_lower = (r[2] or "").lower()

            title_matches = sum(1 for w in query_words if w in case_id_lower)
            lexical_boost = 0.0
            if len(query_words) > 0:
                if title_matches == len(query_words):
                    lexical_boost = 0.50
                elif title_matches > 0:
                    lexical_boost = 0.25 * (title_matches / len(query_words))

            if any(w in ipc_lower or w in bns_lower for w in query_words):
                lexical_boost += 0.15

            scored.append((vec_sim + lexical_boost, case_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scored[:top_k]]

    # Test Benchmark Queries (Canonical Indian Jurisprudence + Cross-Regime BNS Queries)
    test_queries = [
        # Criminal Core
        {"query": "murder conviction Section 302 Indian Penal Code appeal", "type": "Statutory IPC"},
        {"query": "rape prosecution Section 376 consent bail petition", "type": "Statutory IPC"},
        {"query": "dowry death cruelty by husband Section 304B", "type": "Statutory IPC"},
        {"query": "attempt to murder Section 307 dangerous weapon", "type": "Statutory IPC"},
        {"query": "cheating and criminal breach of trust Section 420 Section 406", "type": "Statutory IPC"},
        
        # Fundamental Rights / Liberty
        {"query": "habeas corpus unlawful detention personal liberty Article 21", "type": "Constitutional"},
        {"query": "Public Interest Litigation environmental pollution ecological damage", "type": "PIL / Constitutional"},
        {"query": "anticipatory bail custodial interrogation arrest protection", "type": "Procedural Criminal"},

        # Civil / Commercial
        {"query": "temporary injunction prima facie case balance of convenience CPC Order 39", "type": "Civil CPC"},
        {"query": "arbitration award challenge Section 34 arbitral tribunal", "type": "Commercial Civil"},

        # Cross-regime candidates with unambiguous IPC counterparts are checked
        # as section-hit proxies only; this is not a relevance judgment.
        {"query": "murder charges under BNS Section 103 criminal appeal", "type": "Cross-Regime BNS"},
        {"query": "sexual assault prosecution under BNS Section 64", "type": "Cross-Regime BNS", "expected_ipc": None},
        {"query": "dowry death investigation under Section 80 BNS", "type": "Cross-Regime BNS", "expected_ipc": {"304b"}},
        {"query": "snatching under BNS Section 304", "type": "Cross-Regime BNS", "expected_ipc": None}
    ]
    test_queries[10]["expected_ipc"] = {"302"}

    print("\n--- Running Retrieval Comparisons across Test Queries ---")
    results = []

    for q_item in test_queries:
        q = q_item["query"]
        q_type = q_item["type"]

        bm25_top = search_bm25(q, top_k=10)
        bert_top = search_legalbert_vector(q, top_k=10)
        hybrid_top = search_hybrid(q, top_k=10)
        rrf_top = search_bm25_legalbert_rrf(q, top_k=10)

        # Overlap@10 between BM25 and Legal-BERT
        overlap_bert_bm25 = len(set(bm25_top) & set(bert_top))
        overlap_hybrid_bert = len(set(hybrid_top) & set(bert_top))

        expected_ipc = q_item.get("expected_ipc", "not_applicable")

        def mapped_ipc_hit_at_3(ranked_cases):
            if q_type != "Cross-Regime BNS":
                return "N/A"
            if expected_ipc is None:
                return "AMBIGUOUS/NO DIRECT MAP"
            return "YES" if any(
                expected_ipc.intersection(
                    {section.strip().lower() for section in case_metadata.get(cid, {}).get("ipc", "").split(",")}
                )
                for cid in ranked_cases[:3]
            ) else "NO"

        results.append({
            "Query": q,
            "Domain": q_type,
            "BM25 Top-1": bm25_top[0].replace("_", " ")[:35],
            "LegalBERT Top-1": bert_top[0].replace("_", " ")[:35],
            "Hybrid Top-1": hybrid_top[0].replace("_", " ")[:35],
            "BM25+LegalBERT RRF Top-1": rrf_top[0].replace("_", " ")[:35],
            "BM25-BERT Overlap@10": overlap_bert_bm25,
            "Hybrid-BERT Overlap@10": overlap_hybrid_bert,
            "RRF-BM25 Overlap@10": len(set(rrf_top) & set(bm25_top)),
            "RRF-BERT Overlap@10": len(set(rrf_top) & set(bert_top)),
            "RRF-CurrentHybrid Overlap@10": len(set(rrf_top) & set(hybrid_top)),
            "BM25 Mapped IPC Section Hit@3 (proxy)": mapped_ipc_hit_at_3(bm25_top),
            "LegalBERT Mapped IPC Section Hit@3 (proxy)": mapped_ipc_hit_at_3(bert_top),
            "CurrentHybrid Mapped IPC Section Hit@3 (proxy)": mapped_ipc_hit_at_3(hybrid_top),
            "RRF Mapped IPC Section Hit@3 (proxy)": mapped_ipc_hit_at_3(rrf_top)
        })

    res_df = pd.DataFrame(results)

    print("\n==========================================================================================")
    print("                    RETRIEVAL BENCHMARK & CROSS-REGIME COMPARISON                          ")
    print("==========================================================================================")
    print(res_df[["Query", "Domain", "BM25-BERT Overlap@10", "RRF-BM25 Overlap@10", "RRF-BERT Overlap@10", "RRF-CurrentHybrid Overlap@10", "BM25 Mapped IPC Section Hit@3 (proxy)", "LegalBERT Mapped IPC Section Hit@3 (proxy)", "CurrentHybrid Mapped IPC Section Hit@3 (proxy)", "RRF Mapped IPC Section Hit@3 (proxy)"]].to_string(index=False))

    # Save CSV & LaTeX
    res_df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(res_df.to_latex(index=False, caption="Exploratory Retrieval Ranking Overlap (No Relevance Judgments)", label="tab:retrieval_benchmark"))

    print(f"\n[RETRIEVAL] Saved CSV -> {OUTPUT_CSV}")
    print(f"[RETRIEVAL] Saved LaTeX -> {OUTPUT_TEX}")
    print(f"[RETRIEVAL] Mean BM25 vs Legal-BERT Overlap@10: {res_df['BM25-BERT Overlap@10'].mean():.2f} / 10")
    print(f"[RETRIEVAL] Mean RRF vs BM25 Overlap@10: {res_df['RRF-BM25 Overlap@10'].mean():.2f} / 10")
    print(f"[RETRIEVAL] Mean RRF vs Legal-BERT Overlap@10: {res_df['RRF-BERT Overlap@10'].mean():.2f} / 10")
    print(f"[RETRIEVAL] Mean RRF vs current hybrid Overlap@10: {res_df['RRF-CurrentHybrid Overlap@10'].mean():.2f} / 10")
    print("[SUCCESS] Retrieval Benchmark Completed Cleanly!\n")

if __name__ == "__main__":
    run_retrieval_benchmark()
