from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os
import sys
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from rank_bm25 import BM25Okapi

# Ensure parent directory is in sys.path for config and agent imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL, ALLOW_ORIGINS

app = FastAPI(title="Judicial AI Search Engine")

# Configure CORS using centralized configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Legal-BERT Model (This takes a moment)...")
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("Model loaded successfully!")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class SearchRequest(BaseModel):
    query: str
    yearFilter: str = ""
    topK: int = 10

# Lazy-loaded Agents to save boot RAM
UPLOAD_AGENT = None
PRIORITIZATION_AGENT = None
EXPLAINABILITY_AGENT = None
BIAS_AGENT = None
RECONCILIATION_AGENT = None
PRECEDENT_SEARCH_AGENT = None


def retrieve_hybrid_candidates(query_text: str, year_filter: str, top_k: int):
    """Run the existing Legal-BERT + pgvector + lexical precedent retrieval."""
    global PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT

    encoded_input = tokenizer([query_text.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = F.normalize(query_emb, p=2, dim=1).numpy()[0]
    query_words = [word.lower() for word in query_text.replace("_", " ").split() if len(word) > 2]

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    try:
        select_sql = """
            SELECT c.case_id, MIN(cc.embedding <=> %s::vector) AS min_dist,
                   c.case_date, c.max_severity_score, c.case_summary,
                   c.bns_sections, c.ipc_sections, c.priority_score, c.case_type,
                   c.num_ipc_sections, c.num_cpc_sections, c.num_precedents,
                   c.case_age_days, c.immediate_threat_flag, c.societal_impact_score,
                   c.total_words, c.bias_score, c.bias_details, c.bias_flags,
                   c.legal_regime
            FROM case_chunks cc
            JOIN cases c ON cc.case_id = c.case_id
        """
        if year_filter:
            select_sql += " WHERE c.case_date LIKE %s OR c.case_id LIKE %s"
        select_sql += """
            GROUP BY c.case_id, c.case_date, c.max_severity_score, c.case_summary,
                     c.bns_sections, c.ipc_sections, c.priority_score, c.case_type,
                     c.num_ipc_sections, c.num_cpc_sections, c.num_precedents,
                     c.case_age_days, c.immediate_threat_flag, c.societal_impact_score,
                     c.total_words, c.bias_score, c.bias_details, c.bias_flags, c.legal_regime
        """
        if year_filter:
            year_param = f"%{year_filter}%"
            cur.execute(select_sql, (query_emb.tolist(), year_param, year_param))
        else:
            cur.execute(select_sql, (query_emb.tolist(),))
        records = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    scored_records = []
    for record in records:
        vec_sim = 1.0 - float(record[1])
        case_id_lower = record[0].lower().replace("_", " ")
        ipc_lower = (record[6] or "").lower()
        bns_lower = (record[5] or "").lower()
        title_matches = sum(1 for word in query_words if word in case_id_lower)
        lexical_boost = 0.0
        if query_words:
            if title_matches == len(query_words):
                lexical_boost = 0.50
            elif title_matches:
                lexical_boost = 0.25 * (title_matches / len(query_words))
        if any(word in ipc_lower or word in bns_lower for word in query_words):
            lexical_boost += 0.15
        scored_records.append((vec_sim + lexical_boost, record))

    scored_records.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, record in scored_records[:max(1, min(top_k, 50))]:
        explanation = PRIORITIZATION_AGENT.generate_explanation(
            case_type=record[8] or "Unknown",
            num_ipc_sections=record[9] if record[9] is not None else 0,
            num_cpc_sections=record[10] if record[10] is not None else 0,
            case_age_days=record[12],
            num_precedents=record[11] if record[11] is not None else 0,
            severity=float(record[3]) if record[3] is not None else 3.0,
            immediate_threat=record[13] if record[13] is not None else 0,
            societal_impact=record[14] if record[14] is not None else 1,
            case_date=record[2],
            scoring_method="baseline Rule-Based framework"
        )
        features = {
            "case_type": record[8] or "civil",
            "num_ipc_sections": record[9] if record[9] is not None else 0,
            "num_cpc_sections": record[10] if record[10] is not None else 0,
            "num_precedents": record[11] if record[11] is not None else 0,
            "total_words": record[15] if record[15] is not None else 1000,
            "case_age_days": record[12] if record[12] is not None else 365.0,
            "max_severity_score": float(record[3]) if record[3] is not None else 3.0,
            "immediate_threat_flag": record[13] if record[13] is not None else 0,
            "societal_impact_score": record[14] if record[14] is not None else 1,
        }
        contributions = EXPLAINABILITY_AGENT._fallback_attributions(features)
        current_priority_score = PRIORITIZATION_AGENT.compute_priority_score(
            severity=features["max_severity_score"],
            societal_impact=features["societal_impact_score"],
            immediate_threat=features["immediate_threat_flag"],
            case_age_days=features["case_age_days"],
            case_type=features["case_type"],
        )
        results.append({
            "case_id": record[0],
            "score": round(min(score, 1.0), 4),
            "date": record[2] if record[2] else "Unknown",
            "severity": record[3],
            "summary": record[4] if record[4] else "AI Synopsis Pending Processing.",
            "bns_sections": record[5] or "",
            "ipc_sections": record[6] or "",
            "priority_score": current_priority_score,
            "case_type": record[8] or "Unknown",
            "legal_regime": record[19] or "Unknown",
            "priority_explanation": explanation,
            "contributions": contributions,
            "bias_score": record[16],
            "bias_details": record[17] if record[17] is not None else "Automated audit unavailable.",
            "bias_flags": record[18].split(",") if (record[18] and record[18].strip()) else [],
        })
    return results

# BM25 over the same chunks used to generate the pgvector embeddings.
BM25_RRF_K = 60
BM25_RRF_POOL_SIZE = 100
BM25_INDEX = None
BM25_CHUNK_CASE_IDS = []


def build_bm25_index():
    """Build a case-linked BM25 index from the persisted chunk files."""
    global BM25_INDEX, BM25_CHUNK_CASE_IDS

    chunks_dir = Path(__file__).resolve().parents[1] / "data" / "chunks"
    chunk_texts = []
    chunk_case_ids = []

    for chunk_file in sorted(chunks_dir.glob("*_chunks.json")):
        try:
            with chunk_file.open("r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            case_id = chunk_data.get("case_id") or chunk_file.name[:-len("_chunks.json")]
            for chunk in chunk_data.get("chunks", []):
                text = chunk.get("text", "").strip()
                if text:
                    chunk_texts.append(text.lower().split())
                    chunk_case_ids.append(case_id)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ Skipping unreadable BM25 chunk file {chunk_file.name}: {exc}")

    if not chunk_texts:
        BM25_INDEX = None
        BM25_CHUNK_CASE_IDS = []
        print(f"⚠️ No chunk text found for BM25 index in {chunks_dir}")
        return

    BM25_INDEX = BM25Okapi(chunk_texts)
    BM25_CHUNK_CASE_IDS = chunk_case_ids
    print(f"BM25 index ready: {len(chunk_texts)} chunks across {len(set(chunk_case_ids))} cases.")


@app.on_event("startup")
async def initialize_bm25_index():
    build_bm25_index()


def search_bm25_legalbert_rrf(query_text, vector_records, top_k):
    """Fuse BM25 chunk ranks with per-case Legal-BERT vector ranks using RRF."""
    if BM25_INDEX is None:
        raise HTTPException(status_code=503, detail="BM25 retrieval index is unavailable")

    record_by_case = {record[0]: record for record in vector_records}
    available_case_ids = set(record_by_case)

    vector_ranked = [
        record[0]
        for record in sorted(vector_records, key=lambda record: float(record[1]))[:BM25_RRF_POOL_SIZE]
    ]

    bm25_chunk_scores = BM25_INDEX.get_scores(query_text.lower().split())
    bm25_case_scores = {}
    for case_id, score in zip(BM25_CHUNK_CASE_IDS, bm25_chunk_scores):
        if case_id in available_case_ids:
            bm25_case_scores[case_id] = max(bm25_case_scores.get(case_id, float("-inf")), float(score))

    bm25_ranked = sorted(
        (case_id for case_id, score in bm25_case_scores.items() if score > 0),
        key=bm25_case_scores.get,
        reverse=True
    )[:BM25_RRF_POOL_SIZE]

    fused_scores = {}
    for ranked in (bm25_ranked, vector_ranked):
        for rank, case_id in enumerate(ranked, start=1):
            fused_scores[case_id] = fused_scores.get(case_id, 0.0) + 1.0 / (BM25_RRF_K + rank)

    fused_case_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    return [(fused_scores[case_id], record_by_case[case_id]) for case_id in fused_case_ids]

@app.post("/summarize_upload")
async def process_pdf_upload(file: UploadFile = File(...)):
    global UPLOAD_AGENT, PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT, BIAS_AGENT, RECONCILIATION_AGENT
    conn = None
    cur = None
    try:
        # Standardize file string as Case UID without case-sensitive extensions
        filename_id = file.filename
        if filename_id.lower().endswith(".pdf"):
            filename_id = filename_id[:-4]
        filename_id = filename_id.replace(" ", "_")
        
        # Read PDF Stream to extract text
        contents = await file.read()
        import fitz
        doc = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Generate sha256 Content Hash
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        
        # 1. SMART CACHE LOOKUP BY CONTENT HASH
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT case_summary, max_severity_score, legal_regime, ipc_sections, bns_sections, priority_score, case_type, bias_score, bias_details, bias_flags, case_id,
                   num_ipc_sections, num_cpc_sections, num_precedents, case_age_days, immediate_threat_flag, societal_impact_score, total_words, case_date
            FROM cases WHERE content_hash = %s
        """, (content_hash,))
        result = cur.fetchone()
        
        # Initialize Prioritization, Explainability, & Bias Agents
        if not PRIORITIZATION_AGENT:
            print("🚀 Loading Prioritization Agent...")
            from agents.prioritization.prioritizer import PrioritizationAgent
            PRIORITIZATION_AGENT = PrioritizationAgent()

        if not EXPLAINABILITY_AGENT:
            print("🚀 Loading Explainability Agent...")
            from agents.explainability.explainability_agent import ExplainabilityAgent
            EXPLAINABILITY_AGENT = ExplainabilityAgent()

        if not BIAS_AGENT:
            print("🚀 Loading Bias Agent...")
            from agents.fairness.bias_agent import BiasAgent
            BIAS_AGENT = BiasAgent()

        if not RECONCILIATION_AGENT:
            print("🏛️ Loading Judicial Reconciliation Agent...")
            from agents.reconciliation.reconciliation_agent import JudicialReconciliationAgent
            RECONCILIATION_AGENT = JudicialReconciliationAgent()
            
        if result and result[0]:
            print(f"🔥 Fast Cache Hit! Returned {result[10]} instantly without hitting Gemini.")
            
            # Generate explanation for cache hit
            explanation = PRIORITIZATION_AGENT.generate_explanation(
                case_type=result[6] if result[6] else "Unknown",
                num_ipc_sections=result[11] if result[11] is not None else 0,
                num_cpc_sections=result[12] if result[12] is not None else 0,
                case_age_days=result[14] if result[14] is not None else None,
                num_precedents=result[13] if result[13] is not None else 0,
                severity=result[1] if result[1] is not None else 3.0,
                immediate_threat=result[15] if result[15] is not None else 0,
                societal_impact=result[16] if result[16] is not None else 1,
                case_date=result[18] if result[18] is not None else None
            )

            # Generate dynamic contributions for cache hit with correct index splits
            features_for_explain = {
                "case_type": result[6] if result[6] else "civil",
                "num_ipc_sections": result[11] if result[11] is not None else 0,
                "num_cpc_sections": result[12] if result[12] is not None else 0,
                "num_precedents": result[13] if result[13] is not None else 0,
                "total_words": result[17] if result[17] is not None else 1000,
                "case_age_days": result[14] if result[14] is not None else 365.0,
                "max_severity_score": float(result[1]) if result[1] is not None else 3.0,
                "immediate_threat_flag": result[15] if result[15] is not None else 0,
                "societal_impact_score": result[16] if result[16] is not None else 1
            }
            contributions = EXPLAINABILITY_AGENT.compute_marginal_attributions(features_for_explain)
            
            # Retrieve similar cases using database case_chunks embedding
            similar_cases = []
            try:
                cur.execute("SELECT embedding FROM case_chunks WHERE case_id = %s LIMIT 1", (result[10],))
                emb_res = cur.fetchone()
                if emb_res:
                    cur.execute("""
                        SELECT * FROM (
                            SELECT DISTINCT ON (c.case_id) 
                                c.case_id, 
                                cc.embedding <=> %s::vector AS distance, 
                                c.case_date, 
                                c.max_severity_score,
                                c.case_summary,
                                c.priority_score,
                                c.case_type
                            FROM case_chunks cc
                            JOIN cases c ON cc.case_id = c.case_id
                            WHERE c.case_id != %s
                            ORDER BY c.case_id, distance ASC
                        ) AS distinct_matches
                        ORDER BY distance ASC
                        LIMIT 3;
                    """, (emb_res[0], result[10]))
                    sim_records = cur.fetchall()
                    for r in sim_records:
                        sim_score = 1 - float(r[1])
                        similar_cases.append({
                            "case_id": r[0],
                            "score": round(sim_score, 4),
                            "date": r[2] if r[2] else "Unknown",
                            "severity": r[3],
                            "summary": r[4] if r[4] else "AI Synopsis Pending Processing.",
                            "priority_score": r[5] if r[5] is not None else 0.0,
                            "case_type": r[6] if r[6] else "Unknown"
                        })
            except Exception as sim_err:
                print(f"⚠️ Warning: Cache hit semantic similarity lookup failed: {sim_err}")
            
            # Execute LangGraph Judicial Reconciliation Agent
            reconciliation = {}
            try:
                reconciliation = RECONCILIATION_AGENT.reconcile({
                    "case_id": result[10] if (len(result) > 10 and result[10]) else filename_id,
                    "case_type": result[6] if result[6] else "civil",
                    "legal_regime": result[2] if result[2] else "ipc",
                    "summary": result[0],
                    "priority_score": float(result[5]) if result[5] is not None else 5.0,
                    "severity": float(result[1]) if result[1] is not None else 3.0,
                    "contributions": contributions,
                    "bias_score": result[7],
                    "bias_details": result[8] if result[8] is not None else "Automated audit unavailable.",
                    "bias_flags": result[9].split(",") if (result[9] and result[9].strip()) else [],
                    "similar_cases": similar_cases
                })
            except Exception as rec_err:
                print(f"[WARNING] Cache hit reconciliation failed: {rec_err}")

            return {
                "case_id": result[10] if (len(result) > 10 and result[10]) else filename_id,
                "summary": result[0],
                "severity": result[1],
                "regime": result[2],
                "ipc_sections": result[3],
                "bns_sections": result[4],
                "priority_score": result[5],
                "case_type": result[6],
                "priority_explanation": explanation,
                "similar_cases": similar_cases,
                "contributions": contributions,
                "bias_score": result[7],
                "bias_details": result[8] if result[8] is not None else "Automated audit unavailable.",
                "bias_flags": result[9].split(",") if (result[9] and result[9].strip()) else [],
                "reconciliation": reconciliation
            }
            
        # Cache Miss -> Close DB connections immediately so they aren't held open during Gemini execution
        if cur:
            cur.close()
        if conn:
            conn.close()
        conn = None
        cur = None
            
        # Instance Summarization Agent Once
        if not UPLOAD_AGENT:
            print("🚀 Loading Gemini API Node...")
            from agents.summarization.summarizer import SummarizationAgent
            UPLOAD_AGENT = SummarizationAgent()
            
        # Generates raw string via Gemini
        raw_summary_string = UPLOAD_AGENT.summarize_upload_stream(text)
        
        # EXTRACT FEATURES NATIVELY
        from scripts.extract_features import (
            extract_case_type, detect_legal_regime, extract_ipc_sections, extract_bns_sections,
            load_mappings, calculate_severity, build_bns_severity, translate_ipc_to_bns, IPC_SEVERITY,
            extract_date, parse_date, count_precedents, calculate_pending_days
        )
        
        ipc_to_bns, cpc_valid_sections = load_mappings("data/mappings.json")
        bns_severity_map = build_bns_severity(ipc_to_bns)
        
        case_type = extract_case_type(text)
        regime = detect_legal_regime(None, text)
        ipc_sections_list = []
        bns_sections_list = []
        cpc_sections_list = []
        severity = 3
        
        if case_type == "criminal":
            if regime == "IPC":
                ipc_sections_list = extract_ipc_sections(text)
                bns_equivalent = translate_ipc_to_bns(ipc_sections_list, ipc_to_bns)
                bns_sections_list = [b for b in bns_equivalent if not b.endswith("?")]
                severity = calculate_severity(ipc_sections_list, IPC_SEVERITY)
            else:
                bns_sections_list = extract_bns_sections(text)
                severity = calculate_severity(bns_sections_list, bns_severity_map)
        else:
            from scripts.extract_features import extract_cpc_sections
            cpc_sections_list = extract_cpc_sections(text, cpc_valid_sections)
            severity = min(2 + len(cpc_sections_list), 6)
                
        bns_str = ",".join(bns_sections_list)
        ipc_str = ",".join(ipc_sections_list)
        
        # Calculate dynamic filing date and case age
        date_str = extract_date(text)
        parsed_date = parse_date(date_str)
        case_age_val = calculate_pending_days(parsed_date, text)
        
        # Calculate dynamic priority score using Prioritization Agent
        num_precedents = count_precedents(text)
        immediate_threat = PRIORITIZATION_AGENT.check_immediate_threat(text)
        societal_impact = PRIORITIZATION_AGENT.calculate_societal_impact(text)
        
        if PRIORITIZATION_AGENT.model is not None:
            total_words = len(text.split())
            priority_score = PRIORITIZATION_AGENT.predict_ml_priority(
                case_type=case_type,
                num_ipc_sections=len(ipc_sections_list),
                num_cpc_sections=len(cpc_sections_list),
                num_precedents=num_precedents,
                total_words=total_words,
                case_age_days=case_age_val,
                max_severity_score=float(severity),
                immediate_threat_flag=immediate_threat,
                societal_impact_score=societal_impact
            )
        else:
            priority_score = PRIORITIZATION_AGENT.compute_priority_score(
                severity=int(severity),
                societal_impact=societal_impact,
                immediate_threat=immediate_threat,
                case_age_days=case_age_val,
                case_type=case_type
            )
            
        explanation = PRIORITIZATION_AGENT.generate_explanation(
            case_type=case_type,
            num_ipc_sections=len(ipc_sections_list),
            num_cpc_sections=len(cpc_sections_list),
            case_age_days=case_age_val,
            num_precedents=num_precedents,
            severity=float(severity),
            immediate_threat=immediate_threat,
            societal_impact=societal_impact,
            case_date=date_str
        )

        # Generate Explainability attributions
        features_for_explain = {
            "case_type": case_type,
            "num_ipc_sections": len(ipc_sections_list),
            "num_cpc_sections": len(cpc_sections_list),
            "num_precedents": num_precedents,
            "total_words": len(text.split()),
            "case_age_days": case_age_val if case_age_val is not None else 365.0,
            "max_severity_score": float(severity),
            "immediate_threat_flag": immediate_threat,
            "societal_impact_score": societal_impact
        }
        contributions = EXPLAINABILITY_AGENT.compute_marginal_attributions(features_for_explain)

        # Run Bias & Fairness Audit
        bias_audit = BIAS_AGENT.audit_case_fairness(text)
        bias_score = bias_audit["bias_score"]
        bias_details = bias_audit["bias_details"]
        bias_flags_list = bias_audit.get("flags", [])
        bias_flags_str = ",".join(bias_flags_list)
        
        # Now re-open DB connection to persist upload results
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # 2. CACHE SAVER
        cur.execute("SELECT 1 FROM cases WHERE case_id = %s", (filename_id,))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute("""
                INSERT INTO cases (case_id, max_severity_score, legal_regime, ipc_sections, bns_sections, priority_score, case_type, bias_score, bias_details, bias_flags, content_hash) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (filename_id, severity, regime, ipc_str, bns_str, priority_score, case_type, bias_score, bias_details, bias_flags_str, content_hash))
        
        cur.execute("""
            UPDATE cases 
            SET case_summary = %s, max_severity_score = %s, legal_regime = %s, ipc_sections = %s, bns_sections = %s, priority_score = %s, case_type = %s,
                immediate_threat_flag = %s, societal_impact_score = %s, bias_score = %s, bias_details = %s, bias_flags = %s, content_hash = %s
            WHERE case_id = %s
        """, (raw_summary_string, severity, regime, ipc_str, bns_str, priority_score, case_type, immediate_threat, societal_impact, bias_score, bias_details, bias_flags_str, content_hash, filename_id))
        
        # Retrieve similar cases using text embedding of uploaded PDF
        similar_cases = []
        try:
            text_preview = text[:2000]
            encoded_upload = tokenizer([text_preview.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                upload_output = model(**encoded_upload)
            upload_emb = mean_pooling(upload_output, encoded_upload['attention_mask'])
            upload_emb = F.normalize(upload_emb, p=2, dim=1).numpy()[0]
            
            cur.execute("""
                SELECT * FROM (
                    SELECT DISTINCT ON (c.case_id) 
                        c.case_id, 
                        cc.embedding <=> %s::vector AS distance, 
                        c.case_date, 
                        c.max_severity_score,
                        c.case_summary,
                        c.priority_score,
                        c.case_type
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    WHERE c.case_id != %s
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT 3;
            """, (upload_emb.tolist(), filename_id))
            sim_records = cur.fetchall()
            for r in sim_records:
                sim_score = 1 - float(r[1])
                similar_cases.append({
                    "case_id": r[0],
                    "score": round(sim_score, 4),
                    "date": r[2] if r[2] else "Unknown",
                    "severity": r[3],
                    "summary": r[4] if r[4] else "AI Synopsis Pending Processing.",
                    "priority_score": r[5] if r[5] is not None else 0.0,
                    "case_type": r[6] if r[6] else "Unknown"
                })
        except Exception as sim_err:
            print(f"⚠️ Warning: Semantic search during upload failed: {sim_err}")
            
        conn.commit()
        
        # Execute LangGraph Judicial Reconciliation Agent
        reconciliation = {}
        try:
            if not RECONCILIATION_AGENT:
                print("🏛️ Loading Judicial Reconciliation Agent...")
                from agents.reconciliation.reconciliation_agent import JudicialReconciliationAgent
                RECONCILIATION_AGENT = JudicialReconciliationAgent()

            reconciliation = RECONCILIATION_AGENT.reconcile({
                "case_id": filename_id,
                "case_type": case_type,
                "legal_regime": regime,
                "summary": raw_summary_string,
                "priority_score": float(priority_score) if priority_score is not None else 5.0,
                "severity": float(severity) if severity is not None else 3.0,
                "contributions": contributions,
                "bias_score": bias_score,
                "bias_details": bias_details,
                "bias_flags": bias_flags_list,
                "similar_cases": similar_cases
            })
        except Exception as rec_err:
            print(f"[WARNING] Cache miss reconciliation failed: {rec_err}")

        return {
            "case_id": filename_id,
            "summary": raw_summary_string,
            "severity": severity,
            "regime": regime,
            "ipc_sections": ipc_str,
            "bns_sections": bns_str,
            "priority_score": priority_score,
            "case_type": case_type,
            "priority_explanation": explanation,
            "similar_cases": similar_cases,
            "contributions": contributions,
            "bias_score": bias_score,
            "bias_details": bias_details,
            "bias_flags": bias_flags_list,
            "reconciliation": reconciliation
        }
        
    except Exception as e:
        print(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@app.post("/search")
async def search_cases(req: SearchRequest):
    global PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT, PRECEDENT_SEARCH_AGENT
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not PRIORITIZATION_AGENT:
        from agents.prioritization.prioritizer import PrioritizationAgent
        PRIORITIZATION_AGENT = PrioritizationAgent()
    if not EXPLAINABILITY_AGENT:
        from agents.explainability.explainability_agent import ExplainabilityAgent
        EXPLAINABILITY_AGENT = ExplainabilityAgent()

    if not BIAS_AGENT:
        from agents.fairness.bias_agent import BiasAgent
        BIAS_AGENT = BiasAgent()
        
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Retrieve per-case Legal-BERT vector distances. BM25 ranks the same
        # persisted case chunks; the two ranked lists are fused below with RRF.
        if req.yearFilter:
            sql = """
                SELECT 
                    c.case_id, 
                    MIN(cc.embedding <=> %s::vector) AS min_dist,
                    c.case_date, c.max_severity_score, c.case_summary,
                    c.bns_sections, c.ipc_sections, c.priority_score, c.case_type,
                    c.num_ipc_sections, c.num_cpc_sections, c.num_precedents,
                    c.case_age_days, c.immediate_threat_flag, c.societal_impact_score,
                    c.total_words, c.bias_score, c.bias_details, c.bias_flags
                FROM case_chunks cc
                JOIN cases c ON cc.case_id = c.case_id
                WHERE c.case_date LIKE %s OR c.case_id LIKE %s
                GROUP BY c.case_id, c.case_date, c.max_severity_score, c.case_summary, c.bns_sections, c.ipc_sections, c.priority_score, c.case_type, c.num_ipc_sections, c.num_cpc_sections, c.num_precedents, c.case_age_days, c.immediate_threat_flag, c.societal_impact_score, c.total_words, c.bias_score, c.bias_details, c.bias_flags;
            """
            year_param = f"%{req.yearFilter}%"
            cur.execute(sql, (query_emb.tolist(), year_param, year_param))
        else:
            sql = """
                SELECT 
                    c.case_id, 
                    MIN(cc.embedding <=> %s::vector) AS min_dist,
                    c.case_date, c.max_severity_score, c.case_summary,
                    c.bns_sections, c.ipc_sections, c.priority_score, c.case_type,
                    c.num_ipc_sections, c.num_cpc_sections, c.num_precedents,
                    c.case_age_days, c.immediate_threat_flag, c.societal_impact_score,
                    c.total_words, c.bias_score, c.bias_details, c.bias_flags
                FROM case_chunks cc
                JOIN cases c ON cc.case_id = c.case_id
                GROUP BY c.case_id, c.case_date, c.max_severity_score, c.case_summary, c.bns_sections, c.ipc_sections, c.priority_score, c.case_type, c.num_ipc_sections, c.num_cpc_sections, c.num_precedents, c.case_age_days, c.immediate_threat_flag, c.societal_impact_score, c.total_words, c.bias_score, c.bias_details, c.bias_flags;
            """
            cur.execute(sql, (query_emb.tolist(),))

        records = cur.fetchall()

        top_records = search_bm25_legalbert_rrf(req.query, records, req.topK)

        results = []
        for score, r in top_records:
            # Generate dynamic explanation trace
            exp = PRIORITIZATION_AGENT.generate_explanation(
                case_type=r[8] if r[8] else "Unknown",
                num_ipc_sections=r[9] if r[9] is not None else 0,
                num_cpc_sections=r[10] if r[10] is not None else 0,
                case_age_days=r[12] if r[12] is not None else None,
                num_precedents=r[11] if r[11] is not None else 0,
                severity=float(r[3]) if r[3] is not None else 3.0,
                immediate_threat=r[13] if r[13] is not None else 0,
                societal_impact=r[14] if r[14] is not None else 1,
                case_date=r[2] if r[2] else None
            )

            # Generate Explainability attributions
            features_for_explain = {
                "case_type": r[8] if r[8] else "civil",
                "num_ipc_sections": r[9] if r[9] is not None else 0,
                "num_cpc_sections": r[10] if r[10] is not None else 0,
                "num_precedents": r[11] if r[11] is not None else 0,
                "total_words": r[15] if r[15] is not None else 1000,
                "case_age_days": r[12] if r[12] is not None else 365.0,
                "max_severity_score": r[3] if r[3] is not None else 3.0,
                "immediate_threat_flag": r[13] if r[13] is not None else 0,
                "societal_impact_score": r[14] if r[14] is not None else 1
            }
            contributions = EXPLAINABILITY_AGENT.compute_marginal_attributions(features_for_explain)
            
            results.append({
                "case_id": r[0],
                "score": round(float(score), 6),
                "date": r[2] if r[2] else "Unknown",
                "severity": r[3],
                "summary": r[4] if r[4] else "AI Synopsis Pending Processing.",
                "bns_sections": r[5] if r[5] else "",
                "ipc_sections": r[6] if r[6] else "",
                "priority_score": r[7] if r[7] is not None else 0.0,
                "case_type": r[8] if r[8] else "Unknown",
                "priority_explanation": exp,
                "contributions": contributions,
                "bias_score": r[16],
                "bias_details": r[17] if r[17] is not None else "Automated audit unavailable.",
                "bias_flags": r[18].split(",") if (r[18] and r[18].strip()) else []
            })
            
        return {"results": results}
    except Exception as e:
        print(f"Precedent search failed: {e}")
        raise HTTPException(status_code=500, detail="Precedent search failed")

@app.get("/cases/pending")
async def get_pending_priority_queue(
    limit: int = 50,
    offset: int = 0,
    case_type: str = None,
    legal_regime: str = None
):
    """
    Returns pending cases ordered by AI Priority Score (highest priority first).
    Includes summary statistics and agent explanation traces.
    """
    global PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT
    if not PRIORITIZATION_AGENT:
        from agents.prioritization.prioritizer import PrioritizationAgent
        PRIORITIZATION_AGENT = PrioritizationAgent()
    if not EXPLAINABILITY_AGENT:
        from agents.explainability.explainability_agent import ExplainabilityAgent
        EXPLAINABILITY_AGENT = ExplainabilityAgent()

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        # Build dynamic WHERE clause
        where_clauses = []
        params = []
        if case_type and case_type.lower() != "all":
            where_clauses.append("LOWER(case_type) = %s")
            params.append(case_type.lower())
        if legal_regime and legal_regime.lower() != "all":
            where_clauses.append("LOWER(legal_regime) = %s")
            params.append(legal_regime.lower())

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # Query stats
        stats_sql = f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT(*) FILTER (WHERE priority_score >= 8.0) as critical_count,
                COUNT(*) FILTER (WHERE priority_score >= 6.0 AND priority_score < 8.0) as high_count,
                AVG(priority_score) as avg_priority
            FROM cases {where_sql};
        """
        cur.execute(stats_sql, params)
        stats_row = cur.fetchone()
        total_count = stats_row[0] or 0
        critical_count = stats_row[1] or 0
        high_count = stats_row[2] or 0
        avg_priority = round(float(stats_row[3]), 2) if stats_row[3] is not None else 0.0

        # Query ordered cases
        query_sql = f"""
            SELECT 
                case_id, priority_score, case_type, legal_regime, case_date,
                max_severity_score, case_summary, bns_sections, ipc_sections,
                num_ipc_sections, num_cpc_sections, num_precedents, case_age_days,
                immediate_threat_flag, societal_impact_score, total_words,
                bias_score, bias_details, bias_flags
            FROM cases
            {where_sql}
            ORDER BY priority_score DESC NULLS LAST, case_age_days DESC NULLS LAST
            LIMIT %s OFFSET %s;
        """
        cur.execute(query_sql, params + [limit, offset])
        records = cur.fetchall()

        results = []
        for idx, r in enumerate(records):
            rank = offset + idx + 1
            
            exp = PRIORITIZATION_AGENT.generate_explanation(
                case_type=r[2] if r[2] else "Unknown",
                num_ipc_sections=r[9] if r[9] is not None else 0,
                num_cpc_sections=r[10] if r[10] is not None else 0,
                case_age_days=r[12] if r[12] is not None else None,
                num_precedents=r[11] if r[11] is not None else 0,
                severity=float(r[5]) if r[5] is not None else 3.0,
                immediate_threat=r[13] if r[13] is not None else 0,
                societal_impact=r[14] if r[14] is not None else 1,
                case_date=r[4] if r[4] else None
            )

            features_for_explain = {
                "case_type": r[2] if r[2] else "civil",
                "num_ipc_sections": r[9] if r[9] is not None else 0,
                "num_cpc_sections": r[10] if r[10] is not None else 0,
                "num_precedents": r[11] if r[11] is not None else 0,
                "total_words": r[15] if r[15] is not None else 1000,
                "case_age_days": r[12] if r[12] is not None else 365.0,
                "max_severity_score": float(r[5]) if r[5] is not None else 3.0,
                "immediate_threat_flag": r[13] if r[13] is not None else 0,
                "societal_impact_score": r[14] if r[14] is not None else 1
            }
            contributions = EXPLAINABILITY_AGENT.compute_marginal_attributions(features_for_explain)

            results.append({
                "rank": rank,
                "case_id": r[0],
                "priority_score": round(float(r[1]), 2) if r[1] is not None else 0.0,
                "case_type": r[2] if r[2] else "Unknown",
                "legal_regime": r[3] if r[3] else "Unknown",
                "date": r[4] if r[4] else "Unknown",
                "severity": r[5] if r[5] is not None else 3.0,
                "summary": r[6] if r[6] else "AI Synopsis Pending Processing.",
                "bns_sections": r[7] if r[7] else "",
                "ipc_sections": r[8] if r[8] else "",
                "priority_explanation": exp,
                "contributions": contributions,
                "bias_score": r[16],
                "bias_details": r[17] if r[17] is not None else "Automated audit unavailable.",
                "bias_flags": r[18].split(",") if (r[18] and r[18].strip()) else []
            })

        return {
            "total_count": total_count,
            "critical_count": critical_count,
            "high_count": high_count,
            "avg_priority": avg_priority,
            "results": results
        }
    except Exception as e:
        print(f"Error fetching pending priority queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch pending priority queue")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

@app.post("/cases/{case_id}/reconcile")
async def reconcile_case_by_id(case_id: str):
    """
    On-demand endpoint to run LangGraph Judicial Reconciliation Agent on any existing case.
    """
    global PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT, RECONCILIATION_AGENT
    if not RECONCILIATION_AGENT:
        print("🏛️ Loading Judicial Reconciliation Agent...")
        from agents.reconciliation.reconciliation_agent import JudicialReconciliationAgent
        RECONCILIATION_AGENT = JudicialReconciliationAgent()
    if not EXPLAINABILITY_AGENT:
        from agents.explainability.explainability_agent import ExplainabilityAgent
        EXPLAINABILITY_AGENT = ExplainabilityAgent()

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT case_id, priority_score, case_type, legal_regime, case_date,
                   max_severity_score, case_summary, bns_sections, ipc_sections,
                   num_ipc_sections, num_cpc_sections, num_precedents, case_age_days,
                   immediate_threat_flag, societal_impact_score, total_words,
                   bias_score, bias_details, bias_flags
            FROM cases WHERE case_id = %s
        """, (case_id,))
        r = cur.fetchone()
        if not r:
            raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

        features_for_explain = {
            "case_type": r[2] if r[2] else "civil",
            "num_ipc_sections": r[9] if r[9] is not None else 0,
            "num_cpc_sections": r[10] if r[10] is not None else 0,
            "num_precedents": r[11] if r[11] is not None else 0,
            "total_words": r[15] if r[15] is not None else 1000,
            "case_age_days": r[12] if r[12] is not None else 365.0,
            "max_severity_score": float(r[5]) if r[5] is not None else 3.0,
            "immediate_threat_flag": r[13] if r[13] is not None else 0,
            "societal_impact_score": r[14] if r[14] is not None else 1
        }
        contributions = EXPLAINABILITY_AGENT.compute_marginal_attributions(features_for_explain)

        # Retrieve similar precedents
        similar_cases = []
        try:
            cur.execute("SELECT embedding FROM case_chunks WHERE case_id = %s LIMIT 1", (case_id,))
            emb_res = cur.fetchone()
            if emb_res:
                cur.execute("""
                    SELECT DISTINCT ON (c.case_id) c.case_id, cc.embedding <=> %s::vector AS distance
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    WHERE c.case_id != %s
                    ORDER BY c.case_id, distance ASC
                    LIMIT 3;
                """, (emb_res[0], case_id))
                sim_records = cur.fetchall()
                for sr in sim_records:
                    similar_cases.append({"case_id": sr[0], "score": round(1 - float(sr[1]), 4)})
        except Exception:
            pass

        reconciliation = RECONCILIATION_AGENT.reconcile({
            "case_id": r[0],
            "case_type": r[2] if r[2] else "civil",
            "legal_regime": r[3] if r[3] else "ipc",
            "summary": r[6] if r[6] else "",
            "priority_score": float(r[1]) if r[1] is not None else 5.0,
            "severity": float(r[5]) if r[5] is not None else 3.0,
            "contributions": contributions,
            "bias_score": r[16],
            "bias_details": r[17] if r[17] is not None else "",
            "bias_flags": r[18].split(",") if (r[18] and r[18].strip()) else [],
            "similar_cases": similar_cases
        })

        return {
            "case_id": case_id,
            "reconciliation": reconciliation
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in on-demand reconciliation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

from typing import List

@app.post("/batch_summarize_upload")
async def process_batch_pdf_upload(files: List[UploadFile] = File(...)):
    """
    Accepts multiple PDF files, runs full agent pipeline on each,
    and returns all cases ordered by priority score descending.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided for batch processing")

    processed_results = []

    for file in files:
        try:
            res = await process_pdf_upload(file=file)
            processed_results.append(res)
        except Exception as err:
            print(f"Failed processing file {file.filename}: {err}")

    # Sort results by priority_score descending
    processed_results.sort(key=lambda x: x.get("priority_score", 0.0) or 0.0, reverse=True)

    # Assign priority rank
    for idx, item in enumerate(processed_results):
        item["rank"] = idx + 1

    return {
        "count": len(processed_results),
        "results": processed_results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

