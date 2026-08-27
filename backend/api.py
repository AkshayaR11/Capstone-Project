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
import hashlib
from datetime import datetime

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

@app.post("/summarize_upload")
async def process_pdf_upload(file: UploadFile = File(...)):
    global UPLOAD_AGENT, PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT, BIAS_AGENT
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
            
            return {
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
                "bias_flags": result[9].split(",") if (result[9] and result[9].strip()) else []
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
        
        return {
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
            "bias_flags": bias_flags_list
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
    global PRIORITIZATION_AGENT, EXPLAINABILITY_AGENT, BIAS_AGENT
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # Generate Semantic Vector
    encoded_input = tokenizer([req.query.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = F.normalize(query_emb, p=2, dim=1).numpy()[0]
    
    # Initialize Prioritization, Explainability, & Bias Agents
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
        
        # Deduplicate results directly via PostgreSQL using DISTINCT ON
        if req.yearFilter:
            sql = """
                SELECT * FROM (
                    SELECT DISTINCT ON (c.case_id) 
                        c.case_id, 
                        cc.embedding <=> %s::vector AS distance, 
                        c.case_date, 
                        c.max_severity_score,
                        c.case_summary,
                        c.bns_sections,
                        c.ipc_sections,
                        c.priority_score,
                        c.case_type,
                        c.num_ipc_sections,
                        c.num_cpc_sections,
                        c.num_precedents,
                        c.case_age_days,
                        c.immediate_threat_flag,
                        c.societal_impact_score,
                        c.total_words,
                        c.bias_score,
                        c.bias_details,
                        c.bias_flags
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    WHERE c.case_date LIKE %s OR c.case_id LIKE %s
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT %s;
            """
            year_param = f"%{req.yearFilter}%"
            cur.execute(sql, (query_emb.tolist(), year_param, year_param, req.topK))
        else:
            sql = """
                SELECT * FROM (
                    SELECT DISTINCT ON (c.case_id) 
                        c.case_id, 
                        cc.embedding <=> %s::vector AS distance, 
                        c.case_date, 
                        c.max_severity_score,
                        c.case_summary,
                        c.bns_sections,
                        c.ipc_sections,
                        c.priority_score,
                        c.case_type,
                        c.num_ipc_sections,
                        c.num_cpc_sections,
                        c.num_precedents,
                        c.case_age_days,
                        c.immediate_threat_flag,
                        c.societal_impact_score,
                        c.total_words,
                        c.bias_score,
                        c.bias_details,
                        c.bias_flags
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT %s;
            """
            cur.execute(sql, (query_emb.tolist(), req.topK))
            
        records = cur.fetchall()
        
        results = []
        for r in records:
            sim_score = 1 - float(r[1])
            
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
                "score": round(sim_score, 4),
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
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
