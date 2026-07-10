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

# Ensure parent directory is in sys.path for agent and script imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI(title="Judicial AI Search Engine")

# React lives on 5173, so we must allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/summarize_upload")
async def process_pdf_upload(file: UploadFile = File(...)):
    global UPLOAD_AGENT, PRIORITIZATION_AGENT
    conn = None
    cur = None
    try:
        # Standardize file string as Case UID without case-sensitive extensions
        case_id = file.filename
        if case_id.lower().endswith(".pdf"):
            case_id = case_id[:-4]
        case_id = case_id.replace(" ", "_")
        
        # 1. SMART CACHE LOOKUP
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        cur.execute("""
            SELECT case_summary, max_severity_score, legal_regime, ipc_sections, bns_sections, priority_score, case_type 
            FROM cases WHERE case_id = %s
        """, (case_id,))
        result = cur.fetchone()
        
        # Initialize Prioritization Agent
        if not PRIORITIZATION_AGENT:
            print("🚀 Loading Prioritization Agent...")
            from agents.prioritization.prioritizer import PrioritizationAgent
            PRIORITIZATION_AGENT = PrioritizationAgent()
            
        if result and result[0]:
            print(f"🔥 Fast Cache Hit! Returned {case_id} instantly without hitting Gemini.")
            
            # Generate explanation for cache hit
            ipc_sections_len = len(result[3].split(",")) if result[3] else 0
            bns_sections_len = len(result[4].split(",")) if result[4] else 0
            explanation = PRIORITIZATION_AGENT.generate_explanation(
                case_type=result[6] if result[6] else "Unknown",
                num_ipc_sections=max(ipc_sections_len, bns_sections_len),
                num_cpc_sections=0,
                case_age_days=0.0,
                num_precedents=5,
                bns_sections=result[4] if result[4] else "",
                ipc_sections=result[3] if result[3] else "",
            )
            
            # Retrieve similar cases using database case_chunks embedding
            similar_cases = []
            try:
                cur.execute("SELECT embedding FROM case_chunks WHERE case_id = %s LIMIT 1", (case_id,))
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
                    """, (emb_res[0], case_id))
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
                "similar_cases": similar_cases
            }
            
        # Cache Miss -> Close DB connections immediately so they aren't held open during Gemini execution
        if cur:
            cur.close()
        if conn:
            conn.close()
        conn = None
        cur = None

        # No Cache Found -> Read PDF Stream
        contents = await file.read()
        
        import fitz
        doc = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
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
            load_mappings, calculate_severity, build_bns_severity, translate_ipc_to_bns, IPC_SEVERITY
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
        
        # Calculate priority using legally-validated rule-based formula
        num_precedents = len(re.findall(r'\b(?:v\.|versus|scc|scr|air|cit)\b', text.lower()))
        priority_score = PRIORITIZATION_AGENT.compute_priority_score(
            case_type=case_type,
            bns_sections=bns_str,
            ipc_sections=ipc_str,
            case_age_days=0.0,
            text=text,
        )

        explanation = PRIORITIZATION_AGENT.generate_explanation(
            case_type=case_type,
            num_ipc_sections=len(ipc_sections_list),
            num_cpc_sections=len(cpc_sections_list),
            case_age_days=0.0,
            num_precedents=num_precedents,
            bns_sections=bns_str,
            ipc_sections=ipc_str,
            text=text,
        )
        
        # Now re-open DB connection to persist upload results
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        
        # 2. CACHE SAVER
        cur.execute("SELECT 1 FROM cases WHERE case_id = %s", (case_id,))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute("""
                INSERT INTO cases (case_id, max_severity_score, legal_regime, ipc_sections, bns_sections, priority_score, case_type) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (case_id, severity, regime, ipc_str, bns_str, priority_score, case_type))
        
        cur.execute("""
            UPDATE cases 
            SET case_summary = %s, max_severity_score = %s, legal_regime = %s, ipc_sections = %s, bns_sections = %s, priority_score = %s, case_type = %s
            WHERE case_id = %s
        """, (raw_summary_string, severity, regime, ipc_str, bns_str, priority_score, case_type, case_id))
        
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
            """, (upload_emb.tolist(), case_id))
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
            "similar_cases": similar_cases
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
    global PRIORITIZATION_AGENT
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # Generate Semantic Vector
    encoded_input = tokenizer([req.query.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = F.normalize(query_emb, p=2, dim=1).numpy()[0]
    
    # Initialize Prioritization Agent
    if not PRIORITIZATION_AGENT:
        from agents.prioritization.prioritizer import PrioritizationAgent
        PRIORITIZATION_AGENT = PrioritizationAgent()
        
    conn = None
    cur = None
    try:
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        
        # Deduplicate results directly via PostgreSQL using DISTINCT ON
        if req.yearFilter:
            # Subquery necessary for distinct + distance ordering
            sql = """
                SELECT * FROM (
                    SELECT DISTINCT ON (c.case_id) 
                        c.case_id, 
                        CASE WHEN REPLACE(c.case_id, '_', ' ') ILIKE %s THEN 0.0 ELSE cc.embedding <=> %s::vector END AS distance, 
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
                        c.case_age_days
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    WHERE c.case_date LIKE %s
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT %s;
            """
            search_param = f"%{req.query}%"
            cur.execute(sql, (search_param, query_emb.tolist(), f"%{req.yearFilter}%", req.topK))
        else:
            sql = """
                SELECT * FROM (
                    SELECT DISTINCT ON (c.case_id) 
                        c.case_id, 
                        CASE WHEN REPLACE(c.case_id, '_', ' ') ILIKE %s THEN 0.0 ELSE cc.embedding <=> %s::vector END AS distance, 
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
                        c.case_age_days
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT %s;
            """
            search_param = f"%{req.query}%"
            cur.execute(sql, (search_param, query_emb.tolist(), req.topK))
            
        records = cur.fetchall()
        
        results = []
        for r in records:
            sim_score = 1 - float(r[1])
            
            # Generate rule-based explanation trace
            exp = PRIORITIZATION_AGENT.generate_explanation(
                case_type=r[8] if r[8] else "Unknown",
                num_ipc_sections=r[9] if r[9] is not None else 0,
                num_cpc_sections=r[10] if r[10] is not None else 0,
                case_age_days=r[12] if r[12] is not None else None,
                num_precedents=r[11] if r[11] is not None else 0,
                bns_sections=r[5] if r[5] else "",
                ipc_sections=r[6] if r[6] else "",
            )
            
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
                "priority_explanation": exp
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

@app.get("/cases/{case_id}/summary")
async def get_case_summary(case_id: str):
    global UPLOAD_AGENT
    conn = None
    cur = None
    try:
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        
        # Check if already cached in DB
        cur.execute("SELECT case_summary FROM cases WHERE case_id = %s", (case_id,))
        res = cur.fetchone()
        
        if not res:
            raise HTTPException(status_code=404, detail="Case not found in database")
            
        if res[0] and res[0] != "AI Synopsis Pending Processing.":
            return {"summary": res[0]}
            
        # Not cached - check if the raw text file exists in data/extracted_text/
        txt_path = f"data/extracted_text/{case_id}.txt"
        if not os.path.exists(txt_path):
            raise HTTPException(status_code=404, detail=f"Case text file not found at {txt_path}")
            
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Case text file is empty")
            
        # Instantiate Summarization Agent if not loaded
        if not UPLOAD_AGENT:
            print("🚀 Loading Gemini API Node for on-demand summary...")
            from agents.summarization.summarizer import SummarizationAgent
            UPLOAD_AGENT = SummarizationAgent()
            
        # Generate summary
        print(f"🔮 Generating on-demand summary for {case_id}...")
        raw_summary_string = UPLOAD_AGENT.summarize_upload_stream(text)
        
        # Cache back to database
        cur.execute("UPDATE cases SET case_summary = %s WHERE case_id = %s", (raw_summary_string, case_id))
        conn.commit()
        
        return {"summary": raw_summary_string}
    except Exception as e:
        print(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cur: cur.close()
        if conn: conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
