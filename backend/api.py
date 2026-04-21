from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import os

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

# Lazy-loaded Summarization Agent to save boot RAM
UPLOAD_AGENT = None

@app.post("/summarize_upload")
async def process_pdf_upload(file: UploadFile = File(...)):
    global UPLOAD_AGENT
    try:
        # Standardize file string as Case UID without case-sensitive extensions
        case_id = file.filename
        if case_id.lower().endswith(".pdf"):
            case_id = case_id[:-4]
        case_id = case_id.replace(" ", "_")
        
        # 1. SMART CACHE LOOKUP
        conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        cur = conn.cursor()
        cur.execute("SELECT case_summary FROM cases WHERE case_id = %s", (case_id,))
        result = cur.fetchone()
        
        if result and result[0]:
            print(f"🔥 Fast Cache Hit! Returned {case_id} instantly without hitting Gemini.")
            cur.close()
            conn.close()
            return {"summary": result[0]}
            
        # No Cache Found -> Read PDF Stream
        contents = await file.read()
        
        import fitz
        doc = fitz.open(stream=contents, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
        # Instance Agent Once
        if not UPLOAD_AGENT:
            print("🚀 Loading Gemini API Node...")
            import sys, os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from agents.summarization.summarizer import SummarizationAgent
            UPLOAD_AGENT = SummarizationAgent()
            
        # Gemerates raw string via Gemini
        raw_summary_string = UPLOAD_AGENT.summarize_upload_stream(text)
        
        # EXTRACT FEATURES NATIVELY
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from scripts.extract_features import (
            extract_case_type, detect_legal_regime, extract_ipc_sections, extract_bns_sections,
            load_mappings, calculate_severity, build_bns_severity, translate_ipc_to_bns, IPC_SEVERITY
        )
        
        ipc_to_bns, _ = load_mappings("data/mappings.json")
        bns_severity_map = build_bns_severity(ipc_to_bns)
        
        case_type = extract_case_type(text)
        regime = detect_legal_regime(None, text)
        ipc_sections_list = []
        bns_sections_list = []
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
                
        bns_str = ",".join(bns_sections_list)
        ipc_str = ",".join(ipc_sections_list)
        
        # 2. CACHE SAVER
        if not result:
            cur.execute("""
                INSERT INTO cases (case_id, max_severity_score, legal_regime, ipc_sections, bns_sections) 
                VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING
            """, (case_id, severity, regime, ipc_str, bns_str))
            
        cur.execute("""
            UPDATE cases 
            SET case_summary = %s, max_severity_score = %s, legal_regime = %s, ipc_sections = %s, bns_sections = %s
            WHERE case_id = %s
        """, (raw_summary_string, severity, regime, ipc_str, bns_str, case_id))
        
        conn.commit()
        cur.close()
        conn.close()

        return {"summary": raw_summary_string}
        
    except Exception as e:
        print(f"Extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_cases(req: SearchRequest):
    if not req.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    # Generate Semantic Vector
    encoded_input = tokenizer([req.query.lower()], padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    query_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    query_emb = F.normalize(query_emb, p=2, dim=1).numpy()[0]
    
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
                        c.ipc_sections
                    FROM case_chunks cc
                    JOIN cases c ON cc.case_id = c.case_id
                    WHERE c.case_id LIKE %s
                    ORDER BY c.case_id, distance ASC
                ) AS distinct_matches
                ORDER BY distance ASC
                LIMIT %s;
            """
            search_param = f"%{req.query}%"
            cur.execute(sql, (search_param, query_emb.tolist(), f"%_{req.yearFilter}_%", req.topK))
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
                        c.ipc_sections
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
        cur.close()
        conn.close()
        
        results = []
        for r in records:
            sim_score = 1 - float(r[1])
            results.append({
                "case_id": r[0],
                "score": round(sim_score, 4),
                "date": r[2] if r[2] else "Unknown",
                "severity": r[3],
                "summary": r[4] if r[4] else "AI Synopsis Pending Processing.",
                "bns_sections": r[5] if r[5] else "",
                "ipc_sections": r[6] if r[6] else ""
            })
            
        return {"results": results}
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
