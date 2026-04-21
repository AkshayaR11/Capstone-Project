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
        # Standardize file string as Case UID
        case_id = file.filename.replace(".pdf", "").replace(" ", "_")
        
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
        
        # 2. CACHE SAVER
        if not result:
            cur.execute("INSERT INTO cases (case_id, max_severity_score) VALUES (%s, 5) ON CONFLICT DO NOTHING", (case_id,))
            
        cur.execute("UPDATE cases SET case_summary = %s WHERE case_id = %s", (raw_summary_string, case_id))
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
                        c.case_summary
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
                        c.case_summary
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
                "summary": r[4] if r[4] else "AI Synopsis Pending Processing."
            })
            
        return {"results": results}
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
