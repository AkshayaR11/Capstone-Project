import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import json
import os

class SummarizationAgent:
    def __init__(self):
        print("🚀 Initializing agent...")

        # Summarizer (keep small for speed)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # LegalBERT
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # DB
        self.conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        register_vector(self.conn)

        # Cache
        self.chunk_cache = {}

        # Precompute query embeddings (VERY IMPORTANT)
        self.query_embeddings = {
            "facts": self.get_embedding("background facts evidence witnesses FIR case details"),
            "issues": self.get_embedding("legal issues questions before the court dispute law"),
            "reasoning": self.get_embedding("court reasoning analysis interpretation held observed"),
            "judgment": self.get_embedding("final decision verdict appeal allowed dismissed set aside")
        }

        print("✅ Agent ready.\n")

    # ==========================
    # EMBEDDING
    # ==========================
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**inputs)

        emb = output.last_hidden_state.mean(dim=1)
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0].numpy()

    # ==========================
    # LOAD CHUNKS (CACHED)
    # ==========================
    def load_chunks(self, case_id):
        if case_id in self.chunk_cache:
            return self.chunk_cache[case_id]

        filepath = os.path.join("data/chunks", f"{case_id}_chunks.json")

        if not os.path.exists(filepath):
            return {}

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunk_map = {c["chunk_id"]: c["text"] for c in data["chunks"]}
        self.chunk_cache[case_id] = chunk_map
        return chunk_map

    # ==========================
    # RETRIEVE TOP CHUNKS
    # ==========================
    def get_top_chunks(self, case_id, section, k=5):
        query_embedding = self.query_embeddings[section]

        cur = self.conn.cursor()

        cur.execute("""
            SELECT chunk_id
            FROM case_chunks
            WHERE case_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s;
        """, (case_id, query_embedding, k))

        results = cur.fetchall()

        chunk_map = self.load_chunks(case_id)

        chunks = []
        for (chunk_id,) in results:
            if chunk_id in chunk_map:
                chunks.append(chunk_map[chunk_id])

        # Deduplicate
        chunks = list(set(chunks))

        return chunks

    # ==========================
    # FILTER BY KEYWORDS
    # ==========================
    def filter_chunks(self, chunks, section):
        keywords = {
            "facts": [
                "appellant", "respondent", "facts", "background",
                "case of", "filed", "transaction", "agreement"
            ],

            "issues": [
                "whether", "issue", "question", "point for determination"
            ],

            "reasoning": [
                "held", "observed", "reason", "considered", "it was argued"
            ],

            "judgment": [
                "appeal allowed", "appeal dismissed",
                "conviction set aside", "order", "judgment"
            ]
        }

        filtered = []
        for chunk in chunks:
            if any(word in chunk.lower() for word in keywords[section]):
                filtered.append(chunk)

        return filtered if filtered else chunks

    # ==========================
    # SUMMARIZE (FAST)
    # ==========================
    def summarize_chunks(self, chunks):
        if not chunks:
            return "Not available"

        # Combine top chunks and strictly limit size to prevent PyTorch tensor crash
        combined = " ".join(chunks[:3])[:3000]

        try:
            result = self.summarizer(
                combined,
                max_length=100,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            return result[0]["summary_text"]
        except:
            return "Summary failed"

    # ==========================
    # MAIN STRUCTURED SUMMARY
    # ==========================
    def generate_structured_summary(self, case_id):
        print(f"\n🧠 Processing case: {case_id}")

        sections = ["facts", "issues", "reasoning", "judgment"]
        final_summary = {}

        for section in sections:
            print(f"   🔍 {section}")

            chunks = self.get_top_chunks(case_id, section)
            chunks = self.filter_chunks(chunks, section)

            summary = self.summarize_chunks(chunks)
            final_summary[section] = summary

        return final_summary

    # ==========================
    # STREAM UPLOAD PROCESSING (IN-MEMORY BYPASS)
    # ==========================
    def summarize_upload_stream(self, text):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        import numpy as np
        
        print("\n🧠 Processing dynamically uploaded PDF...")
        
        # 1. Split Text In-Memory
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        raw_chunks = text_splitter.split_text(text)
        
        if not raw_chunks:
            return {"facts": "N/A", "issues": "N/A", "reasoning": "N/A", "judgment": "N/A"}
            
        # 2. Vectorize Array
        print("   🔍 Generating Semantic Matrix for Upload...")
        chunk_embeddings = [self.get_embedding(c) for c in raw_chunks]
        
        sections = ["facts", "issues", "reasoning", "judgment"]
        final_summary = {}
        
        for section in sections:
            print(f"   🔍 Extracting Context: {section}")
            query_emb = self.query_embeddings[section]
            
            # 3. Calculate Cosine Similarities Natively (Bypassing pgvector logic)
            similarities = []
            for emb in chunk_embeddings:
                sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
                similarities.append(sim)
                
            # Get Top 5 Chunks
            top_k_idx = np.argsort(similarities)[-5:][::-1]
            top_chunks = [raw_chunks[i] for i in top_k_idx]
            
            # Filter and Summarize using existing optimized functions
            filtered_chunks = self.filter_chunks(top_chunks, section)
            final_summary[section] = self.summarize_chunks(filtered_chunks)
            
        return final_summary

    # ==========================
    # RUN BATCH
    # ==========================
    def run_batch(self, limit=10):
        cur = self.conn.cursor()

        cur.execute("""
            SELECT case_id FROM cases
            WHERE case_summary IS NULL
            LIMIT %s;
        """, (limit,))

        cases = cur.fetchall()

        for (case_id,) in cases:
            structured = self.generate_structured_summary(case_id)

            summary_text = f"""
FACTS:
{structured['facts']}

ISSUES:
{structured['issues']}

REASONING:
{structured['reasoning']}

JUDGMENT:
{structured['judgment']}
"""

            update_cur = self.conn.cursor()
            update_cur.execute("""
                UPDATE cases
                SET case_summary = %s
                WHERE case_id = %s
            """, (summary_text, case_id))

            self.conn.commit()
            update_cur.close()

            print("   ✅ Stored.\n")

        cur.close()
        self.conn.close()
        print("🏁 Done.")
        

# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    agent = SummarizationAgent()
    agent.run_batch(limit=10)