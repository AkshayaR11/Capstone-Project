import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
import json
import os

class SummarizationAgent:
    def __init__(self):
        print("🚀 Initializing agent...")

        # Gemini API Initialization (Replaces local offline LLMs)
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # We use Flash because it natively takes 1M Tokens and is blazing fast
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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
    def summarize_chunks(self, chunks, section_name="legal aspect"):
        if not chunks:
            return "Not available"

        combined = " ".join(chunks[:3])[:3000]

        try:
            import time
            time.sleep(4) # Respect Gemini Free Tier 15 RPM
            
            prompt = f"Analyze this exact judicial text focusing primarily on the '{section_name}'. Provide a highly concise, incredibly accurate paragraph summarizing ONLY the facts relating to the '{section_name}':\n\n{combined}"
            response = self.gemini_model.generate_content(prompt)
            
            return response.text.replace("\n", " ").strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "Summary failed"

    # ==========================
    # MAIN STRUCTURED SUMMARY (1-SHOT GEMINI GENERATOR)
    # ==========================
    def generate_structured_summary(self, case_id):
        print(f"\n🧠 Processing case: {case_id}")

        # Gemini digests massive contexts so we can just grab everything at once!
        chunk_map = self.load_chunks(case_id)
        if not chunk_map:
            return "Summary failed"

        # Combine entirely natively
        all_text = " ".join(chunk_map.values())[:500000]

        try:
            import time
            print("   ⏳ Respecting Gemini Rate Limits (Waiting 15s)...")
            time.sleep(15) # Safe for Gemini 2.5 Flash (Max 4 requests per minute to stay under 5 RPM)
            
            prompt = """You are generating structured summaries of Indian Supreme Court and High Court judgments.
Your objective is to produce accurate, faithful, concise, and legally useful summaries suitable for judges, lawyers, researchers, and legal assistants.

Follow these requirements strictly.

# General Principles
* Extract information only from the judgment.
* Never invent facts, reasoning, holdings, legal principles, or implications.
* Never speculate.
* Preserve the Court's reasoning faithfully.
* Maintain neutral legal language.
* Prioritize correctness over completeness.
* Do not omit the final operative order.

# Output Format and Hierarchy
Generate exactly two levels of summary, maintaining this precise hierarchical structure:

LEVEL 1 — Executive Summary
### Facts
(One short paragraph: Parties, material facts, relevant procedural history)
### Issues
(2-5 concise bullet points of legal questions decided)
### Holding
(Clear statement of whether appeal was allowed/dismissed, relief granted, final decision)
### Key Reasoning
(3-6 concise bullet points of the Court's reasoning)
### Final Operative Order
(Orders passed, costs, directions, relief)

LEVEL 2 — Detailed Analysis (Expandable)
### Facts
(Parties, background, material events, procedural history, lower court findings)
### Issues
(Actual issues decided, no duplicates)
### Reasoning
(For each issue, explain the reasoning, applied legal principles, and relevant statutes/precedents. For precedents, explain why they were relevant, how they were applied, and whether they were followed, distinguished, or rejected.)
#### Majority Opinion
#### Concurring Opinion (if any)
#### Dissenting Opinion (if any)
### Final Judgment
(Outcome, relief, costs, operative directions)
### Key Legal Principles
(3-6 concise bullet points describing the legal principles established. Do not include practical advice, policy commentary, real-world impact, or speculation.)

Here is the precedent text:
""" + all_text
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "Summary failed"

    # ==========================
    # STREAM UPLOAD PROCESSING (IN-MEMORY BYPASS)
    # ==========================
    def summarize_upload_stream(self, text):
        print("\n🧠 Sending dynamically uploaded PDF directly to Gemini Cloud...")
        try:
            prompt = """You are generating structured summaries of Indian Supreme Court and High Court judgments.
Your objective is to produce accurate, faithful, concise, and legally useful summaries suitable for judges, lawyers, researchers, and legal assistants.

Follow these requirements strictly.

# General Principles
* Extract information only from the judgment.
* Never invent facts, reasoning, holdings, legal principles, or implications.
* Never speculate.
* Preserve the Court's reasoning faithfully.
* Maintain neutral legal language.
* Prioritize correctness over completeness.
* Do not omit the final operative order.

# Output Format and Hierarchy
Generate exactly two levels of summary, maintaining this precise hierarchical structure:

LEVEL 1 — Executive Summary
### Facts
(One short paragraph: Parties, material facts, relevant procedural history)
### Issues
(2-5 concise bullet points of legal questions decided)
### Holding
(Clear statement of whether appeal was allowed/dismissed, relief granted, final decision)
### Key Reasoning
(3-6 concise bullet points of the Court's reasoning)
### Final Operative Order
(Orders passed, costs, directions, relief)

LEVEL 2 — Detailed Analysis (Expandable)
### Facts
(Parties, background, material events, procedural history, lower court findings)
### Issues
(Actual issues decided, no duplicates)
### Reasoning
(For each issue, explain the reasoning, applied legal principles, and relevant statutes/precedents. For precedents, explain why they were relevant, how they were applied, and whether they were followed, distinguished, or rejected.)
#### Majority Opinion
#### Concurring Opinion (if any)
#### Dissenting Opinion (if any)
### Final Judgment
(Outcome, relief, costs, operative directions)
### Key Legal Principles
(3-6 concise bullet points describing the legal principles established. Do not include practical advice, policy commentary, real-world impact, or speculation.)

Here is the precedent text:
""" + text[:500000]
            
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Upload Summary Failed: {e}")
            return "Failed to process PDF via Gemini."

    # ==========================
    # RUN BATCH
    # ==========================
    def run_batch(self, limit=3):
        cur = self.conn.cursor()

        cur.execute("""
            SELECT case_id FROM cases
            WHERE case_summary IS NULL
            LIMIT %s;
        """, (limit,))

        cases = cur.fetchall()

        for (case_id,) in cases:
            summary_text = self.generate_structured_summary(case_id)

            if summary_text == "Summary failed":
                print("   ❌ Skipping Case (Error/Timeout)\n")
                continue

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
    agent.run_batch(limit=3)