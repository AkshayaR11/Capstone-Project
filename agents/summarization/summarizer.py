import psycopg2
import json
import os
import time
import sys
from pgvector.psycopg2 import register_vector
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATABASE_URL

class SummarizationAgent:
    def __init__(self):
        print("🚀 Initializing Summarization Agent...")

        # Gemini API Initialization
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            print("Warning: GEMINI_API_KEY environment variable is not set.")
            self.gemini_model = None

        # DB
        self.conn = psycopg2.connect(DATABASE_URL)
        register_vector(self.conn)

        # Cache
        self.chunk_cache = {}
        print("✅ Summarization Agent ready.\n")

    def _call_gemini(self, prompt, retries=3):
        """Generates content via Gemini model with exponential backoff on 429 quota exceptions."""
        if not self.gemini_model:
            return "Gemini API unavailable (missing key)."

        for attempt in range(retries):
            try:
                return self.gemini_model.generate_content(prompt).text.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                    sleep_time = (2 ** attempt) * 5
                    print(f"   ⚠️ Gemini 429 Rate Limit. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise
        return "Summary failed"

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

    def generate_structured_summary(self, case_id):
        print(f"\n🧠 Processing case: {case_id}")

        chunk_map = self.load_chunks(case_id)
        if not chunk_map:
            return "Summary failed"

        all_text = " ".join(chunk_map.values())[:80000]
        
        prompt = """You are a legal assistant specializing in Indian law (IPC, BNS, CPC, CrPC).
Analyze this Supreme Court judgment and output exactly 4 sections:

## FACTS
Parties, background, FIR details, sections invoked, timeline.

## ISSUES
The specific legal questions the court had to decide.

## REASONING
Court's analysis, precedents cited (with citations), principles applied.

## JUDGMENT
Final order — allowed/dismissed/quashed, with section references and relief granted.

Be precise. Use Indian legal terminology. Do not add information not present in the text.

JUDGMENT TEXT:
""" + all_text

        return self._call_gemini(prompt)

    def summarize_upload_stream(self, text):
        print("\n🧠 Sending dynamically uploaded PDF directly to Gemini Cloud...")
        prompt = """You are a legal assistant specializing in Indian law (IPC, BNS, CPC, CrPC).
Analyze this Supreme Court judgment and output exactly 4 sections:

## FACTS
Parties, background, FIR details, sections invoked, timeline.

## ISSUES
The specific legal questions the court had to decide.

## REASONING
Court's analysis, precedents cited (with citations), principles applied.

## JUDGMENT
Final order — allowed/dismissed/quashed, with section references and relief granted.

Be precise. Use Indian legal terminology. Do not add information not present in the text.

JUDGMENT TEXT:
""" + text[:80000]

        return self._call_gemini(prompt)

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

if __name__ == "__main__":
    agent = SummarizationAgent()
    agent.run_batch(limit=3)