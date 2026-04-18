"""
Offline Summarization Agent
Iterates through all un-summarized cases in PostgreSQL, merges chunks, and applies facebook/bart-large-cnn.
"""
import psycopg2
from transformers import pipeline
import textwrap

class SummarizationAgent:
    def __init__(self):
        print("Loading HuggingFace BART Summarization Pipeline (this may take a few minutes)...")
        # Load pipeline forcing cpu/gpu safely. max_length dictates max output length.
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
        print("✅ Local AI Agent Online.")

    def run_batch_summaries(self, limit=10):
        print(f"Fetching up to {limit} cases for summarization...")
        cur = self.conn.cursor()
        
        # We only want cases that literally have NO summary generated yet
        cur.execute("SELECT case_id FROM cases WHERE case_summary IS NULL LIMIT %s;", (limit,))
        cases = cur.fetchall()
        
        for (case_id,) in cases:
            print(f"\n🧠 Synthesizing Case: {case_id}")
            
            # Fetch the raw text file directly from local filesystem since Postgres only stores Vectors here
            try:
                import os, sys
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
                from scripts.clean_text import clean_legal_text
                
                # We pull from extracted_text since clean_text is run on-the-fly during pipeline ingestion
                filepath = os.path.join("data", "extracted_text", f"{case_id}.txt")
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_context = f.read()
                    
                # We MUST clean the raw text to strip the Massive Headers, otherwise the LLM just summarizes the Citations!
                full_context = clean_legal_text(raw_context)
                
            except FileNotFoundError:
                print(f"   => [FAILED] Raw text file not found for {case_id}")
                continue
            
            # AI models have strict positional embeddings (BART limit is strictly 1024 tokens). Let's slice raw context safely to 1800 characters to guarantee stability.
            truncated_context = full_context[:1800]
            
            try:
                # Execution of the Local LLM Summarizer
                result = self.summarizer(truncated_context, max_length=130, min_length=30, do_sample=False)
                summary_text = result[0]['summary_text']
                
                print(f"   => [SUCCESS] Summary Generated: {textwrap.shorten(summary_text, width=60, placeholder='...')}")
                
                # Push the exact string back up to PostgreSQL so the Web-UI can read it instantly
                update_cur = self.conn.cursor()
                update_cur.execute("UPDATE cases SET case_summary = %s WHERE case_id = %s", (summary_text, case_id))
                self.conn.commit()
                update_cur.close()
                
            except Exception as e:
                print(f"   => [FAILED] {e}")

        cur.close()
        self.conn.close()
        print("\n🏁 Batch Summarization Cycle Complete!")

if __name__ == "__main__":
    agent = SummarizationAgent()
    # For testing, we run just 5 cases so it doesn't freeze your laptop for 2 hours!
    agent.run_batch_summaries(limit=5)
