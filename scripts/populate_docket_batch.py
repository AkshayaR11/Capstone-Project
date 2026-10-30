"""
Populate Docket Batch Script
Sequentially populates case summaries (Gemini 2.5 Flash) and fairness audits
for unpopulated cases in PostgreSQL, with automatic rate-limit smoothing.
"""

import os
import sys
import json
import time
import argparse
import psycopg2
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL
from agents.summarization.summarizer import SummarizationAgent
from agents.fairness.bias_agent import BiasAgent

def run_batch_population(target_count=50, sleep_delay=4.0):
    print(f"\n========================================================")
    print(f"  JUDICIAL AI DOCKET POPULATION PIPELINE (Target: {target_count})")
    print(f"========================================================\n")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Query cases where summary or bias score is missing
    cur.execute("""
        SELECT case_id FROM cases
        WHERE case_summary IS NULL OR bias_score IS NULL
        ORDER BY priority_score DESC NULLS LAST
        LIMIT %s;
    """, (target_count,))
    cases_to_process = [r[0] for r in cur.fetchall()]
    cur.close()

    if not cases_to_process:
        print("[INFO] All cases in the database are already fully populated!")
        conn.close()
        return

    print(f"[INFO] Initializing SummarizationAgent and BiasAgent...")
    summarizer = SummarizationAgent()
    bias_agent = BiasAgent()

    successful = 0
    total = len(cases_to_process)

    print(f"[INFO] Beginning sequential processing of {total} cases...\n")

    for idx, case_id in enumerate(cases_to_process):
        print(f"[{idx+1}/{total}] Processing Case: {case_id}")
        
        # Load text chunks
        chunk_file = os.path.join("data", "chunks", f"{case_id}_chunks.json")
        raw_text = ""
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                raw_text = " ".join(c["text"] for c in cdata.get("chunks", []))
            except Exception as e:
                print(f"   ⚠️ Could not read chunks: {e}")

        if not raw_text.strip():
            print("   ⚠️ No text available for case. Skipping.")
            continue

        try:
            # 1. Check if summary already exists; if not, generate
            cur = conn.cursor()
            cur.execute("SELECT case_summary, bias_score FROM cases WHERE case_id = %s", (case_id,))
            current_row = cur.fetchone()
            existing_summary, existing_bias = current_row if current_row else (None, None)

            summary_text = existing_summary
            if not summary_text or len(summary_text) < 30:
                print("   🧠 Generating structured 4-part legal summary via Gemini...")
                summary_text = summarizer.generate_structured_summary(case_id)
                if not summary_text or summary_text == "Summary failed":
                    print("   ❌ Summary failed. Skipping.")
                    continue
                time.sleep(sleep_delay / 2.0)

            # 2. Check if bias audit already exists; if not, audit
            bias_score = existing_bias
            bias_details = ""
            bias_flags_str = ""
            if bias_score is None:
                print("   ⚖️ Running judicial fairness audit...")
                audit_res = bias_agent.audit_case_fairness(raw_text)
                bias_score = audit_res.get("bias_score")
                bias_details = audit_res.get("bias_details", "")
                bias_flags_str = ",".join(audit_res.get("flags", []))
                time.sleep(sleep_delay / 2.0)

            # 3. Update PostgreSQL
            cur.execute("""
                UPDATE cases 
                SET case_summary = %s, bias_score = %s, bias_details = %s, bias_flags = %s
                WHERE case_id = %s;
            """, (summary_text, bias_score, bias_details, bias_flags_str, case_id))
            conn.commit()
            cur.close()

            successful += 1
            print(f"   ✅ Successfully saved (Neutrality: {bias_score if bias_score is not None else 'N/A'})\n")

            # Rate limit pacing
            time.sleep(sleep_delay)

        except Exception as err:
            print(f"   ⚠️ Error processing {case_id}: {err}")
            conn.rollback()

    conn.close()
    print(f"\n========================================================")
    print(f"  BATCH POPULATION COMPLETE: {successful}/{total} cases updated")
    print(f"========================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch populate case summaries and bias audits.")
    parser.add_argument("--count", type=int, default=50, help="Number of cases to process (default: 50)")
    parser.add_argument("--delay", type=float, default=4.0, help="Sleep delay in seconds between API calls (default: 4.0)")
    args = parser.parse_args()

    run_batch_population(target_count=args.count, sleep_delay=args.delay)
