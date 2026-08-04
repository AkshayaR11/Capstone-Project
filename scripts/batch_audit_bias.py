import psycopg2
import json
import os
import time
import sys
from pathlib import Path

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL
from agents.fairness.bias_agent import BiasAgent

def batch_audit_cases(limit=5):
    print(f"Connecting to database at {DATABASE_URL}...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Find cases without bias score
    cur.execute("""
        SELECT case_id FROM cases 
        WHERE bias_score IS NULL
        LIMIT %s;
    """, (limit,))
    cases = cur.fetchall()

    if not cases:
        print("No cases pending audit.")
        cur.close()
        conn.close()
        return

    print(f"Found {len(cases)} cases pending audit. Initializing BiasAgent...")
    agent = BiasAgent()

    for idx, (case_id,) in enumerate(cases):
        print(f"\n[{idx+1}/{len(cases)}] Auditing case: {case_id}...")
        
        # Load case text from chunks file
        filepath = os.path.join("data/chunks", f"{case_id}_chunks.json")
        if not os.path.exists(filepath):
            print(f"   ⚠️ WARNING: Chunks file not found for {case_id}. Skipping.")
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            text = " ".join(c["text"] for c in chunk_data.get("chunks", []))
            
            if not text.strip():
                print("   ⚠️ Empty text found. Skipping.")
                continue

            # Run Audit
            audit_result = agent.audit_case_fairness(text)
            bias_score = audit_result.get("bias_score")
            bias_details = audit_result.get("bias_details")
            bias_flags_str = ",".join(audit_result.get("flags", []))

            # Update DB
            update_cur = conn.cursor()
            update_cur.execute("""
                UPDATE cases 
                SET bias_score = %s, bias_details = %s, bias_flags = %s
                WHERE case_id = %s
            """, (bias_score, bias_details, bias_flags_str, case_id))
            conn.commit()
            update_cur.close()

            print(f"   Neutrality score: {bias_score} | Details: {bias_details}")
            
            # Respect rate limit if Gemini is active
            if agent.gemini_model is not None:
                print("   Sleeping 4s to respect Gemini API rate limits...")
                time.sleep(4)

        except Exception as e:
            print(f"   Error auditing case {case_id}: {e}")

    cur.close()
    conn.close()
    print("\nBatch audit complete.")

if __name__ == "__main__":
    # Audit 5 cases by default
    batch_audit_cases(limit=5)
