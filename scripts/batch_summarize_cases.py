import os
import sys
import time
import psycopg2
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATABASE_URL
from agents.summarization.summarizer import SummarizationAgent

def run_batch_summarization(limit=None, sleep_sec=2.5):
    load_dotenv()
    print("[*] Initializing Summarization Agent...")
    agent = SummarizationAgent()
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    query = """
        SELECT c.case_id 
        FROM cases c
        WHERE (c.case_summary IS NULL 
               OR c.case_summary = '' 
               OR c.case_summary = 'AI Synopsis Pending Processing.')
        ORDER BY c.priority_score DESC NULLS LAST, c.case_id ASC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
        
    cur.execute(query)
    cases_to_process = cur.fetchall()
    total_needed = len(cases_to_process)
    print(f"\n[*] Total cases pending summarization: {total_needed}\n")

    if total_needed == 0:
        print("[+] All cases already have summaries!")
        conn.close()
        return

    processed_count = 0
    success_count = 0

    for idx, (case_id,) in enumerate(cases_to_process, 1):
        print(f"[{idx}/{total_needed}] Processing case: {case_id}...")
        try:
            summary = agent.generate_structured_summary(case_id)
            if summary and summary != "Summary failed" and "Gemini API unavailable" not in summary:
                cur.execute(
                    "UPDATE cases SET case_summary = %s WHERE case_id = %s",
                    (summary, case_id)
                )
                conn.commit()
                success_count += 1
                print(f"  [+] Saved summary for {case_id}")
            else:
                print(f"  [-] Skipping {case_id}: {summary}")
        except Exception as e:
            print(f"  [!] Error on {case_id}: {e}")
            conn.rollback()

        processed_count += 1
        time.sleep(sleep_sec)

    conn.close()
    print(f"\n[*] Finished! Successfully generated and saved {success_count}/{total_needed} case summaries.")

if __name__ == "__main__":
    limit_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_batch_summarization(limit=limit_arg, sleep_sec=2.5)
