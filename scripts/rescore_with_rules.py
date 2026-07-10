"""
Re-score all cases in the database using the new legally-validated
rule-based prioritization formula (Dr. Saritha P, Phase 2).
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
from agents.prioritization.prioritizer import PrioritizationAgent

DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5432/judicial")

def rescore_all():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("""
        SELECT case_id, case_type, bns_sections, ipc_sections, case_age_days
        FROM cases
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} cases to re-score.")

    agent = PrioritizationAgent()
    updated = 0

    for row in rows:
        case_id, case_type, bns_sections, ipc_sections, age_days = row
        score = agent.compute_priority_score(
            case_type=case_type or "civil",
            bns_sections=bns_sections or "",
            ipc_sections=ipc_sections or "",
            case_age_days=age_days,
            text="",
        )
        cur.execute(
            "UPDATE cases SET priority_score = %s WHERE case_id = %s",
            (score, case_id)
        )
        updated += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"SUCCESS: Re-scored {updated} cases with the new legal formula.")

if __name__ == "__main__":
    rescore_all()
