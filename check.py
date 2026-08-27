# check_summarizer.py  
import sys
sys.path.append(".")
from agents.summarization.summarizer import SummarizationAgent
import psycopg2
from config import DATABASE_URL

agent = SummarizationAgent()

# Pick a case that has a summary stored
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("SELECT case_id, case_summary FROM cases WHERE case_summary IS NOT NULL LIMIT 1")
row = cur.fetchone()
conn.close()

if row:
    print(f"Case: {row[0]}")
    print(f"Summary length: {len(row[1])} chars")
    print(f"Has FACTS section: {'## FACTS' in row[1] or 'FACTS' in row[1]}")
    print(f"Has ISSUES section: {'## ISSUES' in row[1] or 'ISSUES' in row[1]}")
    print(f"Has REASONING section: {'## REASONING' in row[1] or 'REASONING' in row[1]}")
    print(f"Has JUDGMENT section: {'## JUDGMENT' in row[1] or 'JUDGMENT' in row[1]}")
    print("\nFirst 300 chars:")
    print(row[1][:300])