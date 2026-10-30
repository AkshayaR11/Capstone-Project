# save as check_search.py and run
import psycopg2, json
import sys
sys.path.append(".")
from config import DATABASE_URL



conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("""
    SELECT case_id FROM cases 
    WHERE legal_regime = 'BNS' 
    AND (bns_sections IS NULL OR bns_sections = '')
    LIMIT 3
""")
bns_cases = [r[0] for r in cur.fetchall()]
conn.close()

for case_id in bns_cases:
    with open(f"data/chunks/{case_id}_chunks.json") as f:
        import json
        data = json.load(f)
    
    # Search for BNS mentions in first 10 chunks
    print(f"\n=== {case_id} ===")
    for chunk in data["chunks"][:10]:
        text = chunk["text"].lower()
        if any(kw in text for kw in ["bns", "bharatiya", "sanhita", "section"]):
            print(chunk["text"][:300])
            print("---")