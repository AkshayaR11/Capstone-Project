# save as check_summarizer.py and run
import psycopg2
import sys
sys.path.append(".")
from config import DATABASE_URL

conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()
cur.execute("""
    SELECT case_id, case_summary 
    FROM cases 
    WHERE case_summary IS NOT NULL
""")
rows = cur.fetchall()
conn.close()

print(f"Total summaries stored: {len(rows)}")

has_all   = 0
sections  = {"FACTS": 0, "ISSUES": 0, "REASONING": 0, "JUDGMENT": 0}
lengths   = []

for case_id, summary in rows:
    f = "FACTS"     in summary
    i = "ISSUES"    in summary
    r = "REASONING" in summary
    j = "JUDGMENT"  in summary
    if f: sections["FACTS"]     += 1
    if i: sections["ISSUES"]    += 1
    if r: sections["REASONING"] += 1
    if j: sections["JUDGMENT"]  += 1
    if all([f, i, r, j]):
        has_all += 1
    lengths.append(len(summary))

print(f"All 4 sections present: {has_all}/{len(rows)}")
print(f"Section counts: {sections}")
if lengths:
    print(f"Avg length: {sum(lengths)//len(lengths)} chars")
    print(f"Min/Max:    {min(lengths)} / {max(lengths)} chars")

if rows:
    print(f"\nSample case: {rows[0][0]}")
    print(rows[0][1][:400])