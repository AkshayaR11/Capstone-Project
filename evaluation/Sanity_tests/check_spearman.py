# save as check_spearman.py and run
from scipy import stats
import psycopg2
import sys
sys.path.append(".")
from config import DATABASE_URL

conn = psycopg2.connect(DATABASE_URL)
cur  = conn.cursor()
cur.execute("""
    SELECT max_severity_score, ROUND(AVG(priority_score)::numeric, 2)
    FROM cases
    WHERE case_type = 'criminal'
    GROUP BY max_severity_score
    HAVING COUNT(*) >= 3
    ORDER BY max_severity_score DESC
""")
rows = cur.fetchall()
conn.close()

severity = [r[0] for r in rows]
priority = [r[1] for r in rows]

rho, pval = stats.spearmanr(severity, priority)
print("=== SPEARMAN CORRELATION ===")
print(f"Severity scores: {severity}")
print(f"Mean priorities: {priority}")
print(f"Spearman rho:    {rho:.3f}")
print(f"P-value:         {pval:.4f}")
print(f"Significant?     {pval < 0.05}")
print(f"Strong positive? {rho > 0.7}")