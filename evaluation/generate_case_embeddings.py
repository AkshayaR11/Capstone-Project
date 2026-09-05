# save as generate_case_embeddings.py
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
import sys
sys.path.append(".")
from config import DATABASE_URL

print("Connecting...")
conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)
cur = conn.cursor()

# Get all case IDs
cur.execute("SELECT case_id FROM cases")
case_ids = [r[0] for r in cur.fetchall()]
print(f"Processing {len(case_ids)} cases...")

updated = 0
for case_id in case_ids:
    # Fetch all chunk embeddings for this case
    cur.execute("""
        SELECT embedding FROM case_chunks 
        WHERE case_id = %s
    """, (case_id,))
    rows = cur.fetchall()
    
    if not rows:
        continue
    
    # Average all chunk embeddings → single case embedding
    embeddings = np.array([np.array(r[0]) for r in rows])
    case_emb = embeddings.mean(axis=0)
    
    # Normalize
    case_emb = case_emb / np.linalg.norm(case_emb)
    
    # Store
    update_cur = conn.cursor()
    update_cur.execute("""
        UPDATE cases SET case_embedding = %s 
        WHERE case_id = %s
    """, (case_emb.tolist(), case_id))
    update_cur.close()
    updated += 1

conn.commit()
print(f"Updated {updated} case embeddings")

# Add index
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_cases_embedding
    ON cases USING ivfflat (case_embedding vector_cosine_ops)
    WITH (lists = 50);
""")
conn.commit()
cur.close()
conn.close()
print("Done.")