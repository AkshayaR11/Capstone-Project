"""
Migrate local data (CSV, PKL, NPY) into PostgreSQL database using pgvector
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from tqdm import tqdm

DB_URL = "postgresql://admin:password@localhost:5432/judicial"

FEATURES_FILE = "data/prioritization/case_features.csv"
INDEX_FILE = "data/embeddings/chunk_index.pkl"
EMBEDDINGS_FILE = "data/embeddings/all_embeddings.npy"

def setup_database():
    print("🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. Enable pgvector
    print("📦 Enabling pgvector extension...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    # 2. Create tables
    print("🛠️ Creating tables...")
    cur.execute("""
        DROP TABLE IF EXISTS case_chunks;
        DROP TABLE IF EXISTS cases;
        
        CREATE TABLE cases (
            case_id VARCHAR PRIMARY KEY,
            case_type VARCHAR,
            legal_regime VARCHAR,
            ipc_sections TEXT,
            num_ipc_sections INTEGER,
            bns_sections TEXT,
            num_bns_sections INTEGER,
            cpc_sections TEXT,
            num_cpc_sections INTEGER,
            num_precedents INTEGER,
            total_words INTEGER,
            case_date VARCHAR,
            case_age_days FLOAT,
            max_severity_score FLOAT,
            immediate_threat_flag INTEGER,
            societal_impact_score INTEGER,
            priority_score FLOAT,
            case_summary TEXT
        );
        
        CREATE TABLE case_chunks (
            id SERIAL PRIMARY KEY,
            case_id VARCHAR REFERENCES cases(case_id),
            chunk_id INTEGER,
            word_count INTEGER,
            chunk_date VARCHAR,
            court VARCHAR,
            embedding VECTOR(768)
        );
    """)
    return conn, cur

def migrate_cases(conn, cur):
    print(f"\n📄 Loading metadata from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    
    # Handle NaN values explicitly
    df = df.where(pd.notnull(df), None)
    
    records = df.to_dict(orient="records")
    
    insert_query = """
        INSERT INTO cases (
            case_id, case_type, legal_regime, ipc_sections, num_ipc_sections, 
            bns_sections, num_bns_sections, cpc_sections, num_cpc_sections, 
            num_precedents, total_words, case_date, case_age_days,
            max_severity_score, immediate_threat_flag, societal_impact_score, priority_score
        ) VALUES %s
        ON CONFLICT (case_id) DO NOTHING;
    """
    
    data = [(
        r["case_id"], r["case_type"], r.get("legal_regime"), r.get("ipc_sections"), r["num_ipc_sections"],
        r.get("bns_sections"), r.get("num_bns_sections"), r.get("cpc_sections"), r["num_cpc_sections"], 
        r["num_precedents"], r["total_words"], r.get("case_date"), r.get("case_age_days"), 
        r.get("max_severity_score"), r.get("immediate_threat_flag"), r.get("societal_impact_score"), r.get("priority_score")
    ) for r in records]
    
    print(f"📥 Inserting {len(data)} cases into Postgres...")
    execute_values(cur, insert_query, data, page_size=100)
    print("✅ Cases inserted.")


def migrate_embeddings(conn, cur):
    print(f"\n🧠 Loading embeddings matrix from {EMBEDDINGS_FILE}...")
    embeddings = np.load(EMBEDDINGS_FILE)
    
    print(f"📦 Loading chunk indices from {INDEX_FILE}...")
    with open(INDEX_FILE, "rb") as f:
        metadata = pickle.load(f)
    
    assert len(embeddings) == len(metadata), "Mismatch between embeddings count and metadata index!"
    
    insert_query = """
        INSERT INTO case_chunks (case_id, chunk_id, word_count, chunk_date, court, embedding)
        VALUES %s;
    """
    
    print(f"📥 Bulk inserting {len(metadata)} embedding chunks into Postgres...")
    
    # Process in batches of 1000
    batch_size = 1000
    for i in tqdm(range(0, len(metadata), batch_size)):
        batch_meta = metadata[i:i+batch_size]
        batch_emb = embeddings[i:i+batch_size]
        
        data = []
        for j, meta in enumerate(batch_meta):
            data.append((
                meta["case_id"],          # case_id
                meta["chunk_id"],         # chunk_id
                meta["word_count"],       # word_count
                meta.get("date"),         # chunk_date
                meta.get("court"),        # court 
                batch_emb[j].tolist()     # embedding (VECTOR array)
            ))
            
        execute_values(cur, insert_query, data)
        
    print(f"✅ Extracted {len(metadata)} chunks with 768-D Vectors inserted.")


if __name__ == "__main__":
    try:
        conn, cur = setup_database()
        migrate_cases(conn, cur)
        migrate_embeddings(conn, cur)
        
        print("\n🎉 MIGRATION COMPLETE!")
        print("💡 The database is now ready for semantic search using pgvector.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"\n❌ Error during migration: {str(e)}")
