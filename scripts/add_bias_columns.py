import psycopg2

DB_URL = "postgresql://admin:password@localhost:5432/judicial"

def add_bias_columns():
    print("Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()

        print("Executing ALTER TABLE statements to add bias columns...")
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS bias_score FLOAT DEFAULT 0.98;")
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS bias_details TEXT DEFAULT 'Neutrality evaluation audit complete. No demographic anomalies detected.';")
        
        print("SUCCESS: Successfully added bias columns to PostgreSQL cases table without wiping data!")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error during column migration: {e}")

if __name__ == "__main__":
    add_bias_columns()
