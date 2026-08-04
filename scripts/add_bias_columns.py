import psycopg2
import sys
import os

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

def run_migrations():
    print(f"Connecting to PostgreSQL at {DATABASE_URL}...")
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        cur = conn.cursor()

        print("Executing ALTER TABLE statements to add/modify columns...")
        
        # 1. Ensure bias_score exists and its default is NULL
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS bias_score FLOAT DEFAULT NULL;")
        cur.execute("ALTER TABLE cases ALTER COLUMN bias_score SET DEFAULT NULL;")
        
        # 2. Ensure bias_details exists and its default is NULL (or empty)
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS bias_details TEXT DEFAULT NULL;")
        cur.execute("ALTER TABLE cases ALTER COLUMN bias_details SET DEFAULT NULL;")

        # 3. Ensure bias_flags exists
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS bias_flags TEXT DEFAULT NULL;")

        # 4. Ensure content_hash exists
        cur.execute("ALTER TABLE cases ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) UNIQUE DEFAULT NULL;")
        
        print("SUCCESS: Successfully executed database migrations without wiping data!")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error during migration: {e}")

if __name__ == "__main__":
    run_migrations()
