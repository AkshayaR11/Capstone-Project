import psycopg2

def fix_dates():
    print("Fixing bad dates in DB...")
    conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
    cur = conn.cursor()
    cur.execute("UPDATE cases SET case_date = 'Unknown' WHERE case_date ILIKE '% OF %' OR case_date ILIKE '% No %';")
    print(f"Update completed. {cur.rowcount} rows affected.")
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_dates()
