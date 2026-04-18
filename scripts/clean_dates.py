import psycopg2
from datetime import datetime

def fix_all_dates():
    print("Scanning database for invalid dates (like 36/1/2010)...")
    conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
    cur = conn.cursor()
    
    cur.execute("SELECT case_id, case_date FROM cases WHERE case_date != 'Unknown' AND case_date IS NOT NULL;")
    rows = cur.fetchall()

    bad_ids = []
    
    for case_id, d_str in rows:
        d_str = str(d_str).strip()
        parsed = False
        
        # Valid date formats expected from the original regex
        formats = [
            "%d/%m/%Y",
            "%d-%m-%Y", 
            "%d %B %Y",
            "%d %b %Y",
            "%m/%d/%Y",
            "%Y-%m-%d"
        ]
        
        for fmt in formats:
            try:
                # If Python's strict datetime parser accepts it, it's a real calendar date
                datetime.strptime(d_str, fmt)
                parsed = True
                break
            except ValueError:
                continue
                
        if not parsed:
            bad_ids.append(case_id)

    if bad_ids:
        print(f"🚨 Found {len(bad_ids)} mathematically impossible dates. Updating to 'Unknown'...")
        # Update them all in one batch query
        cur.execute("UPDATE cases SET case_date = 'Unknown' WHERE case_id = ANY(%s)", (bad_ids,))
        conn.commit()
    else:
        print("✅ No more bad dates found!")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_all_dates()
