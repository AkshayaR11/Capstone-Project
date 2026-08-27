import psycopg2
import re

def patch_dates():
    print("Connecting to PostgreSQL to patch dates from titles...")
    conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
    cur = conn.cursor()
    
    # Query all cases that have a bad or unknown date
    cur.execute("SELECT case_id, case_date FROM cases WHERE case_date = 'Unknown' OR case_date IS NULL;")
    rows = cur.fetchall()
    
    updated_count = 0
    # The regex targets the specific substring identifying the date in the filename
    # e.g.: State_vs_John_on_14_March_1950_1  --> extracts "14_March_1950"
    pattern = re.compile(r'_on_(\d{1,2}_[A-Za-z]+_\d{4})_')
    
    for row in rows:
        case_id = row[0]
        match = pattern.search(case_id)
        if match:
            # Reconstruct the date string natively (e.g., '10 October 2012')
            raw_date_str = match.group(1)
            fixed_date = raw_date_str.replace('_', ' ')
            
            cur.execute("UPDATE cases SET case_date = %s WHERE case_id = %s;", (fixed_date, case_id))
            updated_count += 1
            
    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Successfully reverse-engineered and patched {updated_count} completely Unknown dates from case titles!")

if __name__ == "__main__":
    patch_dates()
