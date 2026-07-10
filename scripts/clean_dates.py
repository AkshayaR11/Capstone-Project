import psycopg2
import re
from datetime import datetime

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

def parse_date_from_filename(filename: str) -> str | None:
    # Try pattern like _on_23_February_2012
    m = re.search(r'_on_(\d{1,2})_([A-Za-z]+)_(\d{4})', filename, re.IGNORECASE)
    if m:
        day = int(m.group(1))
        month_str = m.group(2).lower()
        year = int(m.group(3))
        if month_str in MONTH_MAP:
            month = MONTH_MAP[month_str]
            return f"{day:02d}/{month:02d}/{year}"
    return None

def fix_all_dates():
    print("Scanning database for invalid dates (like 36/1/2010)...")
    conn = psycopg2.connect("postgresql://admin:password@localhost:5432/judicial")
    cur = conn.cursor()
    
    cur.execute("SELECT case_id, case_date FROM cases;")
    rows = cur.fetchall()

    fixed_count = 0
    unknown_count = 0
    
    for case_id, d_str in rows:
        d_str = str(d_str).strip() if d_str else ""
        parsed = False
        
        # Valid date formats
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
                datetime.strptime(d_str, fmt)
                parsed = True
                break
            except ValueError:
                continue
                
        # If not parsed as a valid calendar date, try to extract from filename
        if not parsed:
            extracted = parse_date_from_filename(case_id)
            if extracted:
                cur.execute("UPDATE cases SET case_date = %s WHERE case_id = %s", (extracted, case_id))
                fixed_count += 1
            else:
                cur.execute("UPDATE cases SET case_date = 'Unknown' WHERE case_id = %s", (case_id,))
                unknown_count += 1

    conn.commit()
    print("Scanning complete!")
    print(f"   Fixed from filename: {fixed_count} cases")
    print(f"   Set to Unknown: {unknown_count} cases")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_all_dates()
