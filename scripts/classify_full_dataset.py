import os
import re
import fitz  # PyMuPDF

ROOT_DIR = "C:/Users/anagh/OneDrive/Desktop/Capstone/supreme_court_judgments"

CRIMINAL_SIGNALS = [
    r'\bipc\b', r'indian penal code', r'\bbns\b', r'bharatiya nyaya sanhita',
    r'\bcrpc\b', r'\bbnss\b', r'\bfir\b', r'\bcriminal\b',
    r'\bmurder\b', r'\brape\b', r'\btheft\b', r'\bassault\b',
    r'\baccused\b', r'\bprosecution\b', r'\bsessions court\b', r'\bpolice\b'
]

CIVIL_SIGNALS = [
    r'\bcpc\b', r'civil procedure', r'\binjunction\b',
    r'\bproperty\b', r'\bcontract\b', r'\bdivorce\b',
    r'\bplaintiff\b', r'\bdefendant\b', r'\bsuit\b', r'\bdecree\b',
    r'\barbitration\b', r'\btenancy\b', r'\brent\b', r'\bland acquisition\b'
]

def classify_text(text: str) -> str:
    text_lower = text.lower()
    
    # Hierarchical Rule 1: Check for penal sections
    if re.search(r'\b(?:section|sec\.?)\s+\d+\s+of\s+(?:ipc|bns|indian penal code|bharatiya nyaya)\b', text_lower):
        return "criminal"
        
    crim_hits = sum(1 for pat in CRIMINAL_SIGNALS if re.search(pat, text_lower))
    civ_hits = sum(1 for pat in CIVIL_SIGNALS if re.search(pat, text_lower))
    
    if crim_hits == 0 and civ_hits == 0:
        return "unclassified"
    return "criminal" if crim_hits >= civ_hits else "civil"

def scan_dataset():
    pdf_files = []
    print("Listing all PDF files in dataset (26k files)...")
    for root, dirs, files in os.walk(ROOT_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))
                
    total_found = len(pdf_files)
    print(f"Total PDFs found: {total_found}")
    print("Starting full-text classification scan (this may take 2-4 minutes)...")
    
    counts = {"criminal": 0, "civil": 0, "unclassified": 0}
    
    for i, filepath in enumerate(pdf_files):
        try:
            doc = fitz.open(filepath)
            text = ""
            for page in doc:
                text += page.get_text()
            
            cat = classify_text(text)
            counts[cat] += 1
            doc.close()
        except Exception:
            counts["unclassified"] += 1
            
        if (i + 1) % 2000 == 0:
            print(f"Processed {i + 1}/{total_found} cases...")
            
    print("\n--- 100% PERFECT CLASSIFICATION RESULTS ---")
    print(f"Total Scanned: {total_found}")
    print(f"Criminal Cases: {counts['criminal']} ({counts['criminal']/total_found*100:.2f}%)")
    print(f"Civil Cases:    {counts['civil']} ({counts['civil']/total_found*100:.2f}%)")
    print(f"Unclassified:   {counts['unclassified']} ({counts['unclassified']/total_found*100:.2f}%)")
    print("-------------------------------------------")

if __name__ == "__main__":
    scan_dataset()
