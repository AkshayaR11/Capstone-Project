"""
generate_mapping.py
Reads ipcbnsmapping.pdf and cpcsections.pdf and generates a clean
mappings.json file used by the DocumentProcessor to translate
old IPC section numbers to new BNS equivalents.

Run once from the project root:
    python scripts/generate_mapping.py
"""

import pdfplumber
import json
import re
import os

IPC_BNS_PDF   = "../ipcbnsmapping.pdf"
CPC_PDF       = "../cpcsections.pdf"
OUTPUT_PATH   = "data"


def clean(val):
    """Strip whitespace and newlines from a cell value."""
    return str(val).strip().replace("\n", " ") if val else ""


def parse_ipc_bns(pdf_path: str) -> dict:
    """
    The PDF has columns: BNS Section | Subject | IPC Section | Summary
    We want a dict: {ipc_section: bns_section}
    """
    mapping = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            rows = page.extract_table()
            if not rows:
                # Fallback: try to parse raw text lines for this page
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    # Look for patterns like:  103    Murder    302
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        bns_candidate = parts[0].rstrip("()")
                        ipc_candidate = parts[-1]
                        if re.match(r'^\d+[A-Za-z]?$', bns_candidate) and \
                           re.match(r'^\d+[A-Za-z]?$', ipc_candidate):
                            mapping[ipc_candidate] = bns_candidate
                continue

            for row in rows:
                if not row or len(row) < 3:
                    continue

                bns_raw = clean(row[0])
                ipc_raw = clean(row[2]) if len(row) > 2 else ""

                # Normalise section numbers:  "2(3)" -> "2", "103A" -> "103A"
                bns_num = re.match(r'^(\d+[A-Za-z]?)', bns_raw)
                ipc_num = re.match(r'^(\d+[A-Za-z]?)', ipc_raw)

                if bns_num and ipc_num:
                    ipc_key = ipc_num.group(1)
                    bns_val = bns_num.group(1)
                    # Skip if IPC and BNS are identical (just renamed)
                    mapping[ipc_key] = bns_val

    return mapping


def parse_cpc_sections(pdf_path: str) -> list:
    """
    Extract all CPC section numbers for reference / validation.
    Returns a sorted list like ["1", "2", "3", ..., "158"]
    """
    sections = set()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            # Match "Section 123" or standalone numbers at start of line
            found = re.findall(r'\b(\d{1,3}[A-Za-z]?)\b', text)
            for f in found:
                if 1 <= int(re.sub(r'[A-Za-z]', '', f) or 0) <= 160:
                    sections.add(f)
    return sorted(sections, key=lambda x: int(re.sub(r'[A-Za-z]', '', x)))


def main():
    print("=" * 60)
    print("  Mapping Generator -- IPC -> BNS  |  CPC Sections")
    print("=" * 60)

    # 1. IPC -> BNS mapping
    if not os.path.exists(IPC_BNS_PDF):
        print(f"ERROR: '{IPC_BNS_PDF}' not found in project root. Aborting.")
        return

    print(f"\n[1/2] Parsing {IPC_BNS_PDF}...")
    ipc_to_bns = parse_ipc_bns(IPC_BNS_PDF)
    print(f"      [OK] Extracted {len(ipc_to_bns)} IPC -> BNS mappings")

    # Preview first 10
    for k, v in list(ipc_to_bns.items())[:10]:
        print(f"      IPC {k:>6}  ->  BNS {v}")

    # 2. CPC sections
    cpc_sections = []
    if os.path.exists(CPC_PDF):
        print(f"\n[2/2] Parsing {CPC_PDF}...")
        cpc_sections = parse_cpc_sections(CPC_PDF)
        print(f"      [OK] Extracted {len(cpc_sections)} CPC section references")
    else:
        print(f"\n[2/2] WARNING: '{CPC_PDF}' not found - skipping CPC extraction.")

    # 3. Save
    output = {
        "IPC_TO_BNS": ipc_to_bns,
        "CPC_SECTIONS": cpc_sections
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
