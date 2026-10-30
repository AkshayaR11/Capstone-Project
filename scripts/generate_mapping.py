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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IPC_BNS_PDF   = PROJECT_ROOT / "ipcbnsmapping.pdf"
CPC_PDF       = PROJECT_ROOT / "cpcsections.pdf"
OUTPUT_PATH   = PROJECT_ROOT / "data" / "mappings.json"


def clean(val):
    """Strip whitespace and newlines from a cell value."""
    return str(val).strip().replace("\n", " ") if val else ""


def parse_ipc_bns_entries(pdf_path: str) -> dict[str, set[str]]:
    """
    The PDF has columns: BNS Section | Subject | IPC Section | Summary.
    Keep every explicit IPC section listed in a cell (some rows list multiple
    IPC sections, e.g. "29 and 29A") as a key in {ipc_section: bns_section}.
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
                            mapping.setdefault(ipc_candidate, set()).add(bns_candidate)
                continue

            for row in rows:
                if not row or len(row) < 3:
                    continue

                bns_raw = clean(row[0])
                ipc_raw = clean(row[2]) if len(row) > 2 else ""

                # Normalise section numbers:  "2(3)" -> "2", "103A" -> "103A"
                bns_num = re.match(r'^(\d+[A-Za-z]?)', bns_raw)
                # Parenthesized numbers are IPC subsections, not separate
                # IPC sections. The cases table stores section numbers only,
                # so strip subsection numbers before extracting the keys.
                ipc_for_sections = re.sub(r'\bclauses?[-\s]*\d+\b', '', ipc_raw, flags=re.IGNORECASE)
                ipc_for_sections = re.sub(r'\(\s*\d+\s*\)', '', ipc_for_sections)
                ipc_for_sections = re.sub(r'(\d)\s*[-–]\s*([A-Za-z])', r'\1\2', ipc_for_sections)
                ipc_nums = re.findall(r'\d+[A-Za-z]?', ipc_for_sections)

                if bns_num and ipc_nums:
                    bns_val = bns_num.group(1)
                    for ipc_key in ipc_nums:
                        mapping.setdefault(ipc_key, set()).add(bns_val)

    return mapping


def parse_ipc_bns(pdf_path: str) -> dict:
    """Return only unambiguous IPC section -> BNS section mappings."""
    entries = parse_ipc_bns_entries(pdf_path)
    return {ipc: next(iter(bns_values)) for ipc, bns_values in entries.items()
            if len(bns_values) == 1}


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
    return sorted(sections, key=lambda x: (
        int(re.sub(r'[A-Za-z]', '', x)),
        re.sub(r'\d', '', x).lower(),
        x
    ))


def main():
    print("=" * 60)
    print("  Mapping Generator -- IPC -> BNS  |  CPC Sections")
    print("=" * 60)

    # 1. IPC -> BNS mapping
    if not IPC_BNS_PDF.exists():
        print(f"ERROR: '{IPC_BNS_PDF}' not found in project root. Aborting.")
        return

    print(f"\n[1/2] Parsing {IPC_BNS_PDF}...")
    ipc_entries = parse_ipc_bns_entries(IPC_BNS_PDF)
    ipc_to_bns = {ipc: next(iter(bns_values)) for ipc, bns_values in ipc_entries.items()
                  if len(bns_values) == 1}
    ambiguous_ipc_to_bns = {ipc: sorted(bns_values) for ipc, bns_values in ipc_entries.items()
                            if len(bns_values) > 1}
    print(f"      [OK] Extracted {len(ipc_to_bns)} IPC -> BNS mappings")
    print(f"      [INFO] {len(ambiguous_ipc_to_bns)} IPC sections have multiple BNS candidates")

    # Preview first 10
    for k, v in list(ipc_to_bns.items())[:10]:
        print(f"      IPC {k:>6}  ->  BNS {v}")

    # 2. CPC sections
    cpc_sections = []
    if CPC_PDF.exists():
        print(f"\n[2/2] Parsing {CPC_PDF}...")
        cpc_sections = parse_cpc_sections(CPC_PDF)
        print(f"      [OK] Extracted {len(cpc_sections)} CPC section references")
    else:
        print(f"\n[2/2] WARNING: '{CPC_PDF}' not found - skipping CPC extraction.")

    # 3. Save
    output = {
        "IPC_TO_BNS": ipc_to_bns,
        "IPC_TO_BNS_AMBIGUOUS": ambiguous_ipc_to_bns,
        "CPC_SECTIONS": cpc_sections
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"\n[DONE] Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
