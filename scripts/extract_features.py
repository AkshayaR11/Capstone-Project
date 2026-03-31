"""
Extract structured features from chunked legal cases
Improved for Indian Supreme Court dataset
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
CHUNKS_DIR = "data/chunks"
OUTPUT_DIR = "data/prioritization"
OUTPUT_FILE = "data/prioritization/case_features.csv"

Path(OUTPUT_DIR).mkdir(exist_ok=True)


# ==============================
# CASE TYPE DETECTION
# ==============================
def extract_case_type(text):
    text_lower = text.lower()

    # Criminal indicators
    if re.search(r'\bipc\b|indian penal code|criminal', text_lower):
        return "criminal"

    # Civil indicators
    if re.search(r'\bcpc\b|civil procedure|injunction|property|contract', text_lower):
        return "civil"

    # Default fallback
    return "civil"


# ==============================
# IPC EXTRACTION (STRICT)
# ==============================
def extract_ipc_sections(text):
    """
    Extract IPC sections ONLY when explicitly mentioned
    Avoids false positives like dates/page numbers
    """
    text_lower = text.lower()

    ipc_pattern = r'section\s+(\d{1,3}[a-z]?)\s*(?:ipc|indian penal code)'
    matches = re.findall(ipc_pattern, text_lower)

    return list(set(matches))


# ==============================
# CPC EXTRACTION
# ==============================
def extract_cpc_sections(text):
    """
    Extract CPC sections for civil cases
    """
    text_lower = text.lower()

    cpc_pattern = r'section\s+(\d{1,3})\s*(?:cpc|civil procedure)'
    matches = re.findall(cpc_pattern, text_lower)

    return list(set(matches))


# ==============================
# PRECEDENT COUNT
# ==============================
def count_precedents(text):
    """
    Count legal citations like:
    AIR 1990 SC 123
    2001 SCC 456
    """
    pattern = r'\b(?:air|scr|scc)\s+\d{4}'
    matches = re.findall(pattern, text.lower())

    return len(matches)


# ==============================
# DATE EXTRACTION
# ==============================
def extract_date(text):
    """
    Extract common legal date formats
    """
    patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}',
        r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',   # 11 April 2005
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return None


def parse_date(date_str):
    if not date_str:
        return None

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %B %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue

    return None


def calculate_case_age(date_obj):
    if not date_obj:
        return None

    return (datetime.now() - date_obj).days


# ==============================
# MAIN FEATURE EXTRACTION
# ==============================
def extract_all_features():
    chunk_files = list(Path(CHUNKS_DIR).glob("*_chunks.json"))

    print(f"📊 Processing {len(chunk_files)} cases...")

    all_features = []

    for chunk_file in tqdm(chunk_files):
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        case_id = data["case_id"]
        chunks = data["chunks"]

        # Combine full text
        full_text = " ".join([c["text"] for c in chunks])

        # ======================
        # FEATURE EXTRACTION
        # ======================

        case_type = extract_case_type(full_text)

        ipc_sections = []
        cpc_sections = []

        if case_type == "criminal":
            ipc_sections = extract_ipc_sections(full_text)
        else:
            cpc_sections = extract_cpc_sections(full_text)

        # Date
        date_str = extract_date(full_text)
        parsed_date = parse_date(date_str)
        case_age = calculate_case_age(parsed_date)

        # Precedents
        num_precedents = count_precedents(full_text)

        # Basic stats
        total_words = sum(c["word_count"] for c in chunks)

        # ======================
        # FINAL FEATURE OBJECT
        # ======================
        features = {
            "case_id": case_id,
            "case_type": case_type,

            "ipc_sections": ",".join(ipc_sections),
            "num_ipc_sections": len(ipc_sections),

            "cpc_sections": ",".join(cpc_sections),
            "num_cpc_sections": len(cpc_sections),

            "num_precedents": num_precedents,
            "total_words": total_words,

            "case_date": date_str,
            "case_age_days": case_age,
        }

        all_features.append(features)

    # ======================
    # SAVE CSV
    # ======================
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ Features extracted successfully")
    print(f"📁 Saved to: {OUTPUT_FILE}")

    print("\n📊 Summary:")
    print(df["case_type"].value_counts())

    return df


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    df = extract_all_features()

    print("\n📋 Sample Output:")
    print(df.head(10))