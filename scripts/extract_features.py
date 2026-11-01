"""
extract_features.py
Extract structured features from chunked legal cases.
Handles Indian Supreme Court dataset (criminal + civil only).
Supports both IPC (pre-July 2024) and BNS (post-July 2024) regimes.

Run after chunk_text.py, before migrate_to_pg.py.
"""

import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import sys
import os

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MAPPINGS_FILE_PATH
from agents.prioritization.scoring import (
    calculate_societal_impact as _calculate_societal_impact,
    check_immediate_threat as _check_immediate_threat,
    compute_priority_score as _compute_priority_score,
)

# ==============================
# CONFIG
# ==============================
CHUNKS_DIR   = "data/chunks"
OUTPUT_DIR   = "data/prioritization"
OUTPUT_FILE  = "data/prioritization/case_features.csv"
MAPPINGS_FILE = MAPPINGS_FILE_PATH        # Centralized in config

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# BNS cutoff date
BNS_CUTOFF = datetime(2024, 7, 1)


# ==============================
# LOAD MAPPINGS
# ==============================
def load_mappings(path: str) -> tuple[dict, list]:
    """
    Load IPC->BNS mapping and CPC section list from mappings.json.
    Returns (ipc_to_bns dict, cpc_sections list).
    Falls back to empty structures if file is missing.
    """
    if not Path(path).exists():
        print(f"WARNING: {path} not found. Running without BNS mapping.")
        return {}, []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ipc_to_bns = data.get("IPC_TO_BNS", {})
    cpc_sections = data.get("CPC_SECTIONS", [])

    # Normalise all keys/values to lowercase stripped strings
    ipc_to_bns = {k.strip().lower(): v.strip() for k, v in ipc_to_bns.items()}

    print(f"Loaded {len(ipc_to_bns)} IPC->BNS mappings, {len(cpc_sections)} CPC section refs")
    return ipc_to_bns, cpc_sections


# ==============================
# CASE TYPE DETECTION
# (Strictly criminal or civil only, per legal expert guidance)
# ==============================
def extract_case_type(text: str) -> str:
    text_lower = text.lower()

    # Case-number labels are especially useful for SLP applications and affidavits,
    # which may not have the judgment-style appellate-jurisdiction heading.
    header_area = text_lower[:5000]
    slp_type = re.search(r"\bslp\s*\(\s*(civil|criminal|crl\.?)\s*\)", header_area)
    if slp_type:
        return "criminal" if slp_type.group(1).startswith("cr") else "civil"
    criminal_heading = re.search(
        r"\bslp\s*\(\s*(?:criminal|crl\.?)\s*\)|\bcriminal\s+(?:appellate\s+jurisdiction|appeal|writ)\b",
        header_area,
    )
    civil_heading = re.search(
        r"\bslp\s*\(\s*civil\s*\)|\bcivil\s+(?:appellate\s+jurisdiction|appeal|writ|original\s+jurisdiction)\b",
        header_area,
    )
    if criminal_heading:
        return "criminal"
    if civil_heading:
        return "civil"

    # 2. Keyword-based signals fallback
    criminal_signals = [
        r'\bipc\b', r'indian penal code', r'\bbns\b', r'bharatiya nyaya sanhita',
        r'\bcrpc\b', r'\bbnss\b', r'\bfir\b', r'\bcriminal\b',
        r'\bmurder\b', r'\brape\b', r'\btheft\b', r'\bassault\b',
        r'\baccused\b', r'\bprosecution\b', r'\bsessions court\b'
    ]
    civil_signals = [
        r'\bcpc\b', r'civil procedure', r'\binjunction\b',
        r'\bproperty\b', r'\bcontract\b', r'\bdivorce\b',
        r'\bplaintiff\b', r'\bdefendant\b', r'\bsuit\b', r'\bdecree\b',
        r'\barbitration\b', r'\btenancy\b'
    ]

    criminal_hits = sum(1 for pat in criminal_signals if re.search(pat, text_lower))
    civil_hits    = sum(1 for pat in civil_signals    if re.search(pat, text_lower))

    return "criminal" if criminal_hits >= civil_hits else "civil"


# ==============================
# LEGAL REGIME DETECTION
# ==============================
def detect_legal_regime(case_date: datetime | None, text: str) -> str:
    """
    Returns 'BNS' or 'IPC'.
    BNS (Bharatiya Nyaya Sanhita) replaced IPC on July 1 2024.
    """
    text_lower = text.lower()
    explicit_bns_sections = re.search(
        r"(?:sections?|u/s|under)\s+[\d,()/\s-]+\s+(?:of\s+the\s+)?(?:bns|bharatiya nyaya sanhita)", text_lower
    )
    explicit_ipc_sections = re.search(
        r"(?:sections?|u/s|under)\s+[\d,()/\s-]+\s+(?:of\s+the\s+)?(?:ipc|indian penal code)", text_lower
    )
    if explicit_bns_sections and not explicit_ipc_sections:
        return "BNS"
    if explicit_ipc_sections and not explicit_bns_sections:
        return "IPC"
    if re.search(r'\bbns\b|bharatiya nyaya sanhita|bharatiya nagarik suraksha', text_lower) and not re.search(r'\bipc\b|indian penal code', text_lower):
        return "BNS"
    if case_date and case_date >= BNS_CUTOFF:
        return "BNS"
    return "IPC"


# ==============================
# IPC SECTION EXTRACTION  (fixed regex)
# ==============================
def extract_ipc_sections(text: str) -> list[str]:
    """
    Handles real-world Indian judgment patterns:
      - u/s 302 IPC
      - Section 302/34 IPC
      - Sections 302, 307 read with 34 IPC
      - S. 376 IPC
      - under the Indian Penal Code, 1860 Sections 302 and 307
    """
    text_lower = text.lower()

    patterns = [
        # Explicit statute-first citations: IPC 302/34, Indian Penal Code 302.
        r'\bipc[, \t]*(?:sections?\s*)?([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)',
        r'indian penal code(?:,?\s*1860)?[, \t]*sections?\s+([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)',
        # u/s 302 ipc  |  u/s 302/34 ipc
        r'u/s\s+([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)\s+ipc',
        # section(s) 302, 307 ipc  |  section 302 ipc
        r'sections?\s+([\d]+[a-z]?(?:[,/\s]+(?:and\s+)?[\d]+[a-z]?)*)\s+(?:read\s+with\s+[\d]+[a-z]?\s+)?ipc',
        # s. 376 ipc
        r's\.\s*([\d]+[a-z]?)\s+ipc',
        # under ipc section 302
        r'(?:under|u/?s\.?)\s+(?:ipc|indian penal code)[,\s]+sections?\s+([\d]+[a-z]?)',
        # ...of the indian penal code, 1860 section 302
        r'indian penal code.*?section\s+([\d]+[a-z]?)',
    ]

    found = set()
    for pat in patterns:
        for match in re.findall(pat, text_lower):
            # match may be "302/34" or "302, 307" — split all of them
            parts = re.split(r'[,/\s]+', match)
            for p in parts:
                p = p.strip().rstrip('.')
                if p and re.fullmatch(r'\d{1,3}[a-z]?', p):
                    found.add(p)

    return sorted(found)


# ==============================
# BNS SECTION EXTRACTION
# ==============================
def extract_bns_sections(text: str) -> list[str]:
    """
    Extract BNS section numbers from post-July 2024 judgments.
    Patterns mirror the IPC ones but reference BNS / BNSS.
    """
    text_lower = text.lower()

    patterns = [
        # Explicit statute-first citations: BNS 103, Bharatiya Nyaya Sanhita 103.
        r'(?:\bbns\b|bharatiya nyaya sanhita)[,\s]+(?:sections?\s*)?([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)',
        r'u/s\s+([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)\s+bns',
        r'sections?\s+([\d]+[a-z]?(?:[,/\s]+(?:and\s+)?[\d]+[a-z]?)*)\s+(?:read\s+with\s+[\d]+[a-z]?\s+)?bns',
        r'bharatiya nyaya sanhita.*?section\s+([\d]+[a-z]?)',
        r'section\s+([\d]+[a-z]?)\s+(?:of\s+)?(?:the\s+)?bns\b',
    ]

    found = set()
    for pat in patterns:
        for match in re.findall(pat, text_lower):
            parts = re.split(r'[,/\s]+', match)
            for p in parts:
                p = p.strip().rstrip('.')
                if p and re.fullmatch(r'\d{1,3}[a-z]?', p):
                    found.add(p)

    return sorted(found)


# ==============================
# IPC -> BNS TRANSLATION
# ==============================
def translate_ipc_to_bns(ipc_sections: list[str], ipc_to_bns: dict) -> list[str]:
    """
    Given a list of IPC section numbers, return the equivalent BNS numbers.
    Sections not found in the mapping are returned as-is with a '?' suffix
    so the gap is visible.
    """
    translated = []
    for sec in ipc_sections:
        bns = ipc_to_bns.get(sec.lower())
        translated.append(bns if bns else f"{sec}?")
    return translated


# ==============================
# CPC SECTION EXTRACTION
# ==============================
def extract_cpc_sections(text: str, valid_cpc: list) -> list[str]:
    """
    Extract CPC sections referenced in civil case text.
    Only returns sections that exist in the validated CPC list.
    """
    text_lower = text.lower()
    valid_set  = set(str(s).lower() for s in valid_cpc)

    patterns = [
        r'order\s+(\d{1,2})\s+rule\s+\d+\s+cpc',
        r'u/s\s+([\d]+[a-z]?(?:[/,\s]+[\d]+[a-z]?)*)\s+cpc',
        r'sections?\s+([\d]+[a-z]?(?:[,/\s]+(?:and\s+)?[\d]+[a-z]?)*)\s+(?:read\s+with\s+[\d]+[a-z]?\s+)?cpc',
        r'section\s+([\d]+[a-z]?)\s+(?:of\s+)?(?:the\s+)?civil procedure',
        r'section\s+([\d]+[a-z]?)\s+cpc',
    ]

    found = set()
    for pat in patterns:
        for match in re.findall(pat, text_lower):
            parts = re.split(r'[,/\s]+', match)
            for p in parts:
                p = p.strip().rstrip('.')
                if p and re.fullmatch(r'\d{1,3}[a-z]?', p):
                    if p in valid_set or not valid_set:  # if no CPC list, accept all
                        found.add(p)

    return sorted(found)


# ==============================
# SEVERITY SCORING
# Based on IPC/BNS sections — higher punishment = higher severity.
# Scores 1-10.  Source: IPC punishment provisions.
# This should be updated with the lawyer's full mapping PDF once available.
# ==============================

# Criminal: IPC section -> severity score
IPC_SEVERITY = {
    # 10 — Death penalty / life imprisonment crimes
    "302": 10,   # Murder
    "376": 10,   # Rape
    "376a": 10,  # Rape causing death
    "396": 10,   # Dacoity with murder
    "364a": 10,  # Kidnapping for ransom
    # 9 — Near-capital offences
    "304": 9,    # Culpable homicide not amounting to murder
    "304b": 9,   # Dowry death
    "366a": 9,   # Procuration of minor girl
    "376b": 9,   # Sexual assault
    # 8 — Serious violence
    "307": 8,    # Attempt to murder
    "395": 8,    # Dacoity
    "120b": 8,   # Criminal conspiracy
    "201": 8,    # Causing disappearance of evidence of an offence
    # 6 — Medium severity
    "326": 6,    # Grievous hurt with dangerous weapons
    "392": 6,    # Robbery
    "354": 6,    # Assault on woman with intent to outrage modesty
    "498a": 6,   # Cruelty by husband/relatives
    # 5 — Moderate
    "420": 5,    # Cheating
    "406": 5,    # Criminal breach of trust
    "409": 5,    # Criminal breach of trust by public servant
    # 3 — Lower severity
    "379": 3,    # Theft
    "323": 3,    # Voluntarily causing hurt
    "427": 3,    # Mischief causing damage
    "504": 3,    # Intentional insult to provoke breach of peace
}

# BNS equivalents (key BNS sections -> severity)
# Derived from the approved IPC mapping, then corrected for sections where multiple
# IPC offences map to one BNS section. Direct values follow the supplied lawyer rubric.
BNS_SEVERITY: dict = {}
BNS_SEVERITY_OVERRIDES = {
    "64": 10, "65": 10, "66": 10, "70": 10,  # Rape / aggravated rape
    "80": 9, "103": 10, "105": 9,             # Dowry death / murder / homicide
    "109": 8, "61": 8, "310": 8,              # Attempt murder / conspiracy / dacoity
    "117": 6, "118": 6, "309": 6, "85": 6,   # Grievous hurt / robbery / cruelty
    "318": 5, "316": 5,                         # Cheating / breach of trust
    "303": 3, "115": 3,                         # Theft / simple hurt
}


def build_bns_severity(ipc_to_bns: dict) -> dict:
    """Derive BNS severity scores from IPC scores + the mapping."""
    bns_sev = {}
    for ipc_sec, score in IPC_SEVERITY.items():
        bns_sec = ipc_to_bns.get(ipc_sec.lower())
        if bns_sec:
            # Take the max in case multiple IPC sections map to same BNS
            bns_sev[bns_sec] = max(bns_sev.get(bns_sec, 0), score)
    bns_sev.update(BNS_SEVERITY_OVERRIDES)
    return bns_sev


def calculate_severity(sections: list[str], severity_map: dict) -> int:
    """
    Return max severity score across all sections found.
    Falls back to 4 (moderate) for unmapped sections.
    """
    if not sections:
        return 3   # Low default when no sections identified

    max_score = 1
    for sec in sections:
        score = severity_map.get(sec.lower(), 4)   # 4 = unknown but present
        max_score = max(max_score, score)

    return max_score


# ==============================
# IMMEDIATE THREAT FLAG
# Any case involving bail/custody/restraint = needs urgent attention
# ==============================
def check_immediate_threat(text: str) -> int:
    return _check_immediate_threat(text)


# ==============================
# SOCIETAL IMPACT SCORE (1-5)
# Per legal experts: severity IS largely societal impact.
# High-impact = affects many people or public interest.
# ==============================
def calculate_societal_impact(text: str) -> int:
    return _calculate_societal_impact(text)


# ==============================
# COMPOSITE PRIORITY SCORE (0-10)
# Weights decided in consultation with legal expert guidance:
#   - Severity of sections: 35%  (what crime was committed)
#   - Societal impact:      25%  (how many affected)
#   - Immediate threat:     30%  (someone at risk right now)
#   - Recency:              10%  (recent = more relevant precedent)
# ==============================
def compute_priority_score(
    severity: int,
    societal_impact: int,
    immediate_threat: int,
    case_age_days: float | None,
    case_type: str
) -> float:
    return _compute_priority_score(
        severity, societal_impact, immediate_threat, case_age_days, case_type
    )


# ==============================
# PRECEDENT COUNT
# ==============================
def count_precedents(text: str) -> int:
    pattern = r'\b(?:air|scr|scc)\s+\d{4}'
    return len(re.findall(pattern, text.lower()))


# ==============================
# DATE EXTRACTION
# ==============================
def extract_date(text: str) -> str | None:
    # Case IDs and docket numbers can contain date-shaped values (for example,
    # 84-00-0541). Prefer the explicit decision date in the opening case-title
    # lines before falling back to general date scanning.
    header_lines = (text or "").splitlines()[:40]
    title_date = (
        r"(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*\d{4}"
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*,?\s*\d{4}"
        r"|\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}-\d{1,2}-\d{1,2})"
    )
    for index, line in enumerate(header_lines):
        if re.search(r"\b(?:vs|versus)\b\.?", line, flags=re.IGNORECASE):
            title_context = " ".join(header_lines[index:index + 3])
            match = re.search(rf"\bon\s+({title_date})", title_context, flags=re.IGNORECASE)
            if match:
                return match.group(1)

    patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}',
        r'\d{1,2}\.\d{1,2}\.\d{4}',
        r'(?i)\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*,?\s*\d{4}',
        r'(?i)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*,?\s*\d{4}',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return None


def parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    # Normalize string: remove commas, dots, standardize spaces, strip whitespace
    clean_str = date_str.replace(",", "").replace("\n", " ").strip()
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", clean_str):
        clean_str = clean_str.replace(".", "/")
    clean_str = re.sub(r'\s+', ' ', clean_str)
    
    # Try common formats
    for fmt in [
        "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
        "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"
    ]:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            continue
            
    # As a final fallback, try to extract just the 4-digit year and map to January 1st of that year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', clean_str)
    if year_match:
        try:
            return datetime(int(year_match.group(1)), 1, 1)
        except ValueError:
            pass
def _normalize_docket_year(year_text: str) -> int | None:
    year = int(year_text)
    if len(year_text) == 4:
        return year
    # Two-digit docket years are common (for example, W.P. No. 11055/86).
    return 2000 + year if year <= 49 else 1900 + year


def extract_writ_petition_year(text: str) -> int | None:
    """Get the filing year from a writ-petition number, including /YY format."""
    text_lower = (text or "")[:20000].lower()
    pattern = (
        r"\b(?:writ\s+petition|w\.?\s*p\.?)\s*(?:no\.?|number)?\s*"
        r"[\d,]+\s*(?:/\s*|\bof\s+)(19\d{2}|20\d{2}|\d{2})\b"
    )
    match = re.search(pattern, text_lower)
    return _normalize_docket_year(match.group(1)) if match else None


def extract_filing_year(text: str) -> int | None:
    text_lower = (text or "")[:20000].lower()
    writ_year = extract_writ_petition_year(text_lower)
    if writ_year is not None:
        return writ_year
    patterns = [
        r"\b(?:slp|special leave petition|civil appeal|criminal appeal|trp|transfer petition|c\.?a\.?|crl\.?\s*a\.?)\b.{0,120}?\bnos?\.?\s*[\d][\d,-]*\s*(?:/\s*|\bof\s+)(19\d{2}|20\d{2})\b",
        r"\b(?:appeal|petition|application)\s+(?:no\.?\s*)?[\d,/-]+\s+of\s+(19\d{2}|20\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return int(match.group(1))
    return None


def extract_filing_date(text: str) -> datetime | None:
    """Prefer a labelled filing/institution date over dates in the case narrative."""
    date_pattern = (
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4}|"
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})"
    )
    match = re.search(
        rf"(?:date\s+of\s+(?:filing|institution)|filing\s+date|filed\s+on|instituted\s+on)\s*(?:is|:|dated)?\s*{date_pattern}",
        text or "",
        flags=re.IGNORECASE,
    )
    return parse_date(match.group(1)) if match else None


def calculate_pending_days(
    judgment_date: datetime | None,
    text: str,
    as_of_date: datetime | None = None,
) -> float | None:
    filing_date = extract_filing_date(text)
    if filing_date is not None:
        # For a reported disposed case, measure filing-to-disposal. For a still
        # pending filing with no judgment date, measure filing-to-as-of date.
        ref_date = judgment_date or as_of_date or datetime.now()
        pending_days = (ref_date - filing_date).days
        return float(pending_days) if pending_days >= 0 else None

    filing_year = extract_writ_petition_year(text) or extract_filing_year(text)
    if filing_year is not None:
        if judgment_date is not None:
            # A docket number gives a year, not an exact date. Use the requested
            # judgment-year minus filing-year duration in whole years.
            age_years = judgment_date.year - filing_year
            return float(max(0, age_years) * 365)
        reference_date = as_of_date or datetime.now()
        start_date = datetime(filing_year, 1, 1)
        pending_days = (reference_date - start_date).days
        return float(pending_days) if pending_days >= 0 else None

    if judgment_date is not None:
        # Older reported judgments often omit the original filing year. Use the
        # decision date as an explicit recency proxy instead of returning NULL.
        reference_date = as_of_date or datetime.now()
        return float(max(0, (reference_date - judgment_date).days))
    return None


def age_days_from_case_date(case_date: str | datetime | None, as_of_date: datetime | None = None) -> float | None:
    """Use a known decision/case date as the recency proxy when filing age is absent."""
    parsed_date = case_date if isinstance(case_date, datetime) else parse_date(case_date)
    if parsed_date is None:
        return None
    reference_date = as_of_date or datetime.now()
    return float(max(0, (reference_date - parsed_date).days))


def extract_priority_metadata(
    text: str,
    as_of_date: datetime | None = None,
    mappings: tuple[dict, list] | None = None,
) -> dict:
    """Extract the shared case features used by the upload and corpus pipelines."""
    ipc_to_bns, cpc_valid_sections = mappings or load_mappings(MAPPINGS_FILE)
    bns_severity_map = build_bns_severity(ipc_to_bns)
    case_type = extract_case_type(text)
    case_date = extract_date(text)
    parsed_date = parse_date(case_date)
    legal_regime = detect_legal_regime(parsed_date, text)

    ipc_sections: list[str] = []
    bns_sections: list[str] = []
    cpc_sections: list[str] = []
    severity = 3

    if case_type == "criminal":
        # Score explicit IPC and BNS charges independently so mixed citations or
        # a stale regime label cannot hide the highest-severity charge.
        ipc_sections = extract_ipc_sections(text)
        bns_sections = extract_bns_sections(text)
        if legal_regime == "IPC" and not bns_sections:
            equivalents = translate_ipc_to_bns(ipc_sections, ipc_to_bns)
            bns_sections = [section for section in equivalents if not section.endswith("?")]
        candidate_scores = []
        if ipc_sections:
            candidate_scores.append(calculate_severity(ipc_sections, IPC_SEVERITY))
        if bns_sections:
            candidate_scores.append(calculate_severity(bns_sections, bns_severity_map))
        severity = max(candidate_scores, default=3)
    else:
        cpc_sections = extract_cpc_sections(text, cpc_valid_sections)

    age_days = calculate_pending_days(parsed_date, text, as_of_date=as_of_date)
    return {
        "case_type": case_type,
        "legal_regime": legal_regime,
        "ipc_sections": ipc_sections,
        "bns_sections": bns_sections,
        "cpc_sections": cpc_sections,
        "severity": severity,
        "case_date": case_date,
        "case_age_days": age_days,
        "immediate_threat_flag": check_immediate_threat(text),
        "societal_impact_score": calculate_societal_impact(text),
        "num_precedents": count_precedents(text),
        "total_words": len((text or "").split()),
    }


# ==============================
# MAIN FEATURE EXTRACTION
# ==============================
def extract_all_features():
    chunk_files = list(Path(CHUNKS_DIR).glob("*_chunks.json"))
    print(f"\nProcessing {len(chunk_files)} cases...\n")

    all_features = []

    for chunk_file in tqdm(chunk_files):
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        case_id = data["case_id"]
        chunks  = data["chunks"]
        full_text = " ".join(c["text"] for c in chunks)
        metadata = extract_priority_metadata(full_text)
        case_type = metadata["case_type"]
        regime = metadata["legal_regime"]
        date_str = metadata["case_date"]
        case_age = metadata["case_age_days"]
        ipc_sections = metadata["ipc_sections"]
        bns_sections = metadata["bns_sections"]
        cpc_sections = metadata["cpc_sections"]
        severity = metadata["severity"]
        immediate_threat = metadata["immediate_threat_flag"]
        societal_impact = metadata["societal_impact_score"]
        priority_score = compute_priority_score(
            severity, societal_impact, immediate_threat, case_age, case_type
        )

        # ── Misc stats ────────────────────────────────────────────
        num_precedents = metadata["num_precedents"]
        total_words    = sum(c["word_count"] for c in chunks)

        all_features.append({
            "case_id":              case_id,
            "case_type":            case_type,
            "legal_regime":         regime,           # "IPC" or "BNS"

            # Criminal sections
            "ipc_sections":         ",".join(ipc_sections),
            "num_ipc_sections":     len(ipc_sections),
            "bns_sections":         ",".join(bns_sections),
            "num_bns_sections":     len(bns_sections),

            # Civil sections
            "cpc_sections":         ",".join(cpc_sections),
            "num_cpc_sections":     len(cpc_sections),

            # Priority signals
            "max_severity_score":       severity,         # 1-10
            "immediate_threat_flag":    immediate_threat, # 0 or 1
            "societal_impact_score":    societal_impact,  # 1-5
            "priority_score":           priority_score,   # 0-10 composite

            # Temporal
            "case_date":            date_str,
            "case_age_days":        case_age,

            # Misc
            "num_precedents":       num_precedents,
            "total_words":          total_words,
        })

    # ── Save ──────────────────────────────────────────────────────
    df = pd.DataFrame(all_features)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFeatures extracted for {len(df)} cases")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"\nCase type breakdown:\n{df['case_type'].value_counts()}")
    print(f"\nLegal regime breakdown:\n{df['legal_regime'].value_counts()}")
    print(f"\nPriority score distribution:")
    print(df["priority_score"].describe().round(2))

    return df


if __name__ == "__main__":
    df = extract_all_features()
    print("\nSample output:")
    print(df[[
        "case_id", "case_type", "legal_regime",
        "ipc_sections", "bns_sections",
        "max_severity_score", "immediate_threat_flag",
        "societal_impact_score", "priority_score"
    ]].head(10).to_string())
