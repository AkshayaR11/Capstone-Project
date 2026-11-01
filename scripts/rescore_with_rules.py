"""
Re-score all cases in the database using the new legally-validated
agentic prioritization approach (Phase 2).
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import psycopg2
import json
from pathlib import Path
from datetime import datetime
from agents.prioritization.prioritizer import PrioritizationAgent
from scripts.extract_features import (
    age_days_from_case_date,
    extract_priority_metadata,
    load_mappings,
    MAPPINGS_FILE,
)
from config import DATABASE_URL

DB_URL = DATABASE_URL

def rescore_all():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute("""
        SELECT case_id, case_type, legal_regime, ipc_sections, bns_sections, cpc_sections,
               case_date, case_age_days, max_severity_score, immediate_threat_flag,
               societal_impact_score, num_ipc_sections, num_bns_sections, num_cpc_sections,
               num_precedents, total_words
        FROM cases
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} cases to re-score.")

    agent = PrioritizationAgent()
    updated = 0

    chunks_dir = Path("data/chunks")
    mappings = load_mappings(MAPPINGS_FILE)
    for row in rows:
        (
            case_id, case_type, regime, ipc_sections, bns_sections, cpc_sections,
            case_date, age_days, severity, threat, impact, num_ipc, num_bns,
            num_cpc, num_precedents, total_words,
        ) = row

        chunk_path = chunks_dir / f"{case_id}_chunks.json"
        if chunk_path.is_file():
            try:
                with chunk_path.open("r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                full_text = " ".join(c.get("text", "") for c in chunk_data.get("chunks", []))
                metadata = extract_priority_metadata(full_text, mappings=mappings)
                case_type = metadata["case_type"]
                regime = metadata["legal_regime"]
                ipc_sections = ",".join(metadata["ipc_sections"])
                bns_sections = ",".join(metadata["bns_sections"])
                cpc_sections = ",".join(metadata["cpc_sections"])
                case_date = metadata["case_date"]
                age_days = metadata["case_age_days"]
                severity = metadata["severity"]
                threat = metadata["immediate_threat_flag"]
                impact = metadata["societal_impact_score"]
                num_ipc = len(metadata["ipc_sections"])
                num_bns = len(metadata["bns_sections"])
                num_cpc = len(metadata["cpc_sections"])
                num_precedents = metadata["num_precedents"]
                total_words = metadata["total_words"]
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Could not re-extract {case_id} from {chunk_path}: {exc}; using saved features.")

        if age_days is None:
            age_days = age_days_from_case_date(case_date, as_of_date=datetime.now())

        score = agent.compute_priority_score(
            severity=severity if severity is not None else 3.0,
            societal_impact=impact if impact is not None else 1,
            immediate_threat=threat if threat is not None else 0,
            case_age_days=age_days,
            case_type=case_type or "civil",
        )
        cur.execute("""
            UPDATE cases
            SET case_type = %s, legal_regime = %s, ipc_sections = %s, bns_sections = %s,
                cpc_sections = %s, case_date = %s, case_age_days = %s,
                max_severity_score = %s, immediate_threat_flag = %s,
                societal_impact_score = %s, num_ipc_sections = %s, num_bns_sections = %s,
                num_cpc_sections = %s, num_precedents = %s, total_words = %s,
                priority_score = %s
            WHERE case_id = %s
        """, (
            case_type or "civil", regime, ipc_sections, bns_sections, cpc_sections,
            case_date, age_days, severity if severity is not None else 3.0,
            threat if threat is not None else 0, impact if impact is not None else 1,
            num_ipc if num_ipc is not None else 0, num_bns if num_bns is not None else 0,
            num_cpc if num_cpc is not None else 0,
            num_precedents if num_precedents is not None else 0,
            total_words if total_words is not None else 0, score, case_id,
        ))
        updated += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"SUCCESS: Re-scored {updated} cases with the new legal formula.")

if __name__ == "__main__":
    rescore_all()
