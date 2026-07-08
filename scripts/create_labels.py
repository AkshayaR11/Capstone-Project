"""
Priority assignment using CLEAN features (aligned with your pipeline)
"""

import sys
import os
import pandas as pd
import math
from pathlib import Path

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.extract_features import compute_priority_score

INPUT_FILE = "data/prioritization/case_features.csv"
OUTPUT_FILE = "data/prioritization/labeled_cases.csv"

Path("data/prioritization").mkdir(exist_ok=True)


# ==============================
# PRIORITY SCORING
# ==============================
def assign_priority_score(row):
    severity = row.get('max_severity_score', 3)
    societal_impact = row.get('societal_impact_score', 1)
    immediate_threat = row.get('immediate_threat_flag', 0)
    case_age_days = row.get('case_age_days')
    case_type = row.get('case_type', 'civil')

    # Handle NaNs or None
    if pd.isna(severity): severity = 3
    if pd.isna(societal_impact): societal_impact = 1
    if pd.isna(immediate_threat): immediate_threat = 0
    if pd.isna(case_age_days) or math.isnan(case_age_days):
        case_age_days = None
    else:
        case_age_days = float(case_age_days)

    return compute_priority_score(
        severity=int(severity),
        societal_impact=int(societal_impact),
        immediate_threat=int(immediate_threat),
        case_age_days=case_age_days,
        case_type=case_type
    )


# ==============================
# CATEGORY
# ==============================
def assign_category(score):
    if score >= 8:
        return "Critical"
    elif score >= 6:
        return "High"
    elif score >= 4:
        return "Medium"
    elif score >= 2:
        return "Low"
    else:
        return "Very Low"


# ==============================
# EXPLANATION
# ==============================
def generate_explanation(row):
    reasons = []

    if row['case_type'] == 'criminal':
        reasons.append("Criminal case")

        if row['num_ipc_sections'] > 0:
            reasons.append(f"{row['num_ipc_sections']} IPC sections involved")

    else:
        reasons.append("Civil case")

        if row['num_cpc_sections'] > 0:
            reasons.append(f"{row['num_cpc_sections']} CPC sections involved")

    if pd.notna(row['case_age_days']):
        years = int(row['case_age_days'] / 365)
        if years > 5:
            reasons.append(f"Pending for {years} years")

    if row['num_precedents'] > 5:
        reasons.append("High legal complexity")

    if not reasons:
        reasons.append("Standard priority")

    return "; ".join(reasons)


# ==============================
# MAIN
# ==============================
def create_priority_labels():
    print("Loading features...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} cases")

    print("\nAssigning scores...")
    df['priority_score'] = df.apply(assign_priority_score, axis=1)

    print("\nAssigning categories...")
    df['priority_category'] = df['priority_score'].apply(assign_category)

    print("\nGenerating explanations...")
    df['priority_explanation'] = df.apply(generate_explanation, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved -> {OUTPUT_FILE}")

    print("\nDistribution:")
    print(df['priority_category'].value_counts())

    return df


if __name__ == "__main__":
    df = create_priority_labels()