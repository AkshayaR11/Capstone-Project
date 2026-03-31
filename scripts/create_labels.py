"""
Priority assignment using CLEAN features (aligned with your pipeline)
"""

import pandas as pd
from pathlib import Path

INPUT_FILE = "data/prioritization/case_features.csv"
OUTPUT_FILE = "data/prioritization/labeled_cases.csv"

Path("data/prioritization").mkdir(exist_ok=True)


# ==============================
# PRIORITY SCORING
# ==============================
def assign_priority_score(row):
    score = 0.0

    # ======================================
    # 1. CASE TYPE (BASE PRIORITY)
    # ======================================
    if row['case_type'] == 'criminal':
        score += 6.0
    else:
        score += 4.0  # civil

    # ======================================
    # 2. SEVERITY (IPC / CPC)
    # ======================================
    if row['case_type'] == 'criminal':
        # more IPC sections → more serious
        if row['num_ipc_sections'] >= 3:
            score += 2.0
        elif row['num_ipc_sections'] >= 1:
            score += 1.0

    else:  # civil
        if row['num_cpc_sections'] >= 3:
            score += 1.5
        elif row['num_cpc_sections'] >= 1:
            score += 1.0

    # ======================================
    # 3. CASE AGE (VERY IMPORTANT)
    # ======================================
    if pd.notna(row['case_age_days']):
        age = row['case_age_days']

        if age > 7000:        # ~20 years
            score += 3.0
        elif age > 3650:      # 10 years
            score += 2.5
        elif age > 1825:      # 5 years
            score += 2.0
        elif age > 730:       # 2 years
            score += 1.5
        elif age > 365:
            score += 1.0
        else:
            score += 0.5

    # ======================================
    # 4. LEGAL COMPLEXITY
    # ======================================
    if row['num_precedents'] > 10:
        score += 1.0
    elif row['num_precedents'] > 5:
        score += 0.5

    # ======================================
    # NORMALIZE
    # ======================================
    score = min(score, 10.0)
    return round(score, 2)


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
    print("📊 Loading features...")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ Loaded {len(df)} cases")

    print("\n🎯 Assigning scores...")
    df['priority_score'] = df.apply(assign_priority_score, axis=1)

    print("\n📊 Assigning categories...")
    df['priority_category'] = df['priority_score'].apply(assign_category)

    print("\n🧠 Generating explanations...")
    df['priority_explanation'] = df.apply(generate_explanation, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved → {OUTPUT_FILE}")

    print("\n📊 Distribution:")
    print(df['priority_category'].value_counts())

    return df


if __name__ == "__main__":
    df = create_priority_labels()