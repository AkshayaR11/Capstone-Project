"""
Global Explainability & Feature Importance Analysis
Computes TreeSHAP attributions across all 863 cases in the judicial docket.
Generates publication beeswarm, importance ranking, and local waterfall plots for Section 6.3 of the paper.
"""

import os
import sys
import math
import joblib
import psycopg2
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

OUTPUT_DIR = "evaluation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "data/prioritization/models/prioritizer.pkl"
ENCODER_PATH = "data/prioritization/models/encoders.pkl"

def generate_shap_evaluation():
    print("[XAI] Loading trained XGBoost model and encoders...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print(f"[ERROR] Model files missing at {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("[XAI] Fetching case features from PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, case_type, num_ipc_sections, num_cpc_sections, num_precedents, 
               total_words, case_age_days, max_severity_score, immediate_threat_flag, 
               societal_impact_score, priority_score
        FROM cases
        WHERE priority_score IS NOT NULL;
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"[XAI] Loaded {len(rows)} cases for global SHAP attribution.")

    records = []
    for r in rows:
        case_type = (r[1] or "civil").strip().lower()
        try:
            case_type_enc = encoder.transform([case_type])[0]
        except Exception:
            case_type_enc = 1 if case_type == "criminal" else 0

        age = float(r[6]) if r[6] is not None and not math.isnan(r[6]) else 365.0
        sev = float(r[7]) if r[7] is not None and not math.isnan(r[7]) else 3.0
        threat = int(r[8]) if r[8] is not None else 0
        impact = int(r[9]) if r[9] is not None else 1

        records.append({
            "case_id": r[0],
            "Case Type": case_type_enc,
            "IPC Sections Count": int(r[2] or 0),
            "CPC Sections Count": int(r[3] or 0),
            "Precedents Cited": int(r[4] or 0),
            "Document Word Count": int(r[5] or 1000),
            "Case Pending Days": age,
            "Offence Severity Rating": sev,
            "Immediate Threat Flag": threat,
            "Societal Impact Score": impact,
            "priority_score": float(r[10])
        })

    df = pd.DataFrame(records)

    feature_cols = [
        "Case Type",
        "IPC Sections Count",
        "CPC Sections Count",
        "Precedents Cited",
        "Document Word Count",
        "Case Pending Days",
        "Offence Severity Rating",
        "Immediate Threat Flag",
        "Societal Impact Score"
    ]
    X = df[feature_cols]

    print("[XAI] Initializing TreeExplainer and calculating SHAP values across full dataset...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # 1. Global Feature Importance Bar Plot
    plt.figure(figsize=(9, 5))
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean Absolute SHAP (Impact on Priority)": mean_abs_shap
    }).sort_values(by="Mean Absolute SHAP (Impact on Priority)", ascending=True)

    plt.barh(importance_df["Feature"], importance_df["Mean Absolute SHAP (Impact on Priority)"], color="#3b82f6", edgecolor="#1e3a8a", height=0.6)
    plt.title("Global Feature Importance in Judicial Prioritization (Mean |SHAP|)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Mean |SHAP Value| (Impact on Case Priority Score)", fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[XAI] Feature Importance Plot Saved -> {bar_path}")

    # 2. Global SHAP Summary / Beeswarm Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X, show=False)
    plt.title("Global SHAP Summary (Beeswarm Distribution across 863 Cases)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    beeswarm_path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[XAI] Beeswarm Plot Saved -> {beeswarm_path}")

    # 3. Save Importance Table to CSV & LaTeX
    importance_sorted = importance_df.sort_values(by="Mean Absolute SHAP (Impact on Priority)", ascending=False)
    csv_path = "evaluation/shap_feature_importance.csv"
    tex_path = "evaluation/shap_feature_importance_table.tex"
    importance_sorted.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(importance_sorted.to_latex(index=False, caption="Global Feature Importance Derived via TreeSHAP on Trained Prioritizer", label="tab:shap_importance"))
    
    print(f"[XAI] Saved CSV -> {csv_path}")
    print(f"[XAI] Saved LaTeX -> {tex_path}")

    print("\n==========================================================================")
    print("             GLOBAL SHAP FEATURE IMPORTANCE RANKING                        ")
    print("==========================================================================")
    print(importance_sorted.to_string(index=False))
    print("[SUCCESS] Global Explainability Analysis Completed Cleanly!\n")

if __name__ == "__main__":
    generate_shap_evaluation()
