"""
Synthetic Agreement-Pipeline Demonstration
Adds Gaussian noise to AI scores to demonstrate the agreement-analysis pipeline.
This is not expert validation and must not be reported as lawyer agreement.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import psycopg2

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

OUTPUT_CSV = "evaluation/simulated_agreement_demo.csv"
OUTPUT_TEX = "evaluation/simulated_agreement_demo.tex"

def sample_demo_cohort(n_cases=20):
    """Samples a demonstration cohort across all 5 priority tiers."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, case_type, priority_score, max_severity_score, immediate_threat_flag, societal_impact_score, case_summary
        FROM cases
        WHERE priority_score IS NOT NULL
        ORDER BY priority_score DESC;
    """)
    rows = cur.fetchall()
    conn.close()

    # Stratified selection: 4 critical, 4 high, 4 medium, 4 low, 4 very low
    df = pd.DataFrame(rows, columns=["case_id", "case_type", "priority_score", "severity", "threat", "impact", "summary"])
    
    critical = df[df["priority_score"] >= 8.0].head(4)
    high = df[(df["priority_score"] >= 6.0) & (df["priority_score"] < 8.0)].head(4)
    medium = df[(df["priority_score"] >= 4.0) & (df["priority_score"] < 6.0)].head(4)
    low = df[(df["priority_score"] >= 2.0) & (df["priority_score"] < 4.0)].head(4)
    very_low = df[df["priority_score"] < 2.0].head(4)

    cohort = pd.concat([critical, high, medium, low, very_low]).reset_index(drop=True)
    return cohort

def run_synthetic_agreement_demo():
    cohort = sample_demo_cohort(20)
    print(f"[SIMULATION DEMO] Selected {len(cohort)} cases; generated ratings below are synthetic, not lawyer validation.")

    # Synthetic scores demonstrate the calculation pipeline only. They are not
    # expert judgments and cannot validate the prioritizer.
    np.random.seed(42)
    ai_scores = cohort["priority_score"].values

    # Synthetic proxy 1: AI score plus small Gaussian noise.
    proxy_1 = np.clip(ai_scores + np.random.normal(0, 0.40, len(ai_scores)), 0.0, 10.0).round(1)
    # Synthetic proxy 2: AI score plus small Gaussian noise.
    proxy_2 = np.clip(ai_scores + np.random.normal(0, 0.48, len(ai_scores)), 0.0, 10.0).round(1)
    # Synthetic proxy 3: AI score plus small Gaussian noise.
    proxy_3 = np.clip(ai_scores + np.random.normal(0, 0.52, len(ai_scores)), 0.0, 10.0).round(1)

    consensus_proxy = np.mean([proxy_1, proxy_2, proxy_3], axis=0).round(2)

    # 1. Spearman Rank Correlation (rho)
    rho_1, pval_1 = stats.spearmanr(ai_scores, proxy_1)
    rho_2, pval_2 = stats.spearmanr(ai_scores, proxy_2)
    rho_3, pval_3 = stats.spearmanr(ai_scores, proxy_3)
    rho_consensus, pval_consensus = stats.spearmanr(ai_scores, consensus_proxy)

    # 2. Pearson Linear Correlation (r)
    r_consensus, pr_pval = stats.pearsonr(ai_scores, consensus_proxy)

    # 3. Exact categorical agreement (not Cohen's kappa)
    def to_category(score):
        if score >= 8.0: return "Critical"
        if score >= 6.0: return "High"
        if score >= 4.0: return "Medium"
        if score >= 2.0: return "Low"
        return "Very Low"

    ai_cats = [to_category(s) for s in ai_scores]
    proxy_cats = [to_category(s) for s in consensus_proxy]
    exact_cat_matches = sum(1 for a, b in zip(ai_cats, proxy_cats) if a == b)
    categorical_agreement_pct = (exact_cat_matches / len(ai_cats)) * 100

    results_table = pd.DataFrame({
        "Synthetic Noise Proxy (not an expert)": [
            "Proxy 1 (AI score + Gaussian noise)",
            "Proxy 2 (AI score + Gaussian noise)",
            "Proxy 3 (AI score + Gaussian noise)",
            "Mean of synthetic proxies"
        ],
        "Spearman Rank Correlation (rho)": [f"{rho_1:.3f}", f"{rho_2:.3f}", f"{rho_3:.3f}", f"{rho_consensus:.3f}"],
        "Descriptive p-value (not validation)": [f"{pval_1:.4e}", f"{pval_2:.4e}", f"{pval_3:.4e}", f"{pval_consensus:.4e}"],
        "Interpretation": ["Synthetic correlation by construction"] * 4
    })

    print("\n==========================================================================")
    print("        SYNTHETIC NOISE DEMO — NOT EXPERT VALIDATION                       ")
    print("==========================================================================")
    print(results_table.to_string(index=False))
    print(f"\nSynthetic categorical band agreement: {categorical_agreement_pct:.1f}%")
    print(f"Synthetic Pearson alignment: r = {r_consensus:.3f} (p = {pr_pval:.4e})")
    print("These values are induced by adding noise to AI scores; they are not evidence of legal agreement.")

    results_table.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(results_table.to_latex(index=False, caption="Synthetic noise demonstration of the agreement-analysis pipeline (not expert validation)", label="tab:simulated_agreement_demo"))

    print(f"\n[SIMULATION DEMO] Saved CSV -> {OUTPUT_CSV}")
    print(f"[SIMULATION DEMO] Saved LaTeX -> {OUTPUT_TEX}")
    print("[SUCCESS] Synthetic demonstration completed; real lawyer ratings are still required.\n")

if __name__ == "__main__":
    run_synthetic_agreement_demo()
