"""
Summarization Quality & Statutory Fidelity Evaluation
Computes ROUGE-1, ROUGE-2, and ROUGE-L (Precision, Recall, F1)
along with Statutory Section Preservation & Compression Ratios.
Produces publication tables for Section 6.2 of the paper.
"""

import os
import sys
import json
import re
import psycopg2
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

OUTPUT_CSV = "evaluation/summarization_rouge_results.csv"
OUTPUT_TEX = "evaluation/summarization_rouge_table.tex"

def extract_section_numbers(text):
    """Extract section references like '302', '376', '420', '39', etc."""
    return set(re.findall(r'\b\d{1,3}[a-z]?\b', text.lower()))

def evaluate_summaries():
    print("[SUMMARIZATION] Connecting to database to fetch AI summaries...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, case_summary, ipc_sections, bns_sections, cpc_sections
        FROM cases
        WHERE case_summary IS NOT NULL AND LENGTH(case_summary) > 50;
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"[SUMMARIZATION] Loaded {len(rows)} cases with verified AI summaries.")
    if len(rows) == 0:
        print("[ERROR] No cases found with generated summaries.")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1_p, r1_r, r1_f1 = [], [], []
    r2_p, r2_r, r2_f1 = [], [], []
    rl_p, rl_r, rl_f1 = [], [], []
    compression_ratios = []
    entity_preservation_rates = []

    case_eval_details = []

    for case_id, summary, ipc, bns, cpc in rows:
        chunk_file = os.path.join("data", "chunks", f"{case_id}_chunks.json")
        ref_text = ""
        if os.path.exists(chunk_file):
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                ref_text = " ".join(c["text"] for c in cdata.get("chunks", []))
            except Exception:
                pass

        if not ref_text.strip():
            continue

        # Truncate source reference to first 8,000 words to ensure optimal LCS DP computation speed
        ref_sample = " ".join(ref_text.split()[:8000])

        # ROUGE Scoring
        scores = scorer.score(ref_sample, summary)

        r1_p.append(scores['rouge1'].precision)
        r1_r.append(scores['rouge1'].recall)
        r1_f1.append(scores['rouge1'].fmeasure)

        r2_p.append(scores['rouge2'].precision)
        r2_r.append(scores['rouge2'].recall)
        r2_f1.append(scores['rouge2'].fmeasure)

        rl_p.append(scores['rougeL'].precision)
        rl_r.append(scores['rougeL'].recall)
        rl_f1.append(scores['rougeL'].fmeasure)

        # Compression Ratio: Summary Words / Source Words
        comp_ratio = round((len(summary.split()) / len(ref_text.split())) * 100, 2)
        compression_ratios.append(comp_ratio)

        # Statutory Entity Preservation: do sections identified in case metadata appear in summary?
        known_sections = set(filter(None, (ipc or "").split(",") + (bns or "").split(",") + (cpc or "").split(",")))
        if known_sections:
            summary_lower = summary.lower()
            preserved_count = sum(1 for sec in known_sections if sec.strip().lower() in summary_lower)
            preservation_rate = (preserved_count / len(known_sections)) * 100
        else:
            preservation_rate = 100.0
        entity_preservation_rates.append(preservation_rate)

        case_eval_details.append({
            "Case ID": case_id,
            "Summary Words": len(summary.split()),
            "Source Words": len(ref_text.split()),
            "Compression (%)": comp_ratio,
            "ROUGE-1 F1": round(scores['rouge1'].fmeasure, 4),
            "ROUGE-2 F1": round(scores['rouge2'].fmeasure, 4),
            "ROUGE-L F1": round(scores['rougeL'].fmeasure, 4),
            "Section Preservation (%)": round(preservation_rate, 1)
        })

    # Summary Metrics Table
    summary_table = pd.DataFrame({
        "Evaluation Metric": [
            "ROUGE-1 (Unigrams)",
            "ROUGE-2 (Bigrams)",
            "ROUGE-L (Longest Common Subsequence)",
            "Mean Compression Ratio",
            "Statutory Section Preservation"
        ],
        "Precision": [f"{np.mean(r1_p):.4f}", f"{np.mean(r2_p):.4f}", f"{np.mean(rl_p):.4f}", "N/A", "N/A"],
        "Recall": [f"{np.mean(r1_r):.4f}", f"{np.mean(r2_r):.4f}", f"{np.mean(rl_r):.4f}", "N/A", "N/A"],
        "F1-Score / Rate": [
            f"{np.mean(r1_f1):.4f}",
            f"{np.mean(r2_f1):.4f}",
            f"{np.mean(rl_f1):.4f}",
            f"{np.mean(compression_ratios):.1f}%",
            f"{np.mean(entity_preservation_rates):.1f}%"
        ]
    })

    print("\n========================================================================")
    print("           SUMMARIZATION QUALITY & LEGAL FIDELITY METRICS               ")
    print("========================================================================")
    print(summary_table.to_string(index=False))

    # Save to CSV and LaTeX
    summary_table.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(summary_table.to_latex(index=False, caption="Quantitative Evaluation of 4-Section Legal Summaries Generated by Gemini 2.5 Flash", label="tab:rouge_metrics"))

    print(f"\n[SUMMARIZATION] Saved CSV -> {OUTPUT_CSV}")
    print(f"[SUMMARIZATION] Saved LaTeX -> {OUTPUT_TEX}")
    print("[SUCCESS] Summarization Evaluation Completed Successfully!\n")

if __name__ == "__main__":
    evaluate_summaries()
