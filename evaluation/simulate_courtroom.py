"""
Courtroom Discrete-Event Simulation Framework
Evaluates AI Prioritization Scheduling vs. Traditional Listing Policies (FIFO, LIFO, Severity-Only)
Produces publication-grade metrics and plots for Section 5 & 6 of the Research Paper.
"""

import os
import sys
import math
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psycopg2

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

OUTPUT_DIR = "evaluation/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_case_dataset():
    """Load cases from PostgreSQL database or fallback to labeled CSV."""
    records = []
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT case_id, case_type, max_severity_score, priority_score, immediate_threat_flag, societal_impact_score, case_age_days
            FROM cases
            WHERE priority_score IS NOT NULL;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for r in rows:
            records.append({
                "case_id": r[0],
                "case_type": r[1] or "civil",
                "severity": float(r[2]) if r[2] is not None else 3.0,
                "priority_score": float(r[3]) if r[3] is not None else 5.0,
                "immediate_threat": int(r[4]) if r[4] is not None else 0,
                "societal_impact": int(r[5]) if r[5] is not None else 1,
                "case_age_days": float(r[6]) if r[6] is not None and not math.isnan(r[6]) else 365.0
            })
        print(f"[DATA] Successfully loaded {len(records)} cases from PostgreSQL.")
    except Exception as e:
        print(f"[DATA] Warning: DB fetch failed ({e}). Falling back to CSV...")
        csv_path = "data/prioritization/labeled_cases.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                records.append({
                    "case_id": str(r.get("case_id", f"case_{_}")),
                    "case_type": str(r.get("case_type", "civil")),
                    "severity": float(r.get("max_severity_score", 3.0)),
                    "priority_score": float(r.get("priority_score", 5.0)),
                    "immediate_threat": int(r.get("immediate_threat_flag", 0)),
                    "societal_impact": int(r.get("societal_impact_score", 1)),
                    "case_age_days": float(r.get("case_age_days", 365.0))
                })
            print(f"[DATA] Loaded {len(records)} cases from CSV.")

    return records

class CourtroomSimulation:
    def __init__(self, cases: list, num_benches=3, cases_per_bench_day=2, simulation_days=180, arrival_rate=2.0):
        """
        Discrete-Event Simulation Parameters:
        - num_benches: 3 Virtual Judges / Benches
        - cases_per_bench_day: 2 cases resolved per bench per day (total throughput: 6 cases/day)
        - simulation_days: 180 working court days
        - arrival_rate: average 2 new case filings per day (Poisson arrival)
        """
        self.raw_cases = cases
        self.num_benches = num_benches
        self.daily_capacity = num_benches * cases_per_bench_day
        self.simulation_days = simulation_days
        self.arrival_rate = arrival_rate

    def _prepare_docket(self, seed=42):
        """Prepares initial backlog + timestamped arrivals."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Initial backlog of 500 cases
        initial_pool = random.sample(self.raw_cases, min(500, len(self.raw_cases)))
        docket = []
        for idx, c in enumerate(initial_pool):
            case = dict(c)
            case["filing_day"] = 0
            case["resolved_day"] = None
            case["wait_time"] = None
            docket.append(case)

        # Incoming new filings over 180 days (Poisson arrivals)
        remaining_pool = [c for c in self.raw_cases if c not in initial_pool]
        if not remaining_pool:
            remaining_pool = self.raw_cases

        for day in range(1, self.simulation_days + 1):
            num_arrivals = np.random.poisson(self.arrival_rate)
            for _ in range(num_arrivals):
                template = random.choice(remaining_pool)
                case = dict(template)
                case["case_id"] = f"new_{day}_{_}_{case['case_id']}"
                case["filing_day"] = day
                case["resolved_day"] = None
                case["wait_time"] = None
                docket.append(case)

        return docket

    def run_policy(self, policy_name: str, seed=42):
        """
        Runs simulation under a specified scheduling policy:
        - 'FIFO': First In First Out (standard filing queue)
        - 'LIFO': Last In First Out (recency bias)
        - 'SEVERITY_ONLY': Greedy statutory severity
        - 'AI_PRIORITY': Multi-Agent Priority Scoring with dynamic aging
        """
        docket = self._prepare_docket(seed=seed)
        unresolved = list(docket)
        resolved = []

        daily_backlog_severity = []
        daily_critical_pending = []

        for current_day in range(self.simulation_days):
            # Cases present in court today
            available = [c for c in unresolved if c["filing_day"] <= current_day]

            # Track current metrics
            total_sev = sum(c["severity"] for c in available)
            critical_cnt = sum(1 for c in available if c["priority_score"] >= 8.0)
            daily_backlog_severity.append(total_sev)
            daily_critical_pending.append(critical_cnt)

            if not available:
                continue

            # Sort available queue by policy
            if policy_name == "FIFO":
                available.sort(key=lambda x: (x["filing_day"], -x["severity"]))
            elif policy_name == "LIFO":
                available.sort(key=lambda x: (-x["filing_day"], -x["severity"]))
            elif policy_name == "SEVERITY_ONLY":
                available.sort(key=lambda x: (-x["severity"], x["filing_day"]))
            elif policy_name == "AI_PRIORITY":
                # Dynamic aging boost: cases that have waited longer gain slight priority increments
                def dynamic_score(c):
                    waiting_days = current_day - c["filing_day"]
                    aging_boost = min(waiting_days / 100.0, 1.5)  # Prevents starvation
                    return c["priority_score"] + aging_boost

                available.sort(key=lambda x: -dynamic_score(x))

            # Resolve top cases up to daily court capacity
            to_resolve = available[:self.daily_capacity]
            for c in to_resolve:
                c["resolved_day"] = current_day
                c["wait_time"] = current_day - c["filing_day"]
                resolved.append(c)
                unresolved.remove(c)

        # Compute summary metrics for resolved cases
        resolved_df = pd.DataFrame(resolved)
        critical_cases = resolved_df[resolved_df["priority_score"] >= 8.0]
        civil_cases = resolved_df[resolved_df["case_type"] == "civil"]

        crit_wait_mean = critical_cases["wait_time"].mean() if len(critical_cases) > 0 else 0.0
        crit_wait_median = critical_cases["wait_time"].median() if len(critical_cases) > 0 else 0.0
        civil_wait_mean = civil_cases["wait_time"].mean() if len(civil_cases) > 0 else 0.0
        overall_wait_mean = resolved_df["wait_time"].mean() if len(resolved_df) > 0 else 0.0
        total_resolved = len(resolved)

        return {
            "policy": policy_name,
            "total_resolved": total_resolved,
            "crit_wait_mean": crit_wait_mean,
            "crit_wait_median": crit_wait_median,
            "civil_wait_mean": civil_wait_mean,
            "overall_wait_mean": overall_wait_mean,
            "daily_backlog_severity": daily_backlog_severity,
            "daily_critical_pending": daily_critical_pending,
            "resolved_df": resolved_df
        }

def run_experiment_and_plot():
    cases = load_case_dataset()
    if not cases:
        print("[ERROR] No cases found to simulate.")
        return

    sim = CourtroomSimulation(cases, num_benches=3, cases_per_bench_day=2, simulation_days=180, arrival_rate=2.0)
    
    policies = ["FIFO", "LIFO", "SEVERITY_ONLY", "AI_PRIORITY"]
    results = {}

    print("\n--- Running Courtroom Scheduling Simulations (180 Days) ---")
    for p in policies:
        print(f"Simulating policy: {p}...")
        results[p] = sim.run_policy(p)

    # 1. Summary Comparison Table
    table_rows = []
    for p in policies:
        res = results[p]
        table_rows.append({
            "Policy": p,
            "Cases Resolved": res["total_resolved"],
            "Critical Turnaround (Mean Days)": round(res["crit_wait_mean"], 1),
            "Critical Turnaround (Median Days)": round(res["crit_wait_median"], 1),
            "Civil Wait Time (Mean Days)": round(res["civil_wait_mean"], 1),
            "Overall Wait (Mean Days)": round(res["overall_wait_mean"], 1)
        })

    summary_df = pd.DataFrame(table_rows)
    print("\n==========================================================================")
    print("           COURT DOCKET SCHEDULING SIMULATION RESULTS (180 DAYS)           ")
    print("==========================================================================")
    print(summary_df.to_string(index=False))

    # Save CSV & LaTeX table
    csv_out = "evaluation/simulation_results.csv"
    tex_out = "evaluation/simulation_results_table.tex"
    summary_df.to_csv(csv_out, index=False)
    with open(tex_out, "w") as f:
        f.write(summary_df.to_latex(index=False, caption="Comparative Courtroom Scheduling Performance across 180 Simulated Court Days", label="tab:simulation_results"))
    print(f"\nSaved CSV -> {csv_out}")
    print(f"Saved LaTeX -> {tex_out}")

    # 2. Plot 1: Cumulative Backlog Severity Decay Curve
    plt.figure(figsize=(9, 5.5))
    days = list(range(len(results["FIFO"]["daily_backlog_severity"])))
    
    colors = {"FIFO": "#64748b", "LIFO": "#f59e0b", "SEVERITY_ONLY": "#8b5cf6", "AI_PRIORITY": "#10b981"}
    styles = {"FIFO": "--", "LIFO": ":", "SEVERITY_ONLY": "-.", "AI_PRIORITY": "-"}
    widths = {"FIFO": 1.8, "LIFO": 1.8, "SEVERITY_ONLY": 2.0, "AI_PRIORITY": 2.8}

    for p in policies:
        plt.plot(days, results[p]["daily_backlog_severity"], label=p, color=colors[p], linestyle=styles[p], linewidth=widths[p])

    plt.title("Cumulative Docket Backlog Severity Decay Over 180 Court Days", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Simulated Court Working Days", fontsize=11)
    plt.ylabel("Total Outstanding Backlog Severity Index", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    decay_path = os.path.join(OUTPUT_DIR, "simulation_backlog_severity_decay.png")
    plt.savefig(decay_path, dpi=300)
    plt.close()
    print(f"Plot Saved -> {decay_path}")

    # 3. Plot 2: Critical Case Turnaround Time Comparison (Bar Chart)
    plt.figure(figsize=(7.5, 4.8))
    policy_labels = ["FIFO (Status Quo)", "LIFO (Recency Bias)", "Severity Greedy", "AI Priority (Ours)"]
    crit_means = [results[p]["crit_wait_mean"] for p in policies]
    bar_colors = ["#94a3b8", "#fcd34d", "#c4b5fd", "#34d399"]

    bars = plt.bar(policy_labels, crit_means, color=bar_colors, edgecolor="#1e293b", linewidth=1.2, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}d", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Mean Turnaround Time to Resolve 'Critical' Cases (Days)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Average Wait Time (Court Days)", fontsize=11)
    plt.ylim(0, max(crit_means) * 1.22)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    turnaround_path = os.path.join(OUTPUT_DIR, "simulation_critical_turnaround.png")
    plt.savefig(turnaround_path, dpi=300)
    plt.close()
    print(f"Plot Saved -> {turnaround_path}")

    # 4. Plot 3: Daily Critical Cases Pending in Queue
    plt.figure(figsize=(9, 5.0))
    for p in policies:
        plt.plot(days, results[p]["daily_critical_pending"], label=p, color=colors[p], linestyle=styles[p], linewidth=widths[p])

    plt.title("Unresolved 'Critical' Urgent Cases Remaining in Docket", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Simulated Court Working Days", fontsize=11)
    plt.ylabel("Number of Critical Cases Pending", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    crit_pending_path = os.path.join(OUTPUT_DIR, "simulation_critical_pending.png")
    plt.savefig(crit_pending_path, dpi=300)
    plt.close()
    print(f"Plot Saved -> {crit_pending_path}")
    print("\n[SUCCESS] Courtroom Simulation Experiment Completed Successfully!\n")

if __name__ == "__main__":
    run_experiment_and_plot()
