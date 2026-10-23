"""
Courtroom Scheduling Simulation & Evaluation Module
Comparing Traditional Baselines (FIFO, LIFO, Random) against AI-Driven Prioritization.
Outputs metrics and publication-ready graphs for research evaluation.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

def run_simulation(
    df: pd.DataFrame,
    num_judges: int = 3,
    cases_per_judge_day: int = 2,
    output_dir: str = "Capstone-Project/evaluation/simulation_results"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean and prepare dataset
    cases = df.copy()
    if 'priority_score' not in cases.columns:
        raise ValueError("Dataset missing 'priority_score' column")
    
    cases['case_date'] = pd.to_datetime(cases['case_date'], errors='coerce')
    cases = cases.dropna(subset=['priority_score']).reset_index(drop=True)
    total_cases = len(cases)
    
    # Capacity: total cases processed per simulated day
    daily_capacity = num_judges * cases_per_judge_day
    
    # Define Scheduling Strategies
    strategies = {
        "FIFO (Traditional - First In First Out)": cases.sort_values(by="case_date", ascending=True).reset_index(drop=True),
        "LIFO (Recency Skew - Last In First Out)": cases.sort_values(by="case_date", ascending=False).reset_index(drop=True),
        "Random Scheduling (Unprioritized Docket)": cases.sample(frac=1, random_state=42).reset_index(drop=True),
        "AI Agent Prioritization (Proposed)": cases.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    }
    
    simulation_results = {}
    decay_trajectories = {}
    
    print(f"Running simulation with {total_cases} cases | {num_judges} Judges ({daily_capacity} cases/day)...")
    
    for name, queue in strategies.items():
        # Assign simulated resolution day
        queue = queue.copy()
        queue['resolution_day'] = (queue.index // daily_capacity) + 1
        
        # Categorize case criticality
        # Critical / High Severity: priority_score >= 7.0 or max_severity_score >= 7.0
        critical_mask = (queue['priority_score'] >= 7.0) | (queue.get('max_severity_score', 0) >= 7.0)
        moderate_mask = (~critical_mask) & (queue['priority_score'] >= 4.0)
        low_mask = queue['priority_score'] < 4.0
        
        avg_wait_critical = queue[critical_mask]['resolution_day'].mean() if critical_mask.any() else 0
        median_wait_critical = queue[critical_mask]['resolution_day'].median() if critical_mask.any() else 0
        p90_wait_critical = queue[critical_mask]['resolution_day'].quantile(0.90) if critical_mask.any() else 0
        
        avg_wait_overall = queue['resolution_day'].mean()
        
        # Calculate Backlog Severity Decay over time
        total_days = int(queue['resolution_day'].max())
        remaining_severity = []
        total_cumulative_severity = queue['priority_score'].sum()
        
        running_backlog = total_cumulative_severity
        daily_res = queue.groupby('resolution_day')['priority_score'].sum().to_dict()
        
        for d in range(1, total_days + 1):
            running_backlog -= daily_res.get(d, 0)
            remaining_severity.append(max(0.0, running_backlog))
            
        decay_trajectories[name] = remaining_severity
        
        simulation_results[name] = {
            "Total Cases": int(total_cases),
            "Critical Cases Count": int(critical_mask.sum()),
            "Avg Resolution Days (Critical)": round(float(avg_wait_critical), 2),
            "Median Resolution Days (Critical)": round(float(median_wait_critical), 2),
            "90th Percentile Resolution Days (Critical)": round(float(p90_wait_critical), 2),
            "Avg Resolution Days (Overall)": round(float(avg_wait_overall), 2),
            "Total Days to Complete Docket": int(total_days)
        }
    
    # Compute relative turnaround speedup for critical cases
    ai_critical_avg = simulation_results["AI Agent Prioritization (Proposed)"]["Avg Resolution Days (Critical)"]
    fifo_critical_avg = simulation_results["FIFO (Traditional - First In First Out)"]["Avg Resolution Days (Critical)"]
    speedup_pct = round(((fifo_critical_avg - ai_critical_avg) / fifo_critical_avg) * 100, 2)
    
    metrics_summary = {
        "parameters": {
            "num_judges": num_judges,
            "cases_per_judge_day": cases_per_judge_day,
            "daily_capacity": daily_capacity,
            "total_cases_simulated": total_cases
        },
        "critical_turnaround_improvement_percent": speedup_pct,
        "strategy_comparison": simulation_results
    }
    
    # Save JSON summary
    summary_path = os.path.join(output_dir, "simulation_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Simulation metrics written to {summary_path}")
    
    # ---------------- PLOTTING ----------------
    # Plot 1: Backlog Severity Decay Curve
    plt.figure(figsize=(10, 6))
    palette = {
        "FIFO (Traditional - First In First Out)": "#4a5568",
        "LIFO (Recency Skew - Last In First Out)": "#e53e3e",
        "Random Scheduling (Unprioritized Docket)": "#dd6b20",
        "AI Agent Prioritization (Proposed)": "#3182ce"
    }
    
    for name, traj in decay_trajectories.items():
        plt.plot(range(1, len(traj) + 1), traj, label=name, color=palette.get(name, "black"),
                 linewidth=2.5 if "AI" in name else 1.8,
                 linestyle="-" if "AI" in name else "--")
                 
    plt.title("Cumulative Threat & Severity Backlog Decay Over Simulation Days", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Simulated Court Working Days", fontsize=12)
    plt.ylabel("Remaining Docket Cumulative Severity Index", fontsize=12)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e0")
    plt.tight_layout()
    decay_chart_path = os.path.join(output_dir, "backlog_decay_curve.png")
    plt.savefig(decay_chart_path, dpi=300)
    plt.close()
    
    # Plot 2: Resolution Time for High-Severity / Critical Cases
    plt.figure(figsize=(9, 5))
    names = list(simulation_results.keys())
    short_names = ["FIFO", "LIFO", "Random", "AI Prioritization\n(Proposed)"]
    crit_days = [simulation_results[k]["Avg Resolution Days (Critical)"] for k in names]
    colors = ["#718096", "#e53e3e", "#dd6b20", "#2b6cb0"]
    
    bars = plt.bar(short_names, crit_days, color=colors, width=0.55, edgecolor="#2d3748")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(crit_days)*0.015), f"{yval:.1f} days", ha='center', va='bottom', fontweight='bold', fontsize=11)
        
    plt.title("Average Resolution Time for Critical / High-Severity Cases", fontsize=13, fontweight="bold", pad=12)
    plt.ylabel("Average Days until Listing/Resolution", fontsize=11)
    plt.ylim(0, max(crit_days) * 1.15)
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, "critical_case_resolution_time.png")
    plt.savefig(bar_chart_path, dpi=300)
    plt.close()
    
    print(f"Generated evaluation plots in {output_dir}")
    return metrics_summary

if __name__ == "__main__":
    features_csv = "Capstone-Project/data/prioritization/labeled_cases.csv"
    if not os.path.exists(features_csv):
        features_csv = "Capstone-Project/data/prioritization/case_features.csv"
    
    df = pd.read_csv(features_csv)
    results = run_simulation(df)
    print("\n--- SIMULATION SUMMARY ---")
    print(json.dumps(results, indent=2))
