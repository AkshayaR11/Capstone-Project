"""Illustrative paired-seed courtroom scheduling simulation.

Urgency labels, arrival rates, and service targets below are explicit simulation
assumptions, not lawyer judgments, legal ground truth, or observed future filings.
Arrivals are bootstrap copies of existing case profiles.
"""

import os
import sys
import random
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psycopg2

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIMULATION_DAYS = 180
INITIAL_BACKLOG = 500
NUM_BENCHES = 3
CASES_PER_BENCH_DAY = 2
DAILY_CAPACITY = NUM_BENCHES * CASES_PER_BENCH_DAY
REPLICATIONS = 30
SEED_BASE = 20261029
WORKLOADS = {
    "BELOW_CAPACITY": 2.0,
    "NEAR_CAPACITY": 6.0,
    "ABOVE_CAPACITY": 8.0,
}
POLICIES = ["FIFO", "SEVERITY_ONLY", "AGING_ONLY", "AI_PRIORITY"]

# Illustrative service targets chosen for scenario analysis, not official standards.
URGENT_WAIT_TARGET_DAYS = 5
OLDER_CASE_AGE_DAYS = 3650
OLDER_WAIT_TARGET_DAYS = 30


def _number(value, default):
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default

def load_case_dataset():
    """Load cases from PostgreSQL database or fallback to labeled CSV."""
    records = []
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            SELECT case_id, case_type, max_severity_score, priority_score, immediate_threat_flag, societal_impact_score, case_age_days
            FROM cases
            WHERE priority_score IS NOT NULL;
        """)
        rows = cur.fetchall()
        for r in rows:
            records.append({
                "case_id": r[0],
                "case_type": r[1] or "civil",
                "severity": _number(r[2], 3.0),
                "priority_score": _number(r[3], 5.0),
                "immediate_threat": int(_number(r[4], 0)),
                "societal_impact": int(_number(r[5], 1)),
                "case_age_days": _number(r[6], 365.0)
            })
        print(f"[DATA] Successfully loaded {len(records)} cases from PostgreSQL.")
    except Exception as e:
        print(f"[DATA] Warning: DB fetch failed ({e}). Falling back to CSV...")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    if not records:
        csv_path = PROJECT_ROOT / "data" / "prioritization" / "labeled_cases.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                records.append({
                    "case_id": str(r.get("case_id", f"case_{_}")),
                    "case_type": str(r.get("case_type", "civil")),
                    "severity": _number(r.get("max_severity_score"), 3.0),
                    "priority_score": _number(r.get("priority_score"), 5.0),
                    "immediate_threat": int(_number(r.get("immediate_threat_flag"), 0)),
                    "societal_impact": int(_number(r.get("societal_impact_score"), 1)),
                    "case_age_days": _number(r.get("case_age_days"), 365.0)
                })
            print(f"[DATA] Loaded {len(records)} cases from CSV.")

    return records

class CourtroomSimulation:
    def __init__(self, cases, num_benches=NUM_BENCHES,
                 cases_per_bench_day=CASES_PER_BENCH_DAY,
                 simulation_days=SIMULATION_DAYS, arrival_rate=2.0,
                 initial_backlog=INITIAL_BACKLOG):
        self.raw_cases = cases
        self.num_benches = num_benches
        self.daily_capacity = num_benches * cases_per_bench_day
        self.simulation_days = simulation_days
        self.arrival_rate = arrival_rate
        self.initial_backlog = initial_backlog

    def _prepare_docket(self, seed=42):
        """Create the same bootstrap docket for every policy at this seed."""
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(seed)
        initial_pool = py_rng.sample(self.raw_cases, min(self.initial_backlog, len(self.raw_cases)))
        initial_ids = {c["case_id"] for c in initial_pool}
        docket = []
        for c in initial_pool:
            case = dict(c)
            case["filing_day"] = 0
            case["resolved_day"] = None
            case["wait_time"] = None
            case["is_bootstrap_arrival"] = False
            case["source_case_id"] = c["case_id"]
            case["scenario_urgent"] = int(case["immediate_threat"] == 1 or case["severity"] >= 9)
            docket.append(case)

        # Synthetic filings copy feature profiles from records not in the initial backlog.
        remaining_pool = [c for c in self.raw_cases if c["case_id"] not in initial_ids]
        if not remaining_pool:
            remaining_pool = self.raw_cases

        for day in range(1, self.simulation_days + 1):
            num_arrivals = int(np_rng.poisson(self.arrival_rate))
            for _ in range(num_arrivals):
                template = py_rng.choice(remaining_pool)
                case = dict(template)
                case["source_case_id"] = template["case_id"]
                case["case_id"] = f"synthetic_{seed}_{day}_{_}_{template['case_id']}"
                case["filing_day"] = day
                case["resolved_day"] = None
                case["wait_time"] = None
                case["is_bootstrap_arrival"] = True
                case["case_age_days"] = 0.0
                case["scenario_urgent"] = int(case["immediate_threat"] == 1 or case["severity"] >= 9)
                docket.append(case)

        return docket

    def run_policy(self, policy_name: str, seed=42):
        """
        Run one policy against the deterministic docket generated by `seed`.
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
            critical_cnt = sum(1 for c in available if c["scenario_urgent"])
            daily_backlog_severity.append(total_sev)
            daily_critical_pending.append(critical_cnt)

            if not available:
                continue

            # Sort available queue by policy
            if policy_name == "FIFO":
                available.sort(key=lambda x: x["filing_day"])
            elif policy_name == "SEVERITY_ONLY":
                available.sort(key=lambda x: (-x["severity"], x["filing_day"]))
            elif policy_name == "AGING_ONLY":
                available.sort(
                    key=lambda x: (-(x["case_age_days"] + current_day - x["filing_day"]), x["filing_day"])
                )
            elif policy_name == "AI_PRIORITY":
                def dynamic_score(c):
                    waiting_days = current_day - c["filing_day"]
                    age_days = max(0.0, c["case_age_days"] + waiting_days)
                    severity_norm = c["severity"] / 10.0
                    if c["case_type"].lower() == "civil":
                        severity_norm = min(severity_norm, 0.4)
                    impact_norm = c["societal_impact"] / 5.0
                    recency = max(0.0, 1.0 - age_days / 3650.0)
                    composite_score = (
                        severity_norm * 3.5
                        + float(c["immediate_threat"]) * 3.0
                        + impact_norm * 2.5
                        + recency
                    )
                    aging_boost = min(waiting_days / 100.0, 1.5)
                    return composite_score + aging_boost

                available.sort(key=lambda x: -dynamic_score(x))
            else:
                raise ValueError(f"Unknown scheduling policy: {policy_name}")

            # Resolve top cases up to daily court capacity
            to_resolve = available[:self.daily_capacity]
            for c in to_resolve:
                c["resolved_day"] = current_day
                c["wait_time"] = current_day - c["filing_day"]
                resolved.append(c)
                unresolved.remove(c)

        # Compute metrics over resolved and still-pending docket items.
        resolved_df = pd.DataFrame(resolved)
        unresolved_df = pd.DataFrame(unresolved)
        if not unresolved_df.empty:
            unresolved_df["wait_at_horizon"] = [
                max(0, self.simulation_days - c["filing_day"]) for c in unresolved
            ]

        def waits(frame, column="wait_time"):
            if frame.empty:
                return np.array([], dtype=float)
            return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)

        def mask_for(frame, predicate):
            return frame.apply(predicate, axis=1) if not frame.empty else pd.Series([], dtype=bool)

        urgent_resolved = resolved_df[mask_for(resolved_df, lambda c: bool(c["scenario_urgent"]))]
        urgent_unresolved = unresolved_df[mask_for(unresolved_df, lambda c: bool(c["scenario_urgent"]))]
        civil_cases = resolved_df[resolved_df["case_type"].str.lower() == "civil"] if not resolved_df.empty else resolved_df
        criminal_cases = resolved_df[resolved_df["case_type"].str.lower() == "criminal"] if not resolved_df.empty else resolved_df
        older_resolved = resolved_df[resolved_df["case_age_days"] >= OLDER_CASE_AGE_DAYS] if not resolved_df.empty else resolved_df
        older_unresolved = unresolved_df[unresolved_df["case_age_days"] >= OLDER_CASE_AGE_DAYS] if not unresolved_df.empty else unresolved_df

        urgent_waits = waits(urgent_resolved)
        overall_waits = waits(resolved_df)
        civil_waits = waits(civil_cases)
        criminal_waits = waits(criminal_cases)
        older_waits = waits(older_resolved)
        urgent_unresolved_waits = waits(urgent_unresolved, "wait_at_horizon")
        older_unresolved_waits = waits(older_unresolved, "wait_at_horizon")

        urgent_misses = int(np.sum(urgent_waits > URGENT_WAIT_TARGET_DAYS) + np.sum(urgent_unresolved_waits > URGENT_WAIT_TARGET_DAYS))
        older_misses = int(np.sum(older_waits > OLDER_WAIT_TARGET_DAYS) + np.sum(older_unresolved_waits > OLDER_WAIT_TARGET_DAYS))

        def mean_or_nan(values):
            return float(np.mean(values)) if len(values) else np.nan

        def quantile_or_nan(values, q):
            return float(np.quantile(values, q)) if len(values) else np.nan

        urgent_total = len(urgent_resolved) + len(urgent_unresolved)
        older_total = len(older_resolved) + len(older_unresolved)

        return {
            "policy": policy_name,
            "total_cases": len(docket),
            "total_resolved": len(resolved),
            "total_unresolved": len(unresolved),
            "urgent_total": urgent_total,
            "urgent_resolved": len(urgent_resolved),
            "urgent_unresolved": len(urgent_unresolved),
            "urgent_wait_mean": mean_or_nan(urgent_waits),
            "urgent_wait_median": quantile_or_nan(urgent_waits, .50),
            "urgent_wait_p90": quantile_or_nan(urgent_waits, .90),
            "urgent_wait_p95": quantile_or_nan(urgent_waits, .95),
            "urgent_target_miss_count": urgent_misses,
            "urgent_target_miss_pct": 100 * urgent_misses / urgent_total if urgent_total else np.nan,
            "overall_wait_mean": mean_or_nan(overall_waits),
            "civil_wait_mean": mean_or_nan(civil_waits),
            "criminal_wait_mean": mean_or_nan(criminal_waits),
            "older_total": older_total,
            "older_resolved": len(older_resolved),
            "older_unresolved": len(older_unresolved),
            "older_wait_mean": mean_or_nan(older_waits),
            "older_wait_p90": quantile_or_nan(older_waits, .90),
            "older_target_miss_count": older_misses,
            "older_target_miss_pct": 100 * older_misses / older_total if older_total else np.nan,
            "daily_backlog_severity": daily_backlog_severity,
            "daily_urgent_pending": daily_critical_pending,
            "resolved_df": resolved_df,
            "unresolved_df": unresolved_df,
        }

METRICS = [
    "total_cases", "total_resolved", "total_unresolved",
    "urgent_total", "urgent_resolved", "urgent_unresolved",
    "urgent_wait_mean", "urgent_wait_median", "urgent_wait_p90", "urgent_wait_p95",
    "urgent_target_miss_count", "urgent_target_miss_pct",
    "overall_wait_mean", "civil_wait_mean", "criminal_wait_mean",
    "older_total", "older_resolved", "older_unresolved", "older_wait_mean",
    "older_wait_p90", "older_target_miss_count", "older_target_miss_pct",
]


def mean_ci95(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(len(vals))
    return mean, mean - half_width, mean + half_width


def run_experiment_and_plot():
    cases = load_case_dataset()
    if not cases:
        print("[ERROR] No cases found to simulate.")
        return

    raw_rows = []
    daily_history = {}
    print(
        f"\n--- Paired-seed simulation: {SIMULATION_DAYS} days, "
        f"{REPLICATIONS} replications, capacity {DAILY_CAPACITY}/day ---"
    )
    print("[ASSUMPTION] Urgent = immediate_threat_flag == 1 OR severity >= 9; this is not legal ground truth.")
    print("[ASSUMPTION] Poisson arrivals bootstrap existing profiles; synthetic age resets to 0 and composite priority is recomputed.")

    for workload, arrival_rate in WORKLOADS.items():
        sim = CourtroomSimulation(cases, arrival_rate=arrival_rate)
        for replication in range(REPLICATIONS):
            seed = SEED_BASE + replication
            for policy in POLICIES:
                result = sim.run_policy(policy, seed=seed)
                row = {
                    "workload": workload,
                    "arrival_rate_per_day": arrival_rate,
                    "capacity_per_day": DAILY_CAPACITY,
                    "policy": policy,
                    "seed": seed,
                    "replication": replication + 1,
                    "arrival_model": "Poisson bootstrap profiles; arrival age reset to 0 and composite priority recomputed",
                    "urgency_assumption": "immediate_threat_flag == 1 OR severity >= 9",
                    "urgent_wait_target_days": URGENT_WAIT_TARGET_DAYS,
                    "older_case_age_threshold_days": OLDER_CASE_AGE_DAYS,
                    "older_wait_target_days": OLDER_WAIT_TARGET_DAYS,
                }
                row.update({metric: result[metric] for metric in METRICS})
                raw_rows.append(row)
                daily_history.setdefault((workload, policy), {}).setdefault("backlog", []).append(
                    result["daily_backlog_severity"]
                )
                daily_history[(workload, policy)].setdefault("urgent_pending", []).append(
                    result["daily_urgent_pending"]
                )
        print(f"[DONE] {workload}: lambda={arrival_rate:g}, policies={len(POLICIES)}, paired seeds={REPLICATIONS}")

    raw_df = pd.DataFrame(raw_rows)
    raw_path = PROJECT_ROOT / "evaluation" / "simulation_replications.csv"
    raw_df.to_csv(raw_path, index=False)

    summary_rows = []
    for (workload, arrival_rate, policy), group in raw_df.groupby(
        ["workload", "arrival_rate_per_day", "policy"], sort=False
    ):
        row = {
            "workload": workload,
            "arrival_rate_per_day": arrival_rate,
            "capacity_per_day": DAILY_CAPACITY,
            "policy": policy,
            "replications": len(group),
        }
        for metric in METRICS:
            mean, low, high = mean_ci95(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # Paired differences use the exact same seed/docket within each workload.
    paired_rows = []
    paired_metrics = [
        "urgent_wait_mean", "urgent_wait_p90", "urgent_unresolved",
        "urgent_target_miss_count", "overall_wait_mean", "civil_wait_mean",
        "criminal_wait_mean", "older_wait_p90", "older_target_miss_count",
    ]
    for workload in WORKLOADS:
        cohort = raw_df[raw_df["workload"] == workload]
        fifo = cohort[cohort["policy"] == "FIFO"].set_index("seed")
        for policy in POLICIES:
            if policy == "FIFO":
                continue
            candidate = cohort[cohort["policy"] == policy].set_index("seed")
            row = {"workload": workload, "policy_vs_fifo": policy, "paired_replications": len(fifo)}
            for metric in paired_metrics:
                delta = candidate[metric] - fifo[metric]
                mean, low, high = mean_ci95(delta)
                row[f"{metric}_difference_mean"] = mean
                row[f"{metric}_difference_ci95_low"] = low
                row[f"{metric}_difference_ci95_high"] = high
            paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)
    paired_path = PROJECT_ROOT / "evaluation" / "simulation_paired_vs_fifo.csv"
    paired_df.to_csv(paired_path, index=False)

    # Compact table: group means with approximate 95% CIs, with trade-offs visible.
    def ci_text(row, metric, digits=1):
        mean = row[f"{metric}_mean"]
        low = row[f"{metric}_ci95_low"]
        high = row[f"{metric}_ci95_high"]
        return "NA" if not np.isfinite(mean) else f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"

    table_rows = []
    for _, row in summary_df.iterrows():
        table_rows.append({
            "Workload (arrivals/day)": f"{row['workload']} ({row['arrival_rate_per_day']:g})",
            "Policy": row["policy"],
            "Resolved (mean)": ci_text(row, "total_resolved"),
            "Urgent wait p90, days [95% CI]": ci_text(row, "urgent_wait_p90"),
            "Urgent unresolved (mean)": ci_text(row, "urgent_unresolved"),
            "Urgent target misses (mean)": ci_text(row, "urgent_target_miss_count"),
            "Overall wait, days [95% CI]": ci_text(row, "overall_wait_mean"),
            "Civil wait, days [95% CI]": ci_text(row, "civil_wait_mean"),
            "Criminal wait, days [95% CI]": ci_text(row, "criminal_wait_mean"),
            "Older-case p90, days [95% CI]": ci_text(row, "older_wait_p90"),
            "Older target misses (mean)": ci_text(row, "older_target_miss_count"),
        })
    table_df = pd.DataFrame(table_rows)

    summary_path = PROJECT_ROOT / "evaluation" / "simulation_results.csv"
    tex_path = PROJECT_ROOT / "evaluation" / "simulation_results_table.tex"
    summary_df.to_csv(summary_path, index=False)
    table_df.to_latex(
        tex_path,
        index=False,
        caption=("Illustrative paired-seed courtroom simulation. Urgency and service targets are scenario assumptions; "
                 "arrivals are bootstrap samples of existing case profiles. Cells report means and approximate 95\\% CIs."),
        label="tab:simulation_results",
        escape=True,
    )

    print("\n=== Mean outcomes by workload and policy (approx. 95% CI) ===")
    print(table_df.to_string(index=False))
    print(f"\nSaved replication rows: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved paired differences vs FIFO: {paired_path}")
    print(f"Saved LaTeX table: {tex_path}")

    # Plot urgent tail waits by workload with between-seed confidence intervals.
    colors = {"FIFO": "#64748b", "SEVERITY_ONLY": "#8b5cf6", "AGING_ONLY": "#f59e0b", "AI_PRIORITY": "#10b981"}
    x = np.arange(len(WORKLOADS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, policy in enumerate(POLICIES):
        rows = summary_df[summary_df["policy"] == policy].set_index("workload").loc[list(WORKLOADS)]
        means = rows["urgent_wait_p90_mean"].to_numpy(dtype=float)
        lows = rows["urgent_wait_p90_ci95_low"].to_numpy(dtype=float)
        highs = rows["urgent_wait_p90_ci95_high"].to_numpy(dtype=float)
        errors = np.vstack([np.maximum(0, means - lows), np.maximum(0, highs - means)])
        ax.bar(x + (idx - 1.5) * width, means, width, yerr=errors, capsize=3,
               label=policy, color=colors[policy], edgecolor="#1e293b", linewidth=.5)
    ax.set_xticks(x, [f"{name.replace('_', ' ').title()}\nλ={rate:g}/day" for name, rate in WORKLOADS.items()])
    ax.set_ylabel("Urgent-case 90th percentile wait (days)")
    ax.set_title("Urgent tail wait across workload scenarios (approx. 95% CI)")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation_critical_turnaround.png", dpi=250)
    plt.close(fig)

    # Pending urgent load, including cases not resolved by the horizon.
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, policy in enumerate(POLICIES):
        rows = summary_df[summary_df["policy"] == policy].set_index("workload").loc[list(WORKLOADS)]
        means = rows["urgent_unresolved_mean"].to_numpy(dtype=float)
        lows = rows["urgent_unresolved_ci95_low"].to_numpy(dtype=float)
        highs = rows["urgent_unresolved_ci95_high"].to_numpy(dtype=float)
        errors = np.vstack([np.maximum(0, means - lows), np.maximum(0, highs - means)])
        ax.bar(x + (idx - 1.5) * width, means, width, yerr=errors, capsize=3,
               label=policy, color=colors[policy], edgecolor="#1e293b", linewidth=.5)
    ax.set_xticks(x, [f"{name.replace('_', ' ').title()}\nλ={rate:g}/day" for name, rate in WORKLOADS.items()])
    ax.set_ylabel("Urgent cases unresolved at horizon (mean)")
    ax.set_title("Unresolved urgent cases across workload scenarios (approx. 95% CI)")
    ax.grid(axis="y", linestyle="--", alpha=.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation_critical_pending.png", dpi=250)
    plt.close(fig)

    # Daily severity backlog bands for the near-capacity scenario.
    workload = "NEAR_CAPACITY"
    fig, ax = plt.subplots(figsize=(10, 5.5))
    days = np.arange(SIMULATION_DAYS)
    for policy in POLICIES:
        traces = np.asarray(daily_history[(workload, policy)]["backlog"], dtype=float)
        mean = traces.mean(axis=0)
        half = 1.96 * traces.std(axis=0, ddof=1) / np.sqrt(traces.shape[0])
        ax.plot(days, mean, label=policy, color=colors[policy], linewidth=2)
        ax.fill_between(days, mean - half, mean + half, color=colors[policy], alpha=.12)
    ax.set_title("Outstanding severity index: near-capacity workload")
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Unresolved docket severity index")
    ax.grid(True, linestyle="--", alpha=.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "simulation_backlog_severity_decay.png", dpi=250)
    plt.close(fig)

    print(f"\nPlots saved under {OUTPUT_DIR}")
    print("[DONE] Simulation outputs are illustrative and use synthetic bootstrap arrivals/urgency assumptions.\n")

if __name__ == "__main__":
    run_experiment_and_plot()
