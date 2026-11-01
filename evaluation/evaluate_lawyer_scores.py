"""Compare the rule score and simple policies with a small lawyer-rated pilot.

Input columns are the five values supplied in evaluation/lawyer_score_pilot.csv.
This is a descriptive pilot, not an independent external validation study: it
contains 28 cases and one supplied final score per case.
"""

from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.prioritization.scoring import compute_priority_score  # noqa: E402

INPUT = ROOT / "evaluation" / "lawyer_score_pilot.csv"
EXISTING_MACHINE = ROOT / "data" / "lawyer_test_results.csv"
OUTPUT_CASES = ROOT / "evaluation" / "lawyer_score_pilot_case_results.csv"
OUTPUT_METRICS = ROOT / "evaluation" / "lawyer_score_pilot_metrics.csv"


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return normalized.removesuffix("pdf")


def tier(score: float) -> str:
    if score >= 8:
        return "Critical"
    if score >= 6:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"


def top_k_names(frame: pd.DataFrame, score_col: str, k: int = 5) -> set[str]:
    # Alphabetical tie-break prevents input ordering (which follows lawyer
    # scores in the supplied file) from inflating baseline top-k overlap.
    return set(
        frame.sort_values([score_col, "Case name"], ascending=[False, True])
        .head(k)["Case name"]
    )


def main() -> None:
    df = pd.read_csv(INPUT)
    old = pd.read_csv(EXISTING_MACHINE)
    df["_key"] = df["Case name"].map(normalize_name)
    if df["_key"].duplicated().any():
        raise ValueError("Duplicate case names in lawyer pilot input")
    old["_key"] = old["File Name"].map(normalize_name)
    matches = []
    for key in df["_key"]:
        candidates = old[old["_key"].str.startswith(key)]
        if candidates.empty:
            raise ValueError(f"No stored result for {key}")
        exact = candidates[candidates["_key"] == key]
        if len(exact) == 1:
            matches.append(exact.iloc[0])
        else:
            candidates = candidates.assign(_extra=candidates["_key"].str.len() - len(key))
            best = candidates[candidates["_extra"] == candidates["_extra"].min()]
            if len(best) != 1:
                raise ValueError(f"Ambiguous stored result for {key}: {best['File Name'].tolist()}")
            matches.append(best.iloc[0])
    stored = pd.DataFrame(matches).reset_index(drop=True)
    merged = pd.concat([df.reset_index(drop=True), stored[["Case Type", "Severity", "Threat", "Impact", "Priority Score"]]], axis=1)

    # Use stored case type and lawyer-supplied numeric inputs. Omit text-only
    # adjustments because they cannot be reconstructed from this score sheet.
    merged["Rule score (lawyer inputs)"] = merged.apply(
        lambda r: compute_priority_score(
            severity=r["Lawyer severity"],
            societal_impact=r["Lawyer impact"],
            immediate_threat=r["Lawyer threat"],
            case_age_days=r["Age years"] * 365,
            case_type=str(r["Case Type"]).lower(),
        ),
        axis=1,
    )

    # Baselines use only the same supplied inputs and fixed, transparent scales.
    merged["Severity-only"] = merged["Lawyer severity"]
    merged["Threat-first"] = merged["Lawyer threat"]
    merged["Age-only"] = (merged["Age years"] / merged["Age years"].max() * 10).round(2)
    merged["FIFO age proxy"] = merged["Age years"]
    merged["Lawyer rank"] = merged["Lawyer final priority"].rank(ascending=False, method="average")

    score_cols = [
        "Rule score (lawyer inputs)",
        "Severity-only",
        "Threat-first",
        "Age-only",
        "FIFO age proxy",
    ]
    merged["Rule score (lawyer inputs) rank"] = merged["Rule score (lawyer inputs)"].rank(
        ascending=False, method="average"
    )
    merged["Lawyer tier"] = merged["Lawyer final priority"].map(tier)
    merged["Rule tier"] = merged["Rule score (lawyer inputs)"].map(tier)
    merged["Absolute score error"] = (
        merged["Rule score (lawyer inputs)"] - merged["Lawyer final priority"]
    ).abs()
    merged["Rank difference (rule minus lawyer)"] = (
        merged["Rule score (lawyer inputs) rank"] - merged["Lawyer rank"]
    )
    merged["Existing stored score"] = merged["Priority Score"]
    merged["Stored input match"] = (
        (merged["Severity"] == merged["Lawyer severity"])
        & (merged["Threat"] == merged["Lawyer threat"])
        & (merged["Impact"] == merged["Lawyer impact"])
    )

    rows = []
    y = merged["Lawyer final priority"]
    lawyer_top5 = top_k_names(merged, "Lawyer final priority")
    for col in score_cols:
        rho, _ = spearmanr(merged[col], y)
        tau, _ = kendalltau(merged[col], y)
        rho_bootstrap = []
        rng = np.random.default_rng(20261101)
        for _ in range(500):
            sample_idx = rng.integers(0, len(merged), len(merged))
            sampled = merged.iloc[sample_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boot_rho = spearmanr(sampled[col], sampled["Lawyer final priority"]).statistic
            if np.isfinite(boot_rho):
                rho_bootstrap.append(float(boot_rho))
        row = {
            "Policy": col,
            "N": len(merged),
            "Spearman rho": rho,
            "Spearman rho bootstrap 95% low": np.quantile(rho_bootstrap, 0.025),
            "Spearman rho bootstrap 95% high": np.quantile(rho_bootstrap, 0.975),
            "Kendall tau": tau,
            "Top-5 overlap": len(top_k_names(merged, col) & lawyer_top5),
        }
        if col == "Rule score (lawyer inputs)":
            row["MAE on 0-10 scale"] = mean_absolute_error(y, merged[col])
            row["RMSE on 0-10 scale"] = mean_squared_error(y, merged[col]) ** 0.5
        rows.append(row)
    matched = merged[merged["Stored input match"]].copy()
    stored_rho, _ = spearmanr(matched["Existing stored score"], matched["Lawyer final priority"])
    stored_tau, _ = kendalltau(matched["Existing stored score"], matched["Lawyer final priority"])
    rows.append({
        "Policy": "Previously stored system score (exact feature-match subset)",
        "N": len(matched),
        "Spearman rho": stored_rho,
        "Kendall tau": stored_tau,
        "Top-5 overlap": len(
            top_k_names(matched, "Existing stored score")
            & top_k_names(matched, "Lawyer final priority")
        ),
        "MAE on 0-10 scale": mean_absolute_error(matched["Lawyer final priority"], matched["Existing stored score"]),
        "RMSE on 0-10 scale": mean_squared_error(matched["Lawyer final priority"], matched["Existing stored score"]) ** 0.5,
    })
    metrics = pd.DataFrame(rows).sort_values("Spearman rho", ascending=False)

    columns = [
        "Case name", "Lawyer severity", "Lawyer threat", "Lawyer impact", "Age years",
        "Lawyer final priority", "Rule score (lawyer inputs)", "Rule tier", "Lawyer tier",
        "Absolute score error", "Lawyer rank", "Rule score (lawyer inputs) rank",
        "Rank difference (rule minus lawyer)", "Existing stored score", "Case Type", "Severity",
        "Threat", "Impact", "Stored input match",
    ]
    merged[columns].to_csv(OUTPUT_CASES, index=False)
    metrics.to_csv(OUTPUT_METRICS, index=False)

    print(f"Cases: {len(merged)}")
    print(f"Stored feature rows matching lawyer inputs: {merged['Stored input match'].sum()}/{len(merged)}")
    print(f"Rule exact-tier agreement: {(merged['Rule tier'] == merged['Lawyer tier']).mean() * 100:.1f}%")
    print(f"Rule score MAE: {merged['Absolute score error'].mean():.3f}")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Case results: {OUTPUT_CASES.relative_to(ROOT)}")
    print(f"Policy metrics: {OUTPUT_METRICS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
