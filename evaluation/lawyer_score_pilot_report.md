# Lawyer-Score Pilot: Results and Benchmark Plan

## Dataset and comparison

The input contains 28 case names with lawyer-supplied severity, threat/liberty, impact, age, and final priority values. The evaluator applies the current deterministic scoring formula to those supplied inputs, using the stored case type and omitting text-only adjustments that cannot be reconstructed from this score sheet. It compares rankings against threat-only, severity-only, age-only, and an age-based FIFO proxy. The supplied sheet has no filing dates, so age is only a proxy for FIFO. Ties are broken alphabetically to avoid inheriting the input order, which is sorted by lawyer priority.

The existing machine-output CSV contains matching numeric severity/threat/impact inputs for only 15 of the 28 cases. Its comparison is therefore shown separately on that exact-match subset.

## Results

| Policy | Cases | Spearman ρ (bootstrap 95% interval) | Kendall τ | Top-5 overlap | MAE / RMSE |
|---|---:|---:|---:|---:|---:|
| Current rule applied to lawyer inputs | 28 | 0.908 (0.728–0.982) | 0.811 | 5/5 | 0.202 / 0.243 |
| Threat-first | 28 | 0.664 (0.447–0.812) | 0.555 | 5/5 | Not comparable on the score scale |
| Severity-only | 28 | 0.565 (0.228–0.773) | 0.474 | 3/5 | Not comparable on the score scale |
| Age-only | 28 | 0.562 (0.224–0.787) | 0.422 | 1/5 | Not comparable on the score scale |
| FIFO proxy (age only) | 28 | 0.562 (0.224–0.787) | 0.422 | 1/5 | Not comparable on the score scale |
| Previously stored system score, exact-feature subset | 15 | 0.963 | 0.903 | 5/5 | 0.188 / 0.204 |

The rule result also has 96.4% exact agreement on the existing score bands (Critical ≥8, High ≥6, Medium ≥4, Low <4). The stored feature row matches the lawyer's severity/threat/impact values for 15/28 cases. Bootstrap intervals use 500 resamples with a fixed seed; with 28 cases they are descriptive and may be unstable.

## Interpretation

The current formula's score ordering is closely aligned with this set of lawyer final scores, and it ranks the same five cases at the top. The benchmark is small and was built from one supplied score per case. If the lawyers independently assigned the final scores after reviewing the cases, this is useful pilot evidence that the policy is consistent with those ratings. If they calculated the final score from the same formula/weights, the agreement is expected and is not independent validation. Do not describe these results as inter-rater agreement unless multiple people independently rated the same cases and their individual ratings are retained.

The 15-case stored-score subset also has strong agreement, but its small size and selection from the 28-case pilot limit generalization. Thirteen cases have at least one feature mismatch between stored machine inputs and lawyer inputs, so stored scores for those cases are not an apples-to-apples comparison.

## Recommended benchmark extensions

1. **Prioritization baselines:** retain the current rule, threat-first, severity-only, age-only, and FIFO using actual filing dates. Add a simple, separately specified category/fast-track policy only when its rules can be documented. Report rank correlation, top-k overlap, score MAE, priority-band agreement, and case-level rank errors.
2. **More cases and split:** expand the case set and freeze a held-out evaluation set before tuning weights. Do not tune the formula and report performance on the same 28 cases.
3. **Retrieval baseline:** use the Indian Legal Prior Case Retrieval task from IL-TUR for an external Indian-law comparison where task/data fit, and retain BM25, Legal-BERT, hybrid, and reciprocal-rank fusion on the project's own corpus. The project-specific cross-regime IPC/BNS query set still needs its own relevance labels; report Recall@k, MRR, and nDCG@k only after those labels exist.
4. **Encoder baseline:** compare `nlpaueb/legal-bert-base-uncased` against InLegalBERT and other Indian legal encoders on the exact same fixed query/corpus split.
5. **Simulation baselines:** continue using FIFO plus threat-first and severity-only, and report urgent wait gains together with non-urgent waits and unresolved backlog. Treat assumptions and bootstrap arrivals as scenario parameters.

Useful external resources: [IL-TUR (Indian Legal Text Understanding and Reasoning)](https://aclanthology.org/2024.acl-long.618/) includes a Legal Prior Case Retrieval task; [InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) is pretrained on Indian Supreme Court and High Court case documents. These provide retrieval/model comparison options; neither is a gold standard for this project's case-priority policy.
