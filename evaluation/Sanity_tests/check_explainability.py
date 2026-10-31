# save as check_explainability.py and run
import sys
sys.path.append(".")
from agents.explainability.explainability_agent import ExplainabilityAgent

agent = ExplainabilityAgent()
print(f"Baseline: {agent.baseline}")

# Check nan in baseline
import math
nan_keys = [k for k, v in agent.baseline.items() 
            if isinstance(v, float) and math.isnan(v)]
print(f"NaN values in baseline: {nan_keys if nan_keys else 'None — good'}")

critical = {
    "case_type": "criminal", "num_ipc_sections": 3,
    "num_cpc_sections": 0, "num_precedents": 5,
    "total_words": 2000, "case_age_days": 500,
    "max_severity_score": 10.0, "immediate_threat_flag": 1,
    "societal_impact_score": 4
}

minor = {
    "case_type": "civil", "num_ipc_sections": 0,
    "num_cpc_sections": 1, "num_precedents": 1,
    "total_words": 600, "case_age_days": 180,
    "max_severity_score": 2.0, "immediate_threat_flag": 0,
    "societal_impact_score": 1
}

r1 = agent.compute_marginal_attributions(critical)
r2 = agent.compute_marginal_attributions(minor)

print("\n=== CRITICAL CASE ===")
for k, v in sorted(r1.items(), key=lambda x: -abs(x[1])):
    print(f"  {k}: {v:+.2f}")

print("\n=== MINOR CIVIL CASE ===")
for k, v in sorted(r2.items(), key=lambda x: -abs(x[1])):
    print(f"  {k}: {v:+.2f}")

print("\n=== DIRECTION CHECKS ===")
print(f"Critical — immediate_threat positive? {r1.get('immediate_threat_flag',0) > 0}")
print(f"Critical — severity positive?         {r1.get('max_severity_score',0) > 0}")
print(f"Critical scores higher than minor?    {sum(r1.values()) > sum(r2.values())}")