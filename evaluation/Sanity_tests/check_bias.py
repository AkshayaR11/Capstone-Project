# save as check_bias.py and run
import sys
sys.path.append(".")
from agents.fairness.bias_agent import BiasAgent

agent = BiasAgent()

biased = """The accused, belonging to a lower caste background, 
was profiled by the police. His religious background was noted 
as a factor. The suspect community to which he belongs has been 
previously flagged."""

neutral = """The appellant challenged the conviction under Section 
302 IPC. The court examined the evidence presented by both 
prosecution and defense. The forensic report was admitted as 
exhibit A. The witnesses were cross-examined."""

r1 = agent.audit_case_fairness(biased)
r2 = agent.audit_case_fairness(neutral)

print("=== BIASED TEXT ===")
print(f"Score:   {r1['bias_score']} (should be < 0.85)")
print(f"Flags:   {r1['flags']}")
print(f"Details: {r1['bias_details']}")

print("\n=== NEUTRAL TEXT ===")
print(f"Score:   {r2['bias_score']} (should be >= 0.85 or None)")
print(f"Flags:   {r2['flags']}")
print(f"Details: {r2['bias_details']}")

print("\n=== DIRECTION CHECK ===")
b = r1['bias_score'] if r1['bias_score'] is not None else 1.0
n = r2['bias_score'] if r2['bias_score'] is not None else 1.0
print(f"Biased scores lower than neutral? {b < n}")