import psycopg2
from collections import Counter
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DATABASE_URL
from agents.reconciliation.reconciliation_agent import audit_dossier_node
from agents.explainability.explainability_agent import ExplainabilityAgent

ex = ExplainabilityAgent()
cur = psycopg2.connect(DATABASE_URL).cursor()
cur.execute("""SELECT case_type, num_ipc_sections, num_cpc_sections, num_precedents, total_words,
  case_age_days, max_severity_score, immediate_threat_flag, societal_impact_score,
  priority_score, bias_score, bias_flags, case_summary FROM cases""")
cols = ["case_type","num_ipc_sections","num_cpc_sections","num_precedents","total_words",
        "case_age_days","max_severity_score","immediate_threat_flag","societal_impact_score"]
tally, n = Counter(), 0
for r in cur.fetchall():
    feats = {k: v for k, v in zip(cols, r[:9]) if v is not None}   # None -> explainer defaults
    out = audit_dossier_node({"priority_score": float(r[9]), "bias_score": r[10],
        "bias_flags": [f for f in (r[11] or "").split(",") if f],
        "case_summary": r[12] or "", "contributions": ex.compute_marginal_attributions(feats)})
    n += 1
    tally["ANY"] += out["conflict_detected"]
    for reason in out["conflict_reasons"]:
        tally[reason.split(":")[0]] += 1
print(n, dict(tally))