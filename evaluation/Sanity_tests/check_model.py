# save as check_model.py and run
import joblib
import pandas as pd
import sys
sys.path.append(".")

model   = joblib.load("data/prioritization/models/prioritizer.pkl")
encoder = joblib.load("data/prioritization/models/encoders.pkl")

print("=== MODEL INFO ===")
print(f"Features: {model.feature_names_in_.tolist()}")
print(f"\nFeature importances:")
for feat, imp in sorted(
    zip(model.feature_names_in_, model.feature_importances_),
    key=lambda x: -x[1]
):
    print(f"  {feat}: {imp*100:.2f}%")

# 5 sanity tests covering different case types
tests = [
    {
        "label": "Murder + bail (expect 8-10)",
        "case_type": "criminal", "num_ipc_sections": 3,
        "num_cpc_sections": 0, "num_precedents": 5,
        "total_words": 2000, "case_age_days": 500,
        "max_severity_score": 10.0, "immediate_threat_flag": 1,
        "societal_impact_score": 4
    },
    {
        "label": "Minor civil dispute (expect 1-3)",
        "case_type": "civil", "num_ipc_sections": 0,
        "num_cpc_sections": 1, "num_precedents": 2,
        "total_words": 800, "case_age_days": 200,
        "max_severity_score": 2.0, "immediate_threat_flag": 0,
        "societal_impact_score": 1
    },
    {
        "label": "PIL environmental (expect 6-8)",
        "case_type": "civil", "num_ipc_sections": 0,
        "num_cpc_sections": 2, "num_precedents": 8,
        "total_words": 3000, "case_age_days": 1000,
        "max_severity_score": 3.0, "immediate_threat_flag": 0,
        "societal_impact_score": 5
    },
    {
        "label": "Rape no bail (expect 5-7)",
        "case_type": "criminal", "num_ipc_sections": 2,
        "num_cpc_sections": 0, "num_precedents": 4,
        "total_words": 1500, "case_age_days": 300,
        "max_severity_score": 10.0, "immediate_threat_flag": 0,
        "societal_impact_score": 2
    },
    {
        "label": "Habeas corpus civil (expect 6-8)",
        "case_type": "civil", "num_ipc_sections": 0,
        "num_cpc_sections": 0, "num_precedents": 3,
        "total_words": 1200, "case_age_days": 100,
        "max_severity_score": 2.0, "immediate_threat_flag": 1,
        "societal_impact_score": 3
    },
]

print("\n=== SANITY TESTS ===")
for t in tests:
    label     = t.pop("label")
    case_type = t.pop("case_type")
    encoded   = encoder.transform([case_type])[0]
    features  = pd.DataFrame([{"case_type": encoded, **t}])
    score     = round(min(max(float(model.predict(features)[0]), 0), 10), 2)
    print(f"{label}: {score}")