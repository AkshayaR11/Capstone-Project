import psycopg2
import joblib
import pandas as pd
import math
import os
import sys

# Ensure parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATABASE_URL

MODEL_PATH = "data/prioritization/models/prioritizer.pkl"
ENCODER_PATH = "data/prioritization/models/encoders.pkl"

def update_database_with_ml():
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Load model and encoder
    print("Loading trained XGBoost model and encoders...")
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        print(f"Error loading model files: {e}. Please train the model first by running train_prioritizer.py.")
        return

    # Fetch cases from database
    print("Fetching case features from PostgreSQL...")
    cur.execute("""
        SELECT case_id, case_type, num_ipc_sections, num_cpc_sections, num_precedents, total_words, case_age_days, max_severity_score, immediate_threat_flag, societal_impact_score
        FROM cases
    """)
    records = cur.fetchall()
    print(f"Found {len(records)} cases to process.")

    updated_count = 0
    for r in records:
        case_id = r[0]
        case_type = r[1] if r[1] else "civil"
        num_ipc = r[2] if r[2] is not None else 0
        num_cpc = r[3] if r[3] is not None else 0
        num_prec = r[4] if r[4] is not None else 0
        tot_words = r[5] if r[5] is not None else 0
        age_days = r[6] if r[6] is not None else 365.0
        max_sev = r[7] if r[7] is not None else 3.0
        threat = r[8] if r[8] is not None else 0
        impact = r[9] if r[9] is not None else 1
        
        # Handle nan values
        if math.isnan(age_days) or math.isinf(age_days):
            age_days = 365.0
        if math.isnan(max_sev) or math.isinf(max_sev):
            max_sev = 3.0
        if math.isnan(threat) or math.isinf(threat):
            threat = 0
        if math.isnan(impact) or math.isinf(impact):
            impact = 1

        # Encode case type
        try:
            case_type_encoded = encoder.transform([case_type])[0]
        except Exception:
            case_type_encoded = 1 if case_type == "criminal" else 0

        # Prepare features
        feature_dict = {
            "case_type": [case_type_encoded],
            "num_ipc_sections": [num_ipc],
            "num_cpc_sections": [num_cpc],
            "num_precedents": [num_prec],
            "total_words": [tot_words],
            "case_age_days": [age_days],
            "max_severity_score": [max_sev],
            "immediate_threat_flag": [threat],
            "societal_impact_score": [impact]
        }
        df_features = pd.DataFrame(feature_dict)
        
        # Predict priority score
        pred_score = float(model.predict(df_features)[0])
        pred_score = round(min(max(pred_score, 0.0), 10.0), 2)

        # Update database
        cur.execute("""
            UPDATE cases
            SET priority_score = %s
            WHERE case_id = %s
        """, (pred_score, case_id))
        updated_count += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"SUCCESS: Successfully updated {updated_count} cases in database with XGBoost predicted priority scores!")

if __name__ == "__main__":
    update_database_with_ml()
