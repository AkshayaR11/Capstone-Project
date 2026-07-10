"""
Train a fresh XGBoost Prioritization Model on the existing 460 cases.
Labels are generated using Dr. Saritha P's legally-validated rule formula.
This replaces Akshaya's old model trained on manually-labelled CSV data.

Run from project root:
    python scripts/train_prioritizer_v2.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import psycopg2
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from agents.prioritization.prioritizer import PrioritizationAgent

DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:password@localhost:5432/judicial")
MODEL_OUT   = "data/prioritization/models/prioritizer.pkl"
ENCODER_OUT = "data/prioritization/models/encoders.pkl"
PLOT_OUT    = "data/prioritization/models/plot_v2.png"

# ------------------------------------------------------------------ #
#  1. PULL CASE FEATURES FROM POSTGRES
# ------------------------------------------------------------------ #
def fetch_cases():
    print("Connecting to PostgreSQL and fetching cases...")
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT
            case_id,
            case_type,
            bns_sections,
            ipc_sections,
            num_ipc_sections,
            num_cpc_sections,
            num_precedents,
            total_words,
            case_age_days,
            max_severity_score,
            immediate_threat_flag,
            societal_impact_score
        FROM cases
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    print(f"Fetched {len(rows)} cases.")
    return rows

# ------------------------------------------------------------------ #
#  2. GENERATE LABELS USING DR. SARITHA'S RULE FORMULA
# ------------------------------------------------------------------ #
def generate_labels(rows, agent):
    print("Generating rule-based priority labels for all cases...")
    records = []
    for row in rows:
        (case_id, case_type, bns_sections, ipc_sections,
         num_ipc, num_cpc, num_prec, tot_words,
         age_days, max_sev, threat, impact) = row

        # Sanitize nulls
        case_type   = (case_type or "civil").lower()
        bns_sections = bns_sections or ""
        ipc_sections = ipc_sections or ""
        num_ipc     = int(num_ipc or 0)
        num_cpc     = int(num_cpc or 0)
        num_prec    = int(num_prec or 0)
        tot_words   = int(tot_words or 0)
        max_sev     = float(max_sev) if max_sev and not math.isnan(float(max_sev)) else 3.0
        threat      = int(threat or 0)
        impact      = int(impact or 1)

        if age_days is None or (isinstance(age_days, float) and (math.isnan(age_days) or math.isinf(age_days))):
            age_days = 365.0
        else:
            age_days = float(age_days)

        # Generate label using our validated rule engine
        label = agent.compute_priority_score(
            case_type=case_type,
            bns_sections=bns_sections,
            ipc_sections=ipc_sections,
            case_age_days=age_days,
            text="",  # no raw text in DB, features already extracted
        )

        records.append({
            "case_id":               case_id,
            "case_type":             case_type,
            "num_ipc_sections":      num_ipc,
            "num_cpc_sections":      num_cpc,
            "num_precedents":        num_prec,
            "total_words":           tot_words,
            "case_age_days":         age_days,
            "max_severity_score":    max_sev,
            "immediate_threat_flag": threat,
            "societal_impact_score": impact,
            "priority_score":        label,
        })

    df = pd.DataFrame(records)
    print(f"Label distribution:\n{df['priority_score'].describe()}")
    return df

# ------------------------------------------------------------------ #
#  3. TRAIN XGBOOST
# ------------------------------------------------------------------ #
def train(df):
    print("\nEncoding features and training XGBoost...")

    enc = LabelEncoder()
    df["case_type_enc"] = enc.fit_transform(df["case_type"])

    FEATURES = [
        "case_type_enc", "num_ipc_sections", "num_cpc_sections",
        "num_precedents", "total_words", "case_age_days",
        "max_severity_score", "immediate_threat_flag", "societal_impact_score",
    ]
    TARGET = "priority_score"

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\nTest MAE  : {mae:.4f}")
    print(f"Test R²   : {r2:.4f}")
    print(f"(R²=1.0 = perfect fit on the legal rule labels)")

    # Plot predictions vs actuals
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidths=0.4)
    plt.plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
    plt.xlabel("Rule-Based Label (Dr. Saritha formula)")
    plt.ylabel("XGBoost Prediction")
    plt.title("XGBoost Priority Model v2 — Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    print(f"Plot saved to {PLOT_OUT}")

    return model, enc

# ------------------------------------------------------------------ #
#  4. SAVE
# ------------------------------------------------------------------ #
def save(model, encoder):
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    joblib.dump(encoder, ENCODER_OUT)
    print(f"\nModel saved  : {MODEL_OUT}")
    print(f"Encoder saved: {ENCODER_OUT}")

# ------------------------------------------------------------------ #
#  5. UPDATE DB WITH NEW PREDICTIONS
# ------------------------------------------------------------------ #
def update_db(model, encoder, df):
    print("\nUpdating database with new XGBoost predicted scores...")
    FEATURES = [
        "case_type_enc", "num_ipc_sections", "num_cpc_sections",
        "num_precedents", "total_words", "case_age_days",
        "max_severity_score", "immediate_threat_flag", "societal_impact_score",
    ]
    df["case_type_enc"] = encoder.transform(df["case_type"])
    preds = model.predict(df[FEATURES])
    preds = np.clip(preds, 0.0, 10.0).round(2)

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    for i, row in df.iterrows():
        cur.execute(
            "UPDATE cases SET priority_score = %s WHERE case_id = %s",
            (float(preds[i]), row["case_id"])
        )
    conn.commit()
    cur.close()
    conn.close()
    print(f"Updated {len(df)} cases in PostgreSQL with XGBoost v2 scores.")

# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    agent = PrioritizationAgent()
    rows  = fetch_cases()
    df    = generate_labels(rows, agent)
    model, encoder = train(df)
    save(model, encoder)
    update_db(model, encoder, df)
    print("\nDone! Restart backend/api.py to load the new model.")
