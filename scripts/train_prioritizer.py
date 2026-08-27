"""
Train ML model for case prioritization (FIXED VERSION)
Aligned with your real features pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "data/prioritization/labeled_cases.csv"
MODEL_DIR = "data/prioritization/models"
MODEL_FILE = f"{MODEL_DIR}/prioritizer.pkl"
ENCODER_FILE = f"{MODEL_DIR}/encoders.pkl"

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)


# ==============================
# FEATURE PREPARATION
# ==============================
def prepare_features(df):
    """
    Prepare features based on ACTUAL pipeline
    """

    feature_cols = [
        "case_type",
        "num_ipc_sections",
        "num_cpc_sections",
        "num_precedents",
        "total_words",
        "case_age_days",
        "max_severity_score",
        "immediate_threat_flag",
        "societal_impact_score"
    ]

    X = df[feature_cols].copy()

    # Handle missing
    X["case_age_days"] = X["case_age_days"].fillna(X["case_age_days"].median())
    X["max_severity_score"] = X["max_severity_score"].fillna(3)
    X["immediate_threat_flag"] = X["immediate_threat_flag"].fillna(0)
    X["societal_impact_score"] = X["societal_impact_score"].fillna(1)

    # Encode case_type
    le = LabelEncoder()
    X["case_type"] = le.fit_transform(X["case_type"])

    joblib.dump(le, ENCODER_FILE)
    print(f"Encoder saved -> {ENCODER_FILE}")

    return X


# ==============================
# TRAIN MODEL
# ==============================
def train_model(X, y):
    print("\nTraining model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror"
    )

    # Cross-validation
    print("\nCross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    print(f"CV MAE: {-cv_scores.mean():.3f}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    print("\nEvaluation:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"R²: {r2_score(y_test, y_pred):.3f}")

    # Feature importance
    print("\nFeature Importance:")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(importance)

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved -> {MODEL_FILE}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 10], [0, 10], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    plt.grid(True)
    plt.savefig(f"{MODEL_DIR}/plot.png")

    return model


# ==============================
# MAIN
# ==============================
def main():
    print("\nLoading data...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} cases")

    print("\nPreparing features...")
    X = prepare_features(df)
    y = df["priority_score"]

    print("\nTraining...")
    model = train_model(X, y)

    print("\nDONE")


if __name__ == "__main__":
    main()