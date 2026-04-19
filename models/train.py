"""
models/train.py
Trains XGBoost fraud detection model with:
  - SMOTE for class imbalance
  - Cross-validation
  - SHAP explainability
  - Model persistence
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    average_precision_score, f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = MODEL_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "total_services", "avg_charge_amount", "avg_medicare_payment",
    "unique_beneficiaries", "avg_beneficiary_risk_score", "service_days",
    "years_in_practice", "is_solo_practice",
    "payment_charge_ratio", "services_per_beneficiary",
    "charge_per_beneficiary", "payment_per_service",
    "high_risk_score_flag", "inpatient_flag",
    "log_submitted_charge_amount", "log_total_medicare_payment", "log_avg_charge_amount",
    "specialty_encoded", "state_encoded", "place_encoded",
    "provider_total_claims", "provider_avg_charge", "provider_avg_payment",
    "provider_avg_services", "provider_unique_beneficiaries",
    "charge_deviation", "services_deviation"
]

TARGET_COL = "is_fraud"


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "claims_features.csv"
    if not path.exists():
        print("Processed data not found. Running ETL pipeline first...")
        from pipeline.etl import run_pipeline
        run_pipeline()
    df = pd.read_csv(path)
    print(f"[DATA] Loaded {len(df):,} records")
    return df


def prepare_features(df: pd.DataFrame):
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features].fillna(0)
    y = df[TARGET_COL]
    print(f"[PREP] Features: {len(available_features)} | Fraud rate: {y.mean():.2%}")
    return X, y, available_features


def train_model(X_train, y_train):
    print("[TRAIN] Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[TRAIN] After SMOTE: {X_res.shape[0]:,} samples (fraud rate: {y_res.mean():.2%})")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=1,  # balanced by SMOTE
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(X_res, y_res, test_size=0.1, random_state=42)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    print("[TRAIN] Model training complete.")
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    print("\n[EVAL] Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print(f"  Average Precision:    {ap:.4f}")
    print(f"  F1 Score:             {f1:.4f}")
    print(f"{'='*50}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    metrics = {"auc_roc": round(auc, 4), "avg_precision": round(ap, 4), "f1_score": round(f1, 4)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Metrics saved to models/metrics.json")
    return metrics


def compute_shap(model, X_test, feature_names):
    print("\n[SHAP] Computing SHAP values (explainability)...")
    sample = X_test.sample(min(2000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Save SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved to models/plots/shap_summary.png")

    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Importance plot saved to models/plots/shap_importance.png")

    # Save mean absolute SHAP values per feature
    mean_shap = pd.DataFrame({
        "feature": feature_names,
        "mean_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)
    mean_shap.to_csv(MODEL_DIR / "shap_importance.csv", index=False)
    print("[SHAP] Feature importance saved to models/shap_importance.csv")

    return shap_values


def save_artifacts(model, scaler, feature_names):
    joblib.dump(model, MODEL_DIR / "fraud_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    print("[SAVE] Model artifacts saved to models/")


def run_training():
    print("=" * 60)
    print("HEALTHCARE FRAUD DETECTION — MODEL TRAINING")
    print("=" * 60)

    df = load_data()
    X, y, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

    # Cross-validation
    print("\n[CV] Running 5-fold cross-validation...")
    cv_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="auc", random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(cv_model, X_train_scaled, y_train, cv=5, scoring="roc_auc")
    print(f"[CV] AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Full training
    model = train_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test, feature_names)
    compute_shap(model, X_test_scaled, feature_names)
    save_artifacts(model, scaler, feature_names)

    print("\n[TRAINING] Complete! Model ready for API serving.")
    return model, metrics


if __name__ == "__main__":
    run_training()
