"""
api/main.py
FastAPI REST endpoint for fraud prediction.
Endpoints:
  POST /predict        — single claim prediction with SHAP explanation
  POST /predict/batch  — batch predictions
  GET  /health         — health check
  GET  /metrics        — model performance metrics
"""

import os
import sys
import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(
    title="Healthcare Fraud Detection API",
    description="ML-powered Medicare claims fraud detection with SHAP explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts on startup
model = None
scaler = None
feature_names = None
explainer = None


@app.on_event("startup")
def load_model():
    global model, scaler, feature_names, explainer
    model_path = MODEL_DIR / "fraud_model.pkl"
    if not model_path.exists():
        print("WARNING: Model not found. Run models/train.py first.")
        return
    model = joblib.load(MODEL_DIR / "fraud_model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    print("Model loaded successfully.")


# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class ClaimInput(BaseModel):
    provider_id: str = Field(..., example="NPI0001234567")
    specialty: str = Field(..., example="Cardiology")
    state: str = Field(..., example="CA")
    years_in_practice: int = Field(..., example=15)
    is_solo_practice: int = Field(..., ge=0, le=1, example=0)
    total_services: int = Field(..., example=1200)
    avg_charge_amount: float = Field(..., example=850.0)
    avg_medicare_payment: float = Field(..., example=720.0)
    unique_beneficiaries: int = Field(..., example=80)
    avg_beneficiary_risk_score: float = Field(..., example=2.5)
    service_days: int = Field(..., example=30)
    submitted_charge_amount: float = Field(..., example=1020000.0)
    total_medicare_payment: float = Field(..., example=864000.0)
    place_of_service: str = Field(..., example="Inpatient Hospital")


class PredictionResponse(BaseModel):
    claim_id: Optional[str]
    provider_id: str
    fraud_probability: float
    is_fraud_predicted: bool
    risk_level: str
    top_risk_factors: List[dict]


# ─── FEATURE ENGINEERING (mirrors ETL) ────────────────────────────────────────

SPECIALTY_MAP = {
    "Internal Medicine": 0, "Family Practice": 1, "Cardiology": 2,
    "Orthopedic Surgery": 3, "Psychiatry": 4, "Ophthalmology": 5,
    "Dermatology": 6, "Gastroenterology": 7, "Neurology": 8,
    "Oncology": 9, "Radiology": 10, "Anesthesiology": 11, "Emergency Medicine": 12
}

STATE_MAP = {s: i for i, s in enumerate([
    "CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
])}

PLACE_MAP = {"Office": 0, "Inpatient Hospital": 1, "Outpatient Hospital": 2}


def engineer_features(claim: ClaimInput) -> pd.DataFrame:
    d = claim.dict()

    features = {
        "total_services": d["total_services"],
        "avg_charge_amount": d["avg_charge_amount"],
        "avg_medicare_payment": d["avg_medicare_payment"],
        "unique_beneficiaries": d["unique_beneficiaries"],
        "avg_beneficiary_risk_score": d["avg_beneficiary_risk_score"],
        "service_days": d["service_days"],
        "years_in_practice": d["years_in_practice"],
        "is_solo_practice": d["is_solo_practice"],
        "payment_charge_ratio": d["avg_medicare_payment"] / (d["avg_charge_amount"] + 1e-6),
        "services_per_beneficiary": d["total_services"] / (d["unique_beneficiaries"] + 1e-6),
        "charge_per_beneficiary": d["submitted_charge_amount"] / (d["unique_beneficiaries"] + 1e-6),
        "payment_per_service": d["total_medicare_payment"] / (d["total_services"] + 1e-6),
        "high_risk_score_flag": int(d["avg_beneficiary_risk_score"] > 2.0),
        "inpatient_flag": int(d["place_of_service"] == "Inpatient Hospital"),
        "log_submitted_charge_amount": np.log1p(d["submitted_charge_amount"]),
        "log_total_medicare_payment": np.log1p(d["total_medicare_payment"]),
        "log_avg_charge_amount": np.log1p(d["avg_charge_amount"]),
        "specialty_encoded": SPECIALTY_MAP.get(d["specialty"], -1),
        "state_encoded": STATE_MAP.get(d["state"], -1),
        "place_encoded": PLACE_MAP.get(d["place_of_service"], 0),
        # Provider-level features (use single claim values as proxy)
        "provider_total_claims": d["total_services"],
        "provider_avg_charge": d["avg_charge_amount"],
        "provider_avg_payment": d["avg_medicare_payment"],
        "provider_avg_services": d["total_services"],
        "provider_unique_beneficiaries": d["unique_beneficiaries"],
        "charge_deviation": 0.0,
        "services_deviation": 0.0,
    }

    return pd.DataFrame([features])


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.get("/metrics")
def get_metrics():
    metrics_path = MODEL_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Model metrics not found. Train the model first.")
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
def predict(claim: ClaimInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    X = engineer_features(claim)

    # Align features
    available = [f for f in feature_names if f in X.columns]
    X_aligned = X[available].reindex(columns=feature_names, fill_value=0)

    X_scaled = pd.DataFrame(scaler.transform(X_aligned), columns=feature_names)

    prob = float(model.predict_proba(X_scaled)[0][1])
    predicted = prob >= 0.5

    risk_level = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"

    # SHAP explanation
    shap_values = explainer.shap_values(X_scaled)[0]
    top_factors = sorted(
        [{"feature": f, "shap_value": round(float(s), 4)} for f, s in zip(feature_names, shap_values)],
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )[:5]

    return PredictionResponse(
        claim_id=None,
        provider_id=claim.provider_id,
        fraud_probability=round(prob, 4),
        is_fraud_predicted=predicted,
        risk_level=risk_level,
        top_risk_factors=top_factors
    )


@app.post("/predict/batch")
def predict_batch(claims: List[ClaimInput]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return [predict(claim) for claim in claims]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
