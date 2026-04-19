"""
data/ingest.py
Generates a realistic Medicare-like dataset using fully vectorized NumPy operations.
Fast: 500K records in ~10 seconds.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

SPECIALTIES = [
    "Internal Medicine", "Family Practice", "Cardiology", "Orthopedic Surgery",
    "Psychiatry", "Ophthalmology", "Dermatology", "Gastroenterology",
    "Neurology", "Oncology", "Radiology", "Anesthesiology", "Emergency Medicine"
]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI",
          "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
PLACES = ["Office", "Inpatient Hospital", "Outpatient Hospital"]


def generate_medicare_dataset(n_providers=5000, n_records=500000):
    print(f"Generating {n_records:,} Medicare claim records...")

    provider_ids = np.array([f"NPI{str(i).zfill(10)}" for i in range(1, n_providers + 1)])
    specialty_idx = np.random.randint(0, len(SPECIALTIES), n_providers)
    state_idx     = np.random.randint(0, len(STATES), n_providers)
    years         = np.random.randint(1, 40, n_providers)
    solo          = np.random.randint(0, 2, n_providers)

    fraud_mask = np.zeros(n_providers, dtype=bool)
    fraud_mask[np.random.choice(n_providers, size=int(n_providers * 0.05), replace=False)] = True

    rec_prov_idx  = np.random.randint(0, n_providers, n_records)
    is_fraud_prov = fraud_mask[rec_prov_idx]

    n_fraud = is_fraud_prov.sum()
    n_legit = n_records - n_fraud

    total_services = np.empty(n_records, dtype=np.int32)
    avg_charge     = np.empty(n_records)
    avg_payment    = np.empty(n_records)
    unique_bene    = np.empty(n_records, dtype=np.int32)
    risk_score     = np.empty(n_records)
    service_days   = np.empty(n_records, dtype=np.int32)
    place_idx_arr  = np.empty(n_records, dtype=np.int32)

    fi = is_fraud_prov
    total_services[fi] = np.random.randint(500, 5000, n_fraud)
    avg_charge[fi]     = np.random.uniform(300, 2000, n_fraud)
    avg_payment[fi]    = avg_charge[fi] * np.random.uniform(0.7, 0.95, n_fraud)
    unique_bene[fi]    = np.random.randint(50, 300, n_fraud)
    risk_score[fi]     = np.random.uniform(1.5, 4.0, n_fraud)
    service_days[fi]   = np.random.randint(1, 50, n_fraud)
    place_idx_arr[fi]  = np.random.choice([0, 1, 2], n_fraud, p=[0.2, 0.5, 0.3])

    li = ~is_fraud_prov
    total_services[li] = np.random.randint(10, 800, n_legit)
    avg_charge[li]     = np.random.uniform(50, 600, n_legit)
    avg_payment[li]    = avg_charge[li] * np.random.uniform(0.3, 0.65, n_legit)
    unique_bene[li]    = np.random.randint(5, 200, n_legit)
    risk_score[li]     = np.random.uniform(0.5, 2.0, n_legit)
    service_days[li]   = np.random.randint(1, 200, n_legit)
    place_idx_arr[li]  = np.random.choice([0, 1, 2], n_legit, p=[0.6, 0.2, 0.2])

    label     = (is_fraud_prov & (np.random.random(n_records) < 0.85)).astype(np.int8)
    submitted = avg_charge * total_services
    total_pay = avg_payment * total_services

    df = pd.DataFrame({
        "claim_id":                   [f"CLM{i:07d}" for i in range(n_records)],
        "provider_id":                provider_ids[rec_prov_idx],
        "specialty":                  [SPECIALTIES[i] for i in specialty_idx[rec_prov_idx]],
        "state":                      [STATES[i]      for i in state_idx[rec_prov_idx]],
        "years_in_practice":          years[rec_prov_idx].astype(int),
        "is_solo_practice":           solo[rec_prov_idx].astype(int),
        "total_services":             total_services,
        "avg_charge_amount":          avg_charge.round(2),
        "avg_medicare_payment":       avg_payment.round(2),
        "unique_beneficiaries":       unique_bene,
        "avg_beneficiary_risk_score": risk_score.round(3),
        "service_days":               service_days,
        "place_of_service":           [PLACES[i] for i in place_idx_arr],
        "submitted_charge_amount":    submitted.round(2),
        "total_medicare_payment":     total_pay.round(2),
        "is_fraud":                   label,
    })

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    return df


def save_raw_data(df: pd.DataFrame):
    out_path = RAW_DIR / "medicare_claims.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved raw data to {out_path}")
    return out_path


if __name__ == "__main__":
    df = generate_medicare_dataset()
    save_raw_data(df)
    print(f"\nSample:\n{df.head()}")
    print(f"\nClass distribution:\n{df['is_fraud'].value_counts()}")