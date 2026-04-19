"""
pipeline/etl.py
Full ETL pipeline:
  1. Load raw Medicare claims CSV
  2. Clean & validate data
  3. Engineer features
  4. Run data quality checks
  5. Load into PostgreSQL
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fraud_user:fraud_pass@localhost:5432/healthcare_fraud")


# ─── 1. EXTRACT ───────────────────────────────────────────────────────────────

def extract(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = RAW_DIR / "medicare_claims.csv"
    if not path.exists():
        print("Raw data not found. Running ingestion...")
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.ingest import generate_medicare_dataset, save_raw_data
        df = generate_medicare_dataset()
        save_raw_data(df)
    df = pd.read_csv(path)
    print(f"[EXTRACT] Loaded {len(df):,} records from {path}")
    return df


# ─── 2. VALIDATE ──────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> pd.DataFrame:
    print("[VALIDATE] Running data quality checks...")
    initial_count = len(df)

    required_cols = [
        "claim_id", "provider_id", "specialty", "state",
        "total_services", "avg_charge_amount", "avg_medicare_payment",
        "unique_beneficiaries", "avg_beneficiary_risk_score", "is_fraud"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop duplicates
    df = df.drop_duplicates(subset=["claim_id"])

    # Drop nulls in critical fields
    df = df.dropna(subset=["provider_id", "avg_charge_amount", "avg_medicare_payment"])

    # Sanity checks — remove impossible values
    df = df[df["avg_charge_amount"] > 0]
    df = df[df["total_services"] > 0]
    df = df[df["unique_beneficiaries"] > 0]
    df = df[df["avg_medicare_payment"] <= df["avg_charge_amount"] * 1.1]  # payment can't far exceed charge

    dropped = initial_count - len(df)
    print(f"[VALIDATE] Dropped {dropped:,} invalid records. Remaining: {len(df):,}")
    return df


# ─── 3. TRANSFORM / FEATURE ENGINEERING ──────────────────────────────────────

def transform(df: pd.DataFrame) -> pd.DataFrame:
    print("[TRANSFORM] Engineering features...")

    # Payment ratio — fraud providers often have unusually high ratios
    df["payment_charge_ratio"] = df["avg_medicare_payment"] / (df["avg_charge_amount"] + 1e-6)

    # Services per beneficiary — fraud providers bill many services per patient
    df["services_per_beneficiary"] = df["total_services"] / (df["unique_beneficiaries"] + 1e-6)

    # Total submitted charge per beneficiary
    df["charge_per_beneficiary"] = df["submitted_charge_amount"] / (df["unique_beneficiaries"] + 1e-6)

    # Payment per service
    df["payment_per_service"] = df["total_medicare_payment"] / (df["total_services"] + 1e-6)

    # High risk score flag (risk score > 2.0 is unusual)
    df["high_risk_score_flag"] = (df["avg_beneficiary_risk_score"] > 2.0).astype(int)

    # Inpatient billing flag — fraud providers over-use inpatient
    df["inpatient_flag"] = (df["place_of_service"] == "Inpatient Hospital").astype(int)

    # Log transforms for skewed monetary features
    for col in ["submitted_charge_amount", "total_medicare_payment", "avg_charge_amount"]:
        df[f"log_{col}"] = np.log1p(df[col])

    # Encode categorical
    df["specialty_encoded"] = df["specialty"].astype("category").cat.codes
    df["state_encoded"] = df["state"].astype("category").cat.codes
    df["place_encoded"] = df["place_of_service"].astype("category").cat.codes

    # Provider-level aggregations (behavioral features)
    provider_agg = df.groupby("provider_id").agg(
        provider_total_claims=("claim_id", "count"),
        provider_avg_charge=("avg_charge_amount", "mean"),
        provider_avg_payment=("avg_medicare_payment", "mean"),
        provider_avg_services=("total_services", "mean"),
        provider_unique_beneficiaries=("unique_beneficiaries", "sum"),
    ).reset_index()

    df = df.merge(provider_agg, on="provider_id", how="left")

    # Claims deviation from provider mean (outlier detection)
    df["charge_deviation"] = df["avg_charge_amount"] - df["provider_avg_charge"]
    df["services_deviation"] = df["total_services"] - df["provider_avg_services"]

    print(f"[TRANSFORM] Feature engineering complete. Shape: {df.shape}")
    return df


# ─── 4. LOAD INTO POSTGRES ────────────────────────────────────────────────────

def load(df: pd.DataFrame, table_name: str = "claims_features"):
    print(f"[LOAD] Connecting to PostgreSQL...")
    engine = create_engine(DATABASE_URL)

    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS fraud;"))
        conn.commit()

    df.to_sql(
        name=table_name,
        con=engine,
        schema="fraud",
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi"
    )
    print(f"[LOAD] Loaded {len(df):,} records into fraud.{table_name}")

    # Also save processed CSV for offline use
    out_path = PROCESSED_DIR / "claims_features.csv"
    df.to_csv(out_path, index=False)
    print(f"[LOAD] Saved processed CSV to {out_path}")


# ─── 5. PIPELINE RUNNER ───────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("HEALTHCARE FRAUD DETECTION — ETL PIPELINE")
    print("=" * 60)

    df = extract()
    df = validate(df)
    df = transform(df)

    try:
        load(df)
    except Exception as e:
        print(f"[LOAD] PostgreSQL load failed ({e}). Saving to CSV only.")
        out_path = PROCESSED_DIR / "claims_features.csv"
        df.to_csv(out_path, index=False)
        print(f"[LOAD] Saved to {out_path}")

    print("\n[PIPELINE] ETL complete!")
    return df


if __name__ == "__main__":
    run_pipeline()
