"""
tests/test_pipeline.py
Unit tests for ETL pipeline and model training.
Run with: pytest tests/
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── ETL TESTS ────────────────────────────────────────────────────────────────

class TestDataIngestion:
    def test_generate_dataset_shape(self):
        from data.ingest import generate_medicare_dataset
        df = generate_medicare_dataset(n_providers=100, n_records=1000)
        assert len(df) == 1000
        assert "is_fraud" in df.columns
        assert "provider_id" in df.columns

    def test_fraud_rate_reasonable(self):
        from data.ingest import generate_medicare_dataset
        df = generate_medicare_dataset(n_providers=200, n_records=2000)
        fraud_rate = df["is_fraud"].mean()
        assert 0.01 < fraud_rate < 0.30, f"Fraud rate {fraud_rate:.2%} outside expected range"

    def test_no_null_critical_cols(self):
        from data.ingest import generate_medicare_dataset
        df = generate_medicare_dataset(n_providers=100, n_records=500)
        critical = ["provider_id", "avg_charge_amount", "is_fraud"]
        for col in critical:
            assert df[col].isnull().sum() == 0, f"Nulls found in {col}"


class TestValidation:
    def test_removes_zero_charges(self):
        from pipeline.etl import validate
        df = pd.DataFrame({
            "claim_id": ["A", "B", "C"],
            "provider_id": ["P1", "P2", "P3"],
            "avg_charge_amount": [0, 100, 200],
            "avg_medicare_payment": [0, 60, 120],
            "total_services": [1, 10, 20],
            "unique_beneficiaries": [1, 5, 10],
            "avg_beneficiary_risk_score": [1.0, 1.5, 2.0],
            "specialty": ["Cardiology"] * 3,
            "state": ["CA"] * 3,
            "is_fraud": [0, 0, 1],
        })
        result = validate(df)
        assert len(result) == 2  # row with 0 charge removed

    def test_removes_duplicates(self):
        from pipeline.etl import validate
        df = pd.DataFrame({
            "claim_id": ["A", "A", "B"],
            "provider_id": ["P1", "P1", "P2"],
            "avg_charge_amount": [100, 100, 200],
            "avg_medicare_payment": [60, 60, 120],
            "total_services": [10, 10, 20],
            "unique_beneficiaries": [5, 5, 10],
            "avg_beneficiary_risk_score": [1.0, 1.0, 1.5],
            "specialty": ["Cardiology"] * 3,
            "state": ["CA"] * 3,
            "is_fraud": [0, 0, 1],
        })
        result = validate(df)
        assert len(result) == 2


class TestFeatureEngineering:
    def setup_method(self):
        from data.ingest import generate_medicare_dataset
        self.df_raw = generate_medicare_dataset(n_providers=50, n_records=500)

    def test_payment_charge_ratio_created(self):
        from pipeline.etl import transform, validate
        df = validate(self.df_raw)
        result = transform(df)
        assert "payment_charge_ratio" in result.columns

    def test_log_features_created(self):
        from pipeline.etl import transform, validate
        df = validate(self.df_raw)
        result = transform(df)
        assert "log_avg_charge_amount" in result.columns
        assert "log_submitted_charge_amount" in result.columns

    def test_services_per_beneficiary_positive(self):
        from pipeline.etl import transform, validate
        df = validate(self.df_raw)
        result = transform(df)
        assert (result["services_per_beneficiary"] >= 0).all()


# ─── MODEL TESTS ──────────────────────────────────────────────────────────────

class TestModelPreparation:
    def setup_method(self):
        from data.ingest import generate_medicare_dataset
        from pipeline.etl import validate, transform
        df = generate_medicare_dataset(n_providers=100, n_records=1000)
        df = validate(df)
        self.df = transform(df)

    def test_features_available(self):
        from models.train import FEATURE_COLS
        available = [f for f in FEATURE_COLS if f in self.df.columns]
        assert len(available) >= 15, f"Only {len(available)} features available"

    def test_target_binary(self):
        assert set(self.df["is_fraud"].unique()).issubset({0, 1})


# ─── API TESTS ────────────────────────────────────────────────────────────────

class TestAPIFeatureEngineering:
    def test_engineer_features_returns_dataframe(self):
        from api.main import ClaimInput, engineer_features
        claim = ClaimInput(
            provider_id="NPI0001234567", specialty="Cardiology", state="CA",
            years_in_practice=10, is_solo_practice=0, total_services=500,
            avg_charge_amount=400.0, avg_medicare_payment=280.0,
            unique_beneficiaries=100, avg_beneficiary_risk_score=1.5,
            service_days=120, submitted_charge_amount=200000.0,
            total_medicare_payment=140000.0, place_of_service="Office"
        )
        X = engineer_features(claim)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == 1
        assert "payment_charge_ratio" in X.columns
        assert "services_per_beneficiary" in X.columns

    def test_high_risk_flag(self):
        from api.main import ClaimInput, engineer_features
        claim = ClaimInput(
            provider_id="NPI999", specialty="Oncology", state="TX",
            years_in_practice=5, is_solo_practice=1, total_services=1500,
            avg_charge_amount=1200.0, avg_medicare_payment=1000.0,
            unique_beneficiaries=50, avg_beneficiary_risk_score=3.5,  # > 2.0
            service_days=10, submitted_charge_amount=1800000.0,
            total_medicare_payment=1500000.0, place_of_service="Inpatient Hospital"
        )
        X = engineer_features(claim)
        assert X["high_risk_score_flag"].iloc[0] == 1
        assert X["inpatient_flag"].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
