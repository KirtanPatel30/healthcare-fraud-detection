"""
dashboard/app.py
Streamlit dashboard for Healthcare Fraud Detection.
Shows:
  - Key metrics (fraud rate, model AUC, total claims)
  - Fraud probability distribution
  - Feature importance (SHAP)
  - Interactive claim predictor
  - Provider risk table
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_DIR = Path(__file__).parent.parent / "models"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

st.set_page_config(
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_DIR / "fraud_model.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        return model, scaler, feature_names
    except Exception as e:
        return None, None, None


@st.cache_data
def load_data():
    path = PROCESSED_DIR / "claims_features.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def load_metrics():
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_shap_importance():
    path = MODEL_DIR / "shap_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


# ─── HEADER ───────────────────────────────────────────────────────────────────

st.title("🏥 Healthcare Claims Fraud Detection")
st.markdown("**Production ML pipeline** | XGBoost + SHAP Explainability | Medicare Claims Analysis")
st.divider()

model, scaler, feature_names = load_artifacts()
df = load_data()
metrics = load_metrics()
shap_df = load_shap_importance()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["📊 Overview", "🔍 Predict Claim", "📈 Feature Importance", "🗃️ Data Explorer"])

model_status = "✅ Model Loaded" if model else "❌ Model Not Found (run train.py)"
st.sidebar.markdown(f"**Model Status:** {model_status}")

if metrics:
    st.sidebar.markdown("### Model Performance")
    st.sidebar.metric("AUC-ROC", f"{metrics.get('auc_roc', 'N/A'):.4f}")
    st.sidebar.metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.4f}")
    st.sidebar.metric("Avg Precision", f"{metrics.get('avg_precision', 'N/A'):.4f}")

# ─── PAGE: OVERVIEW ───────────────────────────────────────────────────────────

if page == "📊 Overview":
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Claims", f"{len(df):,}")
        col2.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
        col3.metric("Unique Providers", f"{df['provider_id'].nunique():,}")
        col4.metric("Avg Charge", f"${df['avg_charge_amount'].mean():,.0f}")

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Fraud Distribution by Specialty")
            fraud_by_spec = df.groupby("specialty")["is_fraud"].mean().reset_index()
            fraud_by_spec.columns = ["Specialty", "Fraud Rate"]
            fraud_by_spec = fraud_by_spec.sort_values("Fraud Rate", ascending=True)
            fig = px.bar(fraud_by_spec, x="Fraud Rate", y="Specialty", orientation="h",
                         color="Fraud Rate", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("Avg Charge: Fraud vs Legitimate")
            fig2 = px.box(df.sample(min(10000, len(df))), x="is_fraud", y="avg_charge_amount",
                          labels={"is_fraud": "Is Fraud", "avg_charge_amount": "Avg Charge ($)"},
                          color="is_fraud", color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Fraud Distribution by State")
        fraud_by_state = df.groupby("state")["is_fraud"].mean().reset_index()
        fraud_by_state.columns = ["state", "fraud_rate"]
        fig3 = px.choropleth(fraud_by_state, locations="state", locationmode="USA-states",
                             color="fraud_rate", scope="usa",
                             color_continuous_scale="Reds",
                             title="Fraud Rate by State")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No processed data found. Run `python pipeline/etl.py` first.")

# ─── PAGE: PREDICT CLAIM ──────────────────────────────────────────────────────

elif page == "🔍 Predict Claim":
    st.subheader("🔍 Real-Time Fraud Prediction")
    st.markdown("Enter claim details below to get a fraud probability score with SHAP explanation.")

    if model is None:
        st.error("Model not loaded. Run `python models/train.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            provider_id = st.text_input("Provider ID", "NPI0001234567")
            specialty = st.selectbox("Specialty", [
                "Internal Medicine", "Family Practice", "Cardiology", "Orthopedic Surgery",
                "Psychiatry", "Ophthalmology", "Dermatology", "Gastroenterology",
                "Neurology", "Oncology", "Radiology", "Anesthesiology", "Emergency Medicine"
            ])
            state = st.selectbox("State", ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"])
            place = st.selectbox("Place of Service", ["Office", "Inpatient Hospital", "Outpatient Hospital"])

        with col2:
            total_services = st.number_input("Total Services", 1, 10000, 500)
            avg_charge = st.number_input("Avg Charge ($)", 10.0, 5000.0, 400.0)
            avg_payment = st.number_input("Avg Medicare Payment ($)", 5.0, 5000.0, 280.0)
            unique_bene = st.number_input("Unique Beneficiaries", 1, 1000, 100)

        with col3:
            risk_score = st.slider("Avg Beneficiary Risk Score", 0.1, 5.0, 1.2, 0.1)
            service_days = st.number_input("Service Days", 1, 365, 120)
            years_practice = st.number_input("Years in Practice", 1, 50, 10)
            is_solo = st.selectbox("Solo Practice", [0, 1])

        if st.button("🔮 Predict Fraud Probability", type="primary"):
            submitted = avg_charge * total_services
            total_payment = avg_payment * total_services

            # Build feature dict
            from api.main import ClaimInput, engineer_features, SPECIALTY_MAP, STATE_MAP, PLACE_MAP
            claim = ClaimInput(
                provider_id=provider_id, specialty=specialty, state=state,
                years_in_practice=years_practice, is_solo_practice=is_solo,
                total_services=total_services, avg_charge_amount=avg_charge,
                avg_medicare_payment=avg_payment, unique_beneficiaries=unique_bene,
                avg_beneficiary_risk_score=risk_score, service_days=service_days,
                submitted_charge_amount=submitted, total_medicare_payment=total_payment,
                place_of_service=place
            )
            X = engineer_features(claim)
            X_aligned = X.reindex(columns=feature_names, fill_value=0)
            X_scaled = pd.DataFrame(scaler.transform(X_aligned), columns=feature_names)

            prob = float(model.predict_proba(X_scaled)[0][1])
            risk_level = "🟢 LOW" if prob < 0.3 else "🟡 MEDIUM" if prob < 0.6 else "🔴 HIGH"

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Fraud Probability", f"{prob:.2%}")
            r2.metric("Risk Level", risk_level)
            r3.metric("Prediction", "⚠️ FRAUD" if prob >= 0.5 else "✅ LEGITIMATE")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if prob > 0.5 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d5f5e3"},
                        {'range': [30, 60], 'color': "#fdebd0"},
                        {'range': [60, 100], 'color': "#fadbd8"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# ─── PAGE: FEATURE IMPORTANCE ─────────────────────────────────────────────────

elif page == "📈 Feature Importance":
    st.subheader("📈 SHAP Feature Importance")
    st.markdown("SHAP (SHapley Additive exPlanations) shows which features drive fraud predictions the most.")

    if shap_df is not None:
        top20 = shap_df.head(20).sort_values("mean_shap")
        fig = px.bar(top20, x="mean_shap", y="feature", orientation="h",
                     title="Top 20 Features by Mean |SHAP| Value",
                     color="mean_shap", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        shap_plot = MODEL_DIR / "plots" / "shap_summary.png"
        if shap_plot.exists():
            st.image(str(shap_plot), caption="SHAP Summary Plot", use_column_width=True)
    else:
        st.warning("SHAP values not found. Run `python models/train.py` first.")

# ─── PAGE: DATA EXPLORER ──────────────────────────────────────────────────────

elif page == "🗃️ Data Explorer":
    st.subheader("🗃️ Claims Data Explorer")
    if df is not None:
        fraud_filter = st.multiselect("Filter by Fraud Label", [0, 1], default=[0, 1])
        specialty_filter = st.multiselect("Filter by Specialty", df["specialty"].unique().tolist(),
                                           default=df["specialty"].unique().tolist())
        filtered = df[(df["is_fraud"].isin(fraud_filter)) & (df["specialty"].isin(specialty_filter))]
        st.write(f"Showing {len(filtered):,} records")

        display_cols = ["claim_id", "provider_id", "specialty", "state",
                        "total_services", "avg_charge_amount", "avg_medicare_payment",
                        "payment_charge_ratio", "is_fraud"]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols].head(500), use_container_width=True)
    else:
        st.warning("No data found. Run `python pipeline/etl.py` first.")
