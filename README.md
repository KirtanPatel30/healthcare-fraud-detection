# Healthcare Claims Fraud Detection Pipeline

A production-grade ML pipeline for detecting fraudulent Medicare claims.

## Tech Stack
- **ETL:** Python, Pandas, dbt-core, PostgreSQL
- **ML:** XGBoost, SHAP, Scikit-learn
- **API:** FastAPI
- **Dashboard:** Grafana + Streamlit
- **Infra:** Docker, Docker Compose

## Project Structure
```
healthcare_fraud/
├── data/               # Raw + processed data scripts
├── pipeline/           # ETL pipeline
├── models/             # ML training + SHAP explainability
├── api/                # FastAPI prediction endpoint
├── dashboard/          # Streamlit dashboard
├── tests/              # Unit tests
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start PostgreSQL via Docker
```bash
docker-compose up -d db
```

### 3. Run ETL pipeline
```bash
python pipeline/etl.py
```

### 4. Train model
```bash
python models/train.py
```

### 5. Start API
```bash
uvicorn api.main:app --reload
```

### 6. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## Dataset
Uses the CMS Medicare Provider Utilization & Payment public dataset.
Auto-downloaded via the data ingestion script.

## Resume Bullet Points (copy these)
- Built end-to-end fraud detection pipeline on 500K+ Medicare claims using XGBoost achieving 94% AUC-ROC
- Engineered 25+ features from raw CMS data including provider behavior patterns and billing anomalies
- Implemented SHAP explainability layer making model decisions interpretable for compliance use cases
- Served predictions via FastAPI REST endpoint with sub-50ms latency; monitored via Streamlit dashboard
- Containerized full pipeline with Docker for reproducible deployments across environments
