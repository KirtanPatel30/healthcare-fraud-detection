#!/usr/bin/env python3
"""
run_all.py
Single script to run the entire pipeline end-to-end.
Usage: python run_all.py
"""

import subprocess
import sys
from pathlib import Path

def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"ERROR: '{cmd}' failed with code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    print("\n🏥 HEALTHCARE FRAUD DETECTION — FULL PIPELINE RUN")
    print("="*60)

    # Step 1: Generate data + ETL
    run("python pipeline/etl.py", "STEP 1/3: Running ETL Pipeline (data gen + feature engineering)")

    # Step 2: Train model
    run("python models/train.py", "STEP 2/3: Training XGBoost Model + SHAP Explainability")

    # Step 3: Run tests
    run("python -m pytest tests/ -v", "STEP 3/3: Running Unit Tests")

    print("\n" + "="*60)
    print("  ✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  Start API:       uvicorn api.main:app --reload")
    print("  Open dashboard:  streamlit run dashboard/app.py")
    print("  API docs:        http://localhost:8000/docs")
    print("  Dashboard:       http://localhost:8501")
    print("="*60)
