# mlops_pipeline.py
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib

from prefect import flow, task

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# -------------------------
# Utilities
# -------------------------
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# 1️⃣ LOAD TRAINING DATA
# -------------------------
@task
def load_training_data(path: Optional[str] = None) -> pd.DataFrame:
    path_str = path or "data/final.csv"
    p = Path(path_str)

    if not p.exists():
        raise FileNotFoundError(f"Training data not found at {p}")

    df = pd.read_csv(p)

    cols = [
        "skill_match_score",
        "complexity_level",
        "availability",
        "past_performance",
        "suitability_score",
    ]

    df = df[cols].dropna().reset_index(drop=True)
    df["complexity_inv"] = 1.0 / df["complexity_level"]
    return df


# -------------------------
# 2️⃣ LOAD PRODUCTION DATA
# -------------------------
@task
def load_production_data(path: Optional[str] = None) -> pd.DataFrame:
    path_str = path or "data/production_new.csv"
    p = Path(path_str)

    if not p.exists():
        raise FileNotFoundError(f"Production data not found at {p}")

    df = pd.read_csv(p)

    cols = [
        "skill_match_score",
        "complexity_level",
        "availability",
        "past_performance",
        "suitability_score",
    ]

    df = df[cols].dropna().reset_index(drop=True)
    df["complexity_inv"] = 1.0 / df["complexity_level"]
    return df


# -------------------------
# 3️⃣ TRAIN MODEL
# -------------------------
@task
def train_model(df: pd.DataFrame) -> str:
    X = df[
        ["skill_match_score", "availability", "past_performance", "complexity_inv"]
    ]
    y = df["suitability_score"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print(f"[INFO] MAE: {mean_absolute_error(y_val, preds):.4f}")
    print(f"[INFO] R2 : {r2_score(y_val, preds):.4f}")

    Path("artifacts").mkdir(exist_ok=True)
    model_path = "artifacts/model.pkl"
    joblib.dump(model, model_path)

    return model_path


# -------------------------
# 4️⃣ DATA DRIFT DETECTION
# -------------------------
@task
def run_drift_detection(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> bool:
    print("[INFO] Running Evidently data drift detection...")

    report = Report(metrics=[DataDriftPreset(drift_share=0.7,stattest_threshold=0.05)])
    report.run(reference_data=reference_data, current_data=current_data)

    Path("reports").mkdir(exist_ok=True)
    report_path = "reports/data_drift_report.html"
    report.save_html(report_path)

    # 🔍 Extract drift result
    report_dict = report.as_dict()
    drift_detected = report_dict["metrics"][0]["result"]["dataset_drift"]

    print(f"[INFO] Drift detected: {drift_detected}")
    return drift_detected


# -------------------------
# 5️⃣ ALERT TASK (Prefect)
# -------------------------
@task
def alert_on_drift(drift_detected: bool):
    Path("reports").mkdir(exist_ok=True)

    alert_file = "reports/drift_status.txt"

    if drift_detected:
        with open(alert_file, "w") as f:
            f.write("DRIFT_DETECTED")
        print("🚨 ALERT: DATA DRIFT DETECTED!")
    else:
        with open(alert_file, "w") as f:
            f.write("NO_DRIFT")
        print("✅ No data drift detected.")



# -------------------------
# 6️⃣ PREFECT FLOW
# -------------------------
@flow(name="Cloud MLOps Observability Pipeline")
def mlops_observability_pipeline():
    train_df = load_training_data()
    prod_df = load_production_data()

    model_path = train_model(train_df)
    print(f"[INFO] Model saved at: {model_path}")

    drift_detected = run_drift_detection(train_df, prod_df)
    alert_on_drift(drift_detected)


# -------------------------
# 7️⃣ ENTRY POINT
# -------------------------
if __name__ == "__main__":
    mlops_observability_pipeline()
