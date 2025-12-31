# mlops_pipeline.py
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import joblib

from prefect import flow, task
import mlflow
import mlflow.sklearn


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score


from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


# -------------------------
# Utilities
# -------------------------
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri("file:./mlruns")

# -------------------------
# 1️⃣ LOAD TRAINING DATA
# -------------------------
@task
def load_training_data(path: Optional[str] = None) -> pd.DataFrame:
    path_str = path or os.getenv("TRAINING_DATA_PATH", "data/final.csv")
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
    path_str = path or os.getenv("PRODUCTION_DATA_PATH", "data/production_new.csv")
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


@task
def train_model(df: pd.DataFrame) -> str:
    # -----------------------------
    # Feature selection
    # -----------------------------
    X = df[
        ["skill_match_score", "availability", "past_performance", "complexity_inv"]
    ]
    y = df["suitability_score"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("MLOps_EL_final_final")

    with mlflow.start_run():

        # -----------------------------
        # Model parameters
        # -----------------------------
        params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": 42
        }

        mlflow.log_params(params)

        # -----------------------------
        # Train model
        # -----------------------------
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # -----------------------------
        # Validation
        # -----------------------------
        preds = model.predict(X_val)

        mse=mean_squared_error(y_val,preds)
        mape=mean_absolute_percentage_error(y_val,preds)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        print(f"Random Forest Metrics:")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"MAPE: {mape:.4f}")
        print(f"R²:   {r2:.4f}")


        # -----------------------------
        # Log metrics
        # -----------------------------
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        mlflow.log_metric("r2", r2)

        # -----------------------------
        # Save model artifact
        # -----------------------------
        Path("artifacts").mkdir(exist_ok=True)
        model_path = "artifacts/model.pkl"
        joblib.dump(model, model_path)

        # -----------------------------
        # Log model to MLflow
        # -----------------------------
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model"
        )

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
