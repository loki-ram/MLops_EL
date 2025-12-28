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


def generate_synthetic_dataset(path: Path, n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic dataset and save it to path."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "skill_match_score": np.random.uniform(0, 1, size=n),
        "complexity_level": np.random.randint(1, 6, size=n),  # 1..5
        "availability": np.random.uniform(0, 1, size=n),
        "past_performance": np.random.uniform(0, 1, size=n),
    })
    df["suitability_score"] = (
        0.5 * df["skill_match_score"]
        + 0.2 * (1.0 / df["complexity_level"])
        + 0.2 * df["availability"]
        + 0.1 * df["past_performance"]
        + np.random.normal(0, 0.02, size=n)
    )
    ensure_parent_dir(path)
    df.to_csv(path, index=False)
    print(f"[INFO] Generated synthetic dataset at {path.resolve()}")
    return df


# -------------------------
# 1️⃣ LOAD TRAINING DATA
# -------------------------
@task
def load_training_data(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load training (baseline) data.
    Default path: data/final.csv
    If file doesn't exist, a synthetic dataset is generated and saved to that path.
    """
    path_str = path or os.getenv("TRAINING_DATA_PATH", "data/final.csv")
    p = Path(path_str)
    print(f"[INFO] Loading training data from: {p}")

    if not p.exists():
        print(f"[WARN] Training data not found at {p}. Generating synthetic dataset for CI/tests.")
        df = generate_synthetic_dataset(p)
    else:
        df = pd.read_csv(p)
        print(f"[INFO] Loaded training data from {p} (rows={len(df)})")

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
    """
    Load production (current) data.
    Default path: data/production.csv
    If file doesn't exist, try to copy training data if available, otherwise generate synthetic.
    """
    path_str = path or os.getenv("PRODUCTION_DATA_PATH", "data/production.csv")
    p = Path(path_str)
    print(f"[INFO] Loading production data from: {p}")

    if not p.exists():
        # If training file exists, copy it (simulate production); otherwise generate synthetic.
        training_path = Path(os.getenv("TRAINING_DATA_PATH", "data/final.csv"))
        if training_path.exists():
            ensure_parent_dir(p)
            df = pd.read_csv(training_path)
            df.to_csv(p, index=False)
            print(f"[INFO] Production data not found. Copied training data to {p}")
        else:
            print(f"[WARN] Production data and training data missing; generating synthetic production dataset.")
            df = generate_synthetic_dataset(p)
    else:
        df = pd.read_csv(p)
        print(f"[INFO] Loaded production data from {p} (rows={len(df)})")

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
def train_model(df: pd.DataFrame, artifacts_dir: str = "artifacts") -> str:
    """
    Train a LinearRegression model on df and save it to artifacts_dir/model.pkl.
    Returns the model file path.
    """
    print("[INFO] Training model...")
    X = df[["skill_match_score", "availability", "past_performance", "complexity_inv"]]
    y = df["suitability_score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print(f"[INFO] Validation MAE: {mae:.4f}")
    print(f"[INFO] Validation R²: {r2:.4f}")

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_path / "model.pkl"
    joblib.dump(model, model_path)
    print(f"[INFO] Saved model to {model_path.resolve()}")

    return str(model_path.resolve())


# -------------------------
# 4️⃣ DATA DRIFT DETECTION
# -------------------------
@task
def run_drift_detection(reference_data: pd.DataFrame, current_data: pd.DataFrame, reports_dir: str = "reports") -> str:
    """
    Run Evidently data drift detection and save an HTML report.
    Returns the path to the saved report.
    """
    print("[INFO] Running data drift detection...")
    report = Report(metrics=[DataDriftPreset()])

    report.run(reference_data=reference_data, current_data=current_data)

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    report_path = reports_path / "data_drift_report.html"
    report.save_html(str(report_path))
    print(f"[INFO] Drift report saved to {report_path.resolve()}")

    return str(report_path.resolve())


# -------------------------
# 5️⃣ PREFECT FLOW
# -------------------------
@flow(name="Cloud MLOps Observability Pipeline")
def mlops_observability_pipeline(training_path: Optional[str] = None, production_path: Optional[str] = None):
    """
    Orchestrates loading data, training model, and running drift detection.
    Paths can be passed in or supplied via environment variables:
      TRAINING_DATA_PATH, PRODUCTION_DATA_PATH
    """
    training_path = training_path or os.getenv("TRAINING_DATA_PATH", "data/final.csv")
    production_path = production_path or os.getenv("PRODUCTION_DATA_PATH", "data/production.csv")

    train_df = load_training_data.submit(training_path).result()
    model_path = train_model.submit(train_df).result()

    # For drift detection we want baseline vs current
    prod_df = load_production_data.submit(production_path).result()

    report_path = run_drift_detection.submit(reference_data=train_df, current_data=prod_df).result()

    print(f"[DONE] Model saved at: {model_path}")
    print(f"[DONE] Drift report saved at: {report_path}")


# -------------------------
# 6️⃣ ENTRY POINT
# -------------------------
if __name__ == "__main__":
    # Running with optional environment variables:
    # TRAINING_DATA_PATH and PRODUCTION_DATA_PATH
    mlops_observability_pipeline()