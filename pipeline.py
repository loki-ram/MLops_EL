import os
import pandas as pd
import numpy as np

# ----------------------------
# ML Models
# ----------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# MLflow
# ----------------------------
import mlflow
import mlflow.sklearn

# ----------------------------
# Prefect
# ----------------------------
from prefect import flow, task

# ----------------------------
# Evidently (LATEST API)
# ----------------------------
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# ============================================================
# Prefect Tasks
# ============================================================

@task
def load_data():
    df = pd.read_csv("D:/MLops_EL/data/final.csv")

    cols = [
        "skill_match_score",
        "complexity_level",
        "availability",
        "past_performance",
        "suitability_score"
    ]

    df = df[cols].dropna().reset_index(drop=True)
    df["complexity_inv"] = 1.0 / df["complexity_level"]

    return df


@task
def split_data(df):
    X = df[
        ["skill_match_score", "availability", "past_performance", "complexity_inv"]
    ]
    y = df["suitability_score"] + np.random.normal(0, 0.015, size=len(df))

    return train_test_split(X, y, test_size=0.2, random_state=42)


@task
def run_evidently(reference_data, current_data):
    os.makedirs("reports", exist_ok=True)

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    report.save_html("reports/data_drift_report.html")


@task
def train_and_log(model, run_name, X_train, X_test, y_train, y_test):

    with mlflow.start_run(run_name=run_name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("model_name", run_name)
        mlflow.log_metric("MAE", mean_absolute_error(y_test, preds))
        mlflow.log_metric("R2", r2_score(y_test, preds))

        mlflow.sklearn.log_model(model, artifact_path="model")


# ============================================================
# Prefect Flow
# ============================================================

@flow(name="Smart-Task-Allocation-MLOps-Pipeline")
def training_pipeline():

    mlflow.set_experiment("MLOps_EL_Final")

    # 1️⃣ Load Data
    df = load_data()

    # 2️⃣ Create Reference & Current Data (for drift)
    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.sample(frac=0.3, random_state=99)

    # 3️⃣ Run Evidently AI
    run_evidently(reference_data, current_data)

    # 4️⃣ Train/Test Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 5️⃣ Train & Log Models
    train_and_log(
        LinearRegression(),
        "LinearRegression",
        X_train, X_test, y_train, y_test
    )

    train_and_log(
        XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            objective="reg:squarederror",
            random_state=42
        ),
        "XGBoost",
        X_train, X_test, y_train, y_test
    )

    train_and_log(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42
        ),
        "RandomForest",
        X_train, X_test, y_train, y_test
    )


# ============================================================
# Run Pipeline
# ============================================================

if __name__ == "__main__":
    training_pipeline()
