import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn

# ============================================================
# 1️⃣ Load & Prepare Data
# ============================================================
df = pd.read_csv("final.csv")

cols = [
    'skill_match_score',
    'complexity_level',
    'availability',
    'past_performance',
    'suitability_score'
]
df = df[cols].dropna().reset_index(drop=True)

df['complexity_inv'] = 1.0 / df['complexity_level']

X = df[['skill_match_score', 'availability', 'past_performance', 'complexity_inv']]
y = df['suitability_score'] + np.random.normal(0, 0.015, size=len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_dir="D:\\MLops_EL\\models"
# ============================================================
# 2️⃣ MLflow Experiment
# ============================================================
mlflow.set_experiment("MLOps_EL_final")

# ============================================================
# 3️⃣ LINEAR REGRESSION
# ============================================================
with mlflow.start_run(run_name="Linear_Regression"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MAE", mean_absolute_error(y_test, preds))
    mlflow.log_metric("R2", r2_score(y_test, preds))

    mlflow.sklearn.log_model(model, "lr_model")


# ============================================================
# 4️⃣ XGBOOST REGRESSOR
# ============================================================
with mlflow.start_run(run_name="XGBoost_Regressor"):

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 4)

    mlflow.log_metric("MAE", mean_absolute_error(y_test, preds))
    mlflow.log_metric("R2", r2_score(y_test, preds))

    mlflow.sklearn.log_model(model, "xg_model")

# ============================================================
# 5️⃣ RANDOM FOREST REGRESSOR
# ============================================================
with mlflow.start_run(run_name="Random_Forest"):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 8)

    mlflow.log_metric("MAE", mean_absolute_error(y_test, preds))
    mlflow.log_metric("R2", r2_score(y_test, preds))
    mlflow.log_metric("MSE", mean_squared_error(y_test, preds))

    mlflow.sklearn.log_model(model, "rf_model")

print("\n✅ All 3 models logged successfully to MLflow!")
