import os
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load models (paths configurable via MODEL_BASE env var)
# -----------------------------
model_base = os.environ.get("MODEL_BASE", "mlruns")
try:
    lr_model = joblib.load(os.path.join(model_base, "1", "models", "m-9fa1d1bfe5d943a792a7bce554db1be0", "artifacts", "model.pkl"))
    rf_model = joblib.load(os.path.join(model_base, "1", "models", "m-f76ae34aee404a7b8e52db8c062df03f", "artifacts", "model.pkl"))
    xg_model = joblib.load(os.path.join(model_base, "1", "models", "m-f803976efe4442018fdc1605dd9a26be", "artifacts", "model.pkl"))
except Exception as e:
    st.error(f"Failed to load models from '{model_base}': {e}")
    st.stop()

st.set_page_config(page_title="Smart Task Allocation", layout="centered")

st.title("🧠 Smart Task Allocation System")
st.write("Compare predictions from multiple ML models")

# -----------------------------
# Input Section
# -----------------------------
st.header("🔹 Input Details")

employee_skills = st.text_input(
    "Employee Skills (comma-separated)",
    "python, django, sql"
)

required_skills = st.text_input(
    "Required Task Skills (comma-separated)",
    "python, flask, sql"
)

complexity_level = st.selectbox(
    "Task Complexity",
    [1, 2, 3],
    format_func=lambda x: {1: "Easy", 2: "Medium", 3: "Hard"}[x]
)

availability = st.slider("Availability", 0.0, 1.0, 0.8)
past_performance = st.slider("Past Performance", 0.0, 1.0, 0.9)

# -----------------------------
# Feature Engineering
# -----------------------------
def skill_match(emp, req):
    emp, req = set(emp), set(req)
    return len(emp & req) / len(emp | req) if emp | req else 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Suitability"):

    emp_list = [s.strip().lower() for s in employee_skills.split(",")]
    req_list = [s.strip().lower() for s in required_skills.split(",")]

    skill_score = skill_match(emp_list, req_list)
    complexity_inv = 1.0 / complexity_level

    X = np.array([[skill_score, availability, past_performance, complexity_inv]])

    preds = {
        "Linear Regression": lr_model.predict(X)[0],
        "Random Forest": rf_model.predict(X)[0],
        "XGBoost": xg_model.predict(X)[0]
    }

    # -----------------------------
    # Model Accuracies (user-provided)
    # -----------------------------
    st.header("🔎 Model Accuracies")
    lr_acc = st.number_input("Linear Regression Accuracy", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
    rf_acc = st.number_input("Random Forest Accuracy", min_value=0.0, max_value=1.0, value=0.88, step=0.01)
    xg_acc = st.number_input("XGBoost Accuracy", min_value=0.0, max_value=1.0, value=0.90, step=0.01)

    acc_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "XGBoost"],
        "Accuracy": [lr_acc, rf_acc, xg_acc]
    }).set_index("Model")

    st.line_chart(acc_df["Accuracy"])
    # ...existing code...

    # -----------------------------
    # Display Results
    # -----------------------------
    st.header("📊 Model Predictions")

    results_df = pd.DataFrame.from_dict(
        preds, orient="index", columns=["Suitability Score"]
    )

    st.table(results_df.round(3))

    best_model = max(preds, key=preds.get)
    st.success(f"🏆 Best Score predicted by: **{best_model}**")
