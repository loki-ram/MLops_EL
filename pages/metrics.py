import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

st.set_page_config(page_title="Model Monitoring", layout="wide")

st.title("📊 Model Monitoring & Observability")

# ============================================================
# 1️⃣ MLflow Metrics Section
# ============================================================
st.header("🔹 MLflow Model Metrics")

client = MlflowClient()

experiment = client.get_experiment_by_name("MLOps_EL_final")

if experiment:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    metrics_data = []

    for run in runs:
        metrics_data.append({
            "Run Name": run.data.tags.get("mlflow.runName", "N/A"),
            "MAE": run.data.metrics.get("MAE"),
            "R2": run.data.metrics.get("R2"),
            "MSE": run.data.metrics.get("MSE")
        })

    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(df_metrics)

else:
    st.warning("No MLflow experiment found")

# ============================================================
# 2️⃣ Evidently AI Section
# ============================================================
st.header("🔹 Evidently AI – Data Drift Report")

report_path = "D:\\MLops_EL\\reports\\data_drift_report.html"

if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    st.components.v1.html(html_content, height=600, scrolling=True)
else:
    st.warning("Evidently report not found")

# ============================================================
# 3️⃣ Prefect Section
# ============================================================
st.header("🔹 Prefect Pipeline Monitoring")

st.markdown(
    """
    Prefect is used to orchestrate and monitor ML pipelines.

    🔗 **Prefect Dashboard**  
    👉 [Open Prefect UI](http://localhost:4200)

    The dashboard shows:
    - Flow run status
    - Execution history
    - Logs and failures
    """
)
