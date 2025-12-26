import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load baseline & production data
baseline = pd.read_csv("data/baseline.csv")
production = pd.read_csv("data/production.csv")

# Create report
report = Report(metrics=[
    DataDriftPreset()
])

# Run report
report.run(
    reference_data=baseline,
    current_data=production
)

# Save report
report.save_html("reports/data_drift_report.html")

print("✅ Data drift report generated")
