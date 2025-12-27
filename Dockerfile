FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for some packages (xgboost)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . .

EXPOSE 8501

# Default model base (can be overridden at runtime / compose)
ENV MODEL_BASE=/app/mlruns

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
