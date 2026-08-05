FROM python:3.11-slim

WORKDIR /app

# system dependencies needed for lightgbm/xgboost/catboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py main.py ./
COPY artifacts/ ./artifacts/

ENV ARTIFACT_DIR=artifacts
ENV PORT=8080

# CloudRun sets $PORT at runtime; uvicorn must bind to it
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}