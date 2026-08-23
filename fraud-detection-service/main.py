"""
main.py

FastAPI service serving the fraud inference method as a REST endpoint

Endpoints:
    GET /health    -> liveness check
    POST /score    -> score one or more transactions
"""

from contextlib import asynccontextmanager
import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_artifacts, score_transactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

# loaded once at container startup, reused across requests
_artifacts = None

def get_artifacts():
    # lazy loader: loads model artifacts on first demand if not already in memory
    global _artifacts
    if _artifacts is None:
        logger.info(f"Loading artifacts from {ARTIFACT_DIR}")
        _artifacts = load_artifacts(ARTIFACT_DIR)
        logger.info(f"Loaded {_artifacts['model_type']} model. Ready to serve.")
    return _artifacts

@asynccontextmanager
async def lifespan(app: FastAPI):
    # pre warm up model cache during startup
    try:
        get_artifacts()
    except Exception as e:
        logger.error(f"Failed to load artifacts on startup: {e}")
    yield

app = FastAPI(title="Fraud Scoring API", lifespan=lifespan)

class ScoreRequest(BaseModel):
    # accepts a list of raw transaction records as dicts
    transactions: list[dict]

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _artifacts is not None,
        "model_type": _artifacts['model_type'] if _artifacts else None
    }

@app.post("/score")
async def score(request: ScoreRequest):
    artifacts = get_artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model unavailable")

    if not request.transactions:
        raise HTTPException(status_code=400, detail="No transactions provided")

    try:
        raw_df = pd.DataFrame(request.transactions)
        scored = score_transactions(raw_df, _artifacts)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {e}")
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring error: {e}")

    id_col = "TransactionID" if "TransactionID" in scored.columns else None
    results = []
    for _, row in scored.iterrows():
        entry = {
            "fraud_probability": float(row['fraud_probability']),
            "is_fraud_predicted": int(row['is_fraud_predicted'])
        }
        if id_col:
            entry["TransactionID"] = row[id_col]
        results.append(entry)

    return {"threshold": _artifacts['threshold'], "results": results}