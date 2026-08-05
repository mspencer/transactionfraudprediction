"""
main.py

FastAPI service serving the fraud inference method as a REST endpoint

Endpoints:
    GET /health    -> liveness check
    POST /score    -> score one or more transactions
"""

import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_artifacts, score_transactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")

app = FastAPI(title="Fraud Scoring API")

# loaded once at container startup, reused across requests
_artifacts = None

@app.on_event("startup")
async def startup_event():
    global _artifacts
    logger.info(f"Loading artifacts from {ARTIFACT_DIR}")
    _artifacts = load_artifacts(ARTIFACT_DIR)
    logger.info(f"Loaded {_artifacts['model_type']} model. Ready to serve.")

class ScoreRequest(BaseModel):
    # accepts a list of raw transaction records as dicts
    transactions: list[dict]

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _artifacts is not None, "model_type": _artifacts['model_type'] if _artifacts else None}

@app.post("/score")
async def score(request: ScoreRequest):
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

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