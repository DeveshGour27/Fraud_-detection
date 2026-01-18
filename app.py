# app.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from transformers import (
    ColumnDropper,
    SafeTypeEncoder,
    PowerTransformColumns,
    ScaleColumns
)


PIPELINE_PATH = "fraud_pipeline.pkl"

try:
    model = joblib.load(PIPELINE_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a financial transaction is fraudulent",
    version="1.0.0"
)


class TransactionInput(BaseModel):
    step: int = Field(..., example=1)
    type: str = Field(..., example="TRANSFER")
    amount: float = Field(..., gt=0, example=10000.0)
    oldbalanceOrg: float = Field(..., ge=0, example=50000.0)
    newbalanceOrig: float = Field(..., ge=0, example=40000.0)
    oldbalanceDest: float = Field(..., ge=0, example=0.0)
    newbalanceDest: float = Field(..., ge=0, example=10000.0)

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "type": "TRANSFER",
                "amount": 10000.0,
                "oldbalanceOrg": 50000.0,
                "newbalanceOrig": 40000.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 10000.0
            }
        }

class PredictionOutput(BaseModel):
    is_fraud: int
    fraud_probability: float


# =========================
# Health Check
# =========================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):

    try:
        # Convert input to DataFrame
        data = pd.DataFrame([transaction.dict()])

        # Predict
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]

        return PredictionOutput(
            is_fraud=int(prediction),
            fraud_probability=round(float(probability), 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
