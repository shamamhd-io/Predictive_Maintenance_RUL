

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np

from src.inference.multi_dataset_predict import predict_rul_multi


# Initialize FastAPI application
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict Remaining Useful Life (RUL) of jet engines using sensor data.",
    version="1.0.0"
)


class PredictionRequest(BaseModel):

    dataset: str = Field(
        ...,
        description="Dataset name: FD001, FD002, FD003 or FD004."
    )

    sensor_data: List[List[float]] = Field(
        ...,
        description="Sensor readings for 30 cycles and dataset-specific features."
    )


# Home endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "supported_datasets": ["FD001", "FD002", "FD003", "FD004"],
        "models": {
            "FD001": "BiLSTM",
            "FD002": "BiLSTM",
            "FD003": "Transformer",
            "FD004": "Transformer"
        }
    }




# Multi-dataset RUL prediction endpoint

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        prediction = predict_rul_multi(
            request.dataset,
            request.sensor_data
        )

        return {
            "dataset": request.dataset,
            "predicted_rul": prediction,
            "unit": "cycles"
        }

    except (ValueError, TypeError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )
