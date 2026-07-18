"""FastAPI service that predicts a house's sale price using the saved XGBoost model."""

from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI

from schemas import HouseFeatures, PredictionResponse
from preprocessing import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts a house's sale price from its features using a trained XGBoost model.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {"status": "ok", "message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    row = preprocess(house.model_dump(by_alias=True))
    pred_log = model.predict(row)[0]
    pred_price = np.expm1(pred_log)  # undo the log1p transform from training

    return PredictionResponse(
        predicted_price=round(float(pred_price), 2),
        predicted_price_log=float(pred_log),
    )
