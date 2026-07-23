"""FastAPI service that predicts a house's sale price using the saved XGBoost model."""

import logging
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from schemas import HouseFeatures, PredictionResponse
from preprocessing import preprocess
from explainer import build_explainer, top_contributions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("house_price_api")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts a house's sale price from its features using a trained XGBoost model.",
    version="1.1.0",
)

# /predict runs XGBoost + SHAP on every call -- not free -- so it's rate
# limited per client IP. /health and / are left unlimited since Render's
# uptime checks poll them continuously.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

logger.info("Loading model from %s", MODEL_PATH)
model = joblib.load(MODEL_PATH)
explainer = build_explainer(model)
logger.info("Model and SHAP explainer loaded successfully")

# Simple in-process counter. Resets on restart and isn't shared across
# workers/replicas -- fine for observability on this single-instance
# deployment, not a substitute for a real metrics backend at higher scale.
prediction_count = 0


@app.get("/")
def root():
    return {"status": "ok", "message": "House Price Prediction API is running"}


@app.get("/health")
def health():
    return {"status": "healthy", "predictions_served": prediction_count}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")
def predict(request: Request, house: HouseFeatures):
    global prediction_count

    try:
        row = preprocess(house.model_dump(by_alias=True))
        pred_log = model.predict(row)[0]
        pred_price = np.expm1(pred_log)  # undo the log1p transform from training
        factors = top_contributions(explainer, row, pred_log, top_n=5)
    except Exception:
        logger.error(
            "Prediction failed for OverallQual=%s, GrLivArea=%s",
            house.OverallQual, house.GrLivArea, exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Prediction failed")

    prediction_count += 1
    logger.info(
        "Prediction #%d: OverallQual=%s GrLivArea=%s -> $%.2f",
        prediction_count, house.OverallQual, house.GrLivArea, pred_price,
    )

    return PredictionResponse(
        predicted_price=round(float(pred_price), 2),
        predicted_price_log=float(pred_log),
        top_factors=factors,
    )
