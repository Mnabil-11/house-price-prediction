# House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)

An end-to-end machine learning project that predicts a house's sale price from its features — from raw data exploration all the way to a containerized, deployed prediction API.

**Live demo:** https://house-price-prediction-obwh.onrender.com/docs
(hosted on Render's free tier — the first request after a period of inactivity may take ~30-60s to wake up)

## Table of contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Model results](#model-results)
- [Getting started](#getting-started)
- [Using the API](#using-the-api)
- [Running with Docker](#running-with-docker)
- [Key design decisions](#key-design-decisions)
- [Progress](#progress)
- 📄 **[Read the full technical report](./MODEL_CARD.md)** — problem     statement, data handling rationale, model comparison, and honest discussion of limitations.

## Overview  

Given a house's characteristics (size, quality, location, age, garage, ...), the API returns a predicted sale price. The project covers the full lifecycle of a small ML product:

```
Raw data → EDA → Cleaning → Feature engineering → Model comparison
         → Best model saved → FastAPI service → Docker container
```

## Pipeline

| Stage | Notebook / file | What happens |
|---|---|---|
| 1. EDA | `notebooks/01_eda.ipynb` | Missing-value audit, target distribution, correlation analysis |
| 2. Cleaning | `notebooks/02_data_cleaning.ipynb` | Impute missing values based on what they actually mean (e.g. "no pool" vs. genuinely missing) |
| 3. Feature engineering | `notebooks/03_feature_engineering.ipynb` | Remove outliers, log-transform the target, engineer features (`TotalSF`, `HouseAge`, `TotalBath`, ...), one-hot encode categoricals |
| 4. Modeling | `notebooks/04_model_training.ipynb` | Train and compare 5 regression models with 5-fold CV, save the best one |
| 5. Serving | `api/` | FastAPI service wrapping the saved model behind a `/predict` endpoint |
| 6. Packaging | `Dockerfile` | Containerize the API so it runs identically on any machine |

## Project structure

```
house-price/
├── data/
│   ├── raw/                         # original, unmodified data
│   └── processed/                   # cleaned / feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis
│   ├── 02_data_cleaning.ipynb       # missing value handling
│   ├── 03_feature_engineering.ipynb # outliers, new features, encoding
│   └── 04_model_training.ipynb      # model training, comparison, saving
├── models/
│   ├── xgboost_model.pkl            # final trained model
│   └── feature_columns.json         # exact feature order the model expects
├── api/
│   ├── main.py                      # FastAPI app (/predict endpoint)
│   ├── schemas.py                   # request/response models + validation
│   ├── preprocessing.py             # turns raw input into model-ready features
│   └── test_client.py               # example of calling the API from Python
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Dataset

The [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (the classic Kaggle "House Prices" competition data): 1460 houses, 79 raw features covering size, quality, location, age, and amenities, with `SalePrice` as the target.

## Model results

5 models were compared with 5-fold cross-validation and a held-out test set, on the log-transformed target:

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| **XGBoost (final model)** | 0.1242 | **0.1234** | **0.904** |
| Lasso | 0.1181 | 0.1256 | 0.900 |
| Ridge | 0.1228 | 0.1296 | 0.894 |
| Linear Regression | 0.1265 | 0.1301 | 0.893 |
| Random Forest | 0.1333 | 0.1423 | 0.872 |

XGBoost was selected as the final model and retrained on the full dataset before saving.

## Getting started

```bash
git clone https://github.com/Mnabil-11/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

To retrain the model, run the notebooks in `notebooks/` in order (01 → 04). The final trained model is already committed to `models/`, so this step is optional.

## Using the API

```bash
cd api
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "OverallQual": 8, "GrLivArea": 1800, "YearBuilt": 2005,
  "TotalBsmtSF": 900, "1stFlrSF": 900, "2ndFlrSF": 900,
  "FullBath": 2, "GarageCars": 2, "GarageArea": 480
}'
```

```json
{"predicted_price": 158478.44, "predicted_price_log": 11.973}
```

Or from Python (see `api/test_client.py` for a full example):

```python
import requests

response = requests.post("http://127.0.0.1:8000/predict", json={
    "OverallQual": 8, "GrLivArea": 1800, "YearBuilt": 2005,
})
print(response.json())
```

## Running with Docker

```bash
docker build -t house-price-api:1.1 .
docker run -d --name house-price-container --restart unless-stopped -p 8000:8000 house-price-api:1.1
```

The API is then available at http://127.0.0.1:8000/docs exactly like the local run above, isolated from whatever else is installed on the host machine.

## Key design decisions

- **NA is not always missing data**: for columns like `PoolQC` or `GarageType`, a blank value means "this house has no pool / no garage" rather than a data-entry gap, so they're imputed with `"None"`/`0` instead of a median.
- **Log-transformed target**: `SalePrice` is right-skewed; training on `log1p(SalePrice)` and inverting with `expm1` at prediction time keeps a few very expensive houses from distorting the loss.
- **Multicollinearity cleanup**: near-duplicate columns (e.g. `GarageCars` vs `GarageArea`) were collapsed to avoid destabilizing the linear models.
- **Input validation with headroom**: the API's `ge`/`le` bounds match the actual min/max seen in training, because tree-based models like XGBoost can't reliably extrapolate beyond the range they were trained on — bounding the input turns a silent, misleading prediction into an explicit 422 error.
- **Docker layer caching**: `requirements.txt` is copied and installed before the application code, so code-only changes don't force a full dependency reinstall on rebuild.
- **Lean image**: `xgboost`'s wheel pulls in a ~300MB GPU dependency (`nvidia-nccl-cu12`) that this CPU-only API never uses; it's installed and removed in the same Docker layer, cutting the final image from 1.8GB to 1.1GB.

## Progress

- [x] Dataset selection
- [x] Exploratory Data Analysis (EDA)
- [x] Data cleaning
- [x] Feature engineering
- [x] Model training and comparison
- [x] Save best model
- [x] FastAPI service
- [x] Docker
- [x] Deployment (Render)
