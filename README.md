# House Price Prediction

End-to-end machine learning project: from raw data to a deployable price prediction API.

## Project structure

```
house-price/
├── data/
│   ├── raw/          # original, unmodified data
│   └── processed/    # cleaned / feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory data analysis
│   ├── 02_data_cleaning.ipynb       # missing value handling
│   ├── 03_feature_engineering.ipynb # outliers, new features, encoding
│   └── 04_model_training.ipynb      # model training, comparison, saving
├── models/
│   ├── xgboost_model.pkl     # final trained model
│   └── feature_columns.json  # feature order expected by the model
├── api/
│   ├── main.py           # FastAPI app (/predict endpoint)
│   ├── schemas.py        # request/response models
│   └── preprocessing.py  # turns raw input into model-ready features
├── requirements.txt
└── README.md
```

## Running the API

```
pip install -r requirements.txt
cd api
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI, or POST house features to `/predict`:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "OverallQual": 8, "GrLivArea": 1800, "YearBuilt": 2005,
  "TotalBsmtSF": 900, "1stFlrSF": 900, "2ndFlrSF": 900,
  "FullBath": 2, "GarageCars": 2, "GarageArea": 480
}'
```

## Model results

5 models were compared with 5-fold cross-validation and a held-out test set (log-transformed `SalePrice`):

| Model | Test RMSE | Test R² |
|---|---|---|
| **XGBoost (final model)** | 0.1234 | 0.904 |
| Lasso | 0.1256 | 0.900 |
| Ridge | 0.1296 | 0.894 |
| Linear Regression | 0.1301 | 0.893 |
| Random Forest | 0.1423 | 0.872 |

## Progress

- [x] Dataset selection
- [x] Exploratory Data Analysis (EDA)
- [x] Data cleaning
- [x] Feature engineering
- [x] Model training and comparison
- [x] Save best model
- [x] FastAPI service
- [ ] Docker
- [ ] Deployment

More details will be added as the project progresses.
