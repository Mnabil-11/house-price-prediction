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
└── README.md
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
- [ ] FastAPI service
- [ ] Docker
- [ ] Deployment

More details will be added as the project progresses.
