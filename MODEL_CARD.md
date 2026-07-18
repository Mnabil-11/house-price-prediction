# Technical Report: House Price Prediction System

**Author:** Mohammed Nabil A. Alyousefi
**Project repository:** [github.com/Mnabil-11/house-price-prediction](https://github.com/Mnabil-11/house-price-prediction)
**Type:** End-to-end regression system (EDA → modeling → serving → containerization)

---

## 1. Executive Summary

This project delivers a complete machine learning system that predicts a
residential property's sale price from its physical and locational
characteristics. It covers the full lifecycle of a small ML product: data
exploration, cleaning, feature engineering, model selection, and a
containerized REST API for serving predictions in real time.

The final model (XGBoost) explains **90.4% of the variance** in sale price
on held-out test data (R² = 0.904), with a test RMSE of 0.1234 on the
log-transformed target. The system is packaged as a Docker image and
exposes a validated, documented HTTP endpoint that returns a prediction in
milliseconds.

**Status:** Feature-complete through serving and containerization.
Production deployment is the one remaining step (see Section 8).

---

## 2. Problem Statement

Real estate pricing depends on dozens of interacting factors — size,
quality, age, location, and amenities — that are difficult to weigh
consistently by hand. The goal of this project is to build a system that:

1. Learns the relationship between a house's features and its sale price
   from historical transaction data.
2. Generalizes to unseen houses with measurable, quantified accuracy.
3. Is deployable as a service that a downstream application (e.g., a
   listings site, an internal pricing tool) could call directly.

This is framed as a **supervised regression problem**: predict a continuous
target (`SalePrice`) from a mix of numerical and categorical features.

---

## 3. Data

**Source:** [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
(Kaggle's "House Prices: Advanced Regression Techniques" competition data).

| Property | Value |
|---|---|
| Observations | 1,460 houses |
| Raw features | 79 (size, quality, location, age, amenities) |
| Target | `SalePrice` (continuous, USD) |

The Ames dataset is a well-established benchmark in the ML community
(successor to the older Boston Housing dataset), chosen specifically
because its richness — 79 raw features, many with non-trivial missingness
semantics — makes it representative of real-world tabular data problems
rather than a toy dataset.

---

## 4. Pipeline

| Stage | Artifact | What happens |
|---|---|---|
| 1. EDA | `notebooks/01_eda.ipynb` | Missing-value audit, target distribution, correlation analysis |
| 2. Cleaning | `notebooks/02_data_cleaning.ipynb` | Impute missing values based on what they actually mean |
| 3. Feature engineering | `notebooks/03_feature_engineering.ipynb` | Remove outliers, log-transform target, engineer features, one-hot encode |
| 4. Modeling | `notebooks/04_model_training.ipynb` | Train and compare 5 regression models with 5-fold CV, save the best one |
| 5. Serving | `api/` | FastAPI service wrapping the saved model behind a `/predict` endpoint |
| 6. Packaging | `Dockerfile` | Containerize the API for consistent, portable execution |

```
Raw data → EDA → Cleaning → Feature engineering → Model comparison
         → Best model saved → FastAPI service → Docker container
```

---

## 5. Feature Engineering & Data Handling

Three data-handling decisions materially affected model quality and are
worth calling out explicitly, since they reflect domain reasoning rather
than default library behavior:

### 5.1 Missingness is not always "missing"

For columns like `PoolQC` (pool quality) or `GarageType`, a blank value in
the raw data means **"this house has no pool / no garage"** — not a
data-entry gap. These were imputed with `"None"` (categorical) or `0`
(numerical) rather than a statistical fill value like the median, which
would have introduced a systematic bias (implying every house without a
pool has an "average" pool).

### 5.2 Log-transforming the target

`SalePrice` is right-skewed — a small number of very expensive homes would
otherwise dominate the loss function during training. The model is trained
on `log1p(SalePrice)` and predictions are inverted with `expm1` at
inference time. This is a standard technique for skewed monetary targets
and stabilizes both training and evaluation.

### 5.3 Multicollinearity cleanup

Near-duplicate columns (e.g., `GarageCars` and `GarageArea`, which both
encode garage size in different units) were collapsed. Multicollinearity
doesn't hurt tree-based models much, but it destabilizes coefficient
estimates in the linear models included in the comparison (Section 6), so
resolving it was necessary for a fair model comparison.

**Engineered features included:** `TotalSF` (combined square footage
across floors and basement), `HouseAge` (derived from `YearBuilt`),
`TotalBath` (combined full/half bathroom count) — each consolidates
several raw columns into a single, more predictive signal.

---

## 6. Model Comparison

Five regression models were evaluated with 5-fold cross-validation and a
held-out test set, all on the log-transformed target:

| Model | CV RMSE | Test RMSE | Test R² |
|---|---|---|---|
| **XGBoost (final model)** | 0.1242 | **0.1234** | **0.904** |
| Lasso | 0.1181 | 0.1256 | 0.900 |
| Ridge | 0.1228 | 0.1296 | 0.894 |
| Linear Regression | 0.1265 | 0.1301 | 0.893 |
| Random Forest | 0.1333 | 0.1423 | 0.872 |

**Model selection rationale:** XGBoost was selected as the final model
based on test-set performance, then **retrained on the full dataset**
(train + test combined) before being saved — a standard practice once a
model has been selected, since it lets the final artifact learn from every
available labeled example rather than leaving 20-30% of the data unused in
production.

**Note on Lasso's competitive CV score:** Lasso's cross-validation RMSE
(0.1181) is actually the best of the five models, edging out XGBoost. This
is a reasonable data point for a reader to notice — it suggests the
underlying relationship is largely linear/additive once the target is
log-transformed and features are engineered, and that XGBoost's edge on
the held-out test set (0.1234 vs. 0.1256) is comparatively modest. A
linear model would also be more interpretable and cheaper to serve; this
project prioritized XGBoost's test-set performance and its ability to
capture non-linear interactions without manual feature crosses, but Lasso
is a legitimate alternative worth documenting rather than dismissing.

---

## 7. Serving & Deployment Architecture

### 7.1 API

The trained model is wrapped in a **FastAPI** service exposing a single
`/predict` endpoint, with interactive documentation auto-generated via
Swagger UI (`/docs`).

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

### 7.2 Input validation with training-range bounds

The API's request schema enforces `ge`/`le` (greater/less-than-or-equal)
bounds matching the actual min/max values seen during training. This is a
deliberate safeguard: tree-based models like XGBoost cannot reliably
extrapolate beyond the value ranges they were trained on. An out-of-range
input (e.g., a house 10x larger than anything in the training set) would
otherwise produce a silent, misleading prediction. Bounding the input
converts that failure mode into an explicit `422 Unprocessable Entity`
error — a fail-loud rather than fail-silent design choice.

### 7.3 Containerization

```bash
docker build -t house-price-api:1.1 .
docker run -d --name house-price-container --restart unless-stopped -p 8000:8000 house-price-api:1.1
```

**Image optimization:** `xgboost`'s standard PyPI wheel pulls in a
~300MB GPU dependency (`nvidia-nccl-cu12`) that is never used by this
CPU-only API. It is installed and removed within the same Docker layer,
reducing the final image size from 1.8GB to 1.1GB — a ~39% reduction with
no functional cost, since the layer caching means the large dependency
never persists in the final image history.

**Build caching:** `requirements.txt` is copied and installed in a layer
before the application code is copied in, so code-only changes during
development don't force a full dependency reinstall on rebuild.

---

## 8. Limitations & Next Steps

Being transparent about what's *not* done is part of a credible technical
report:

| Gap | Impact | Planned resolution |
|---|---|---|
| No automated tests | Regressions in the API or preprocessing logic wouldn't be caught automatically | Add `pytest` + `httpx` test client coverage for `/predict` |
| No CI pipeline | No automatic verification on push/PR | Add a GitHub Actions workflow to run tests on every push |
| Not deployed | Model is only runnable locally or via manually-run Docker | Deploy to a host (Render/Railway) or add a Streamlit demo |
| No model interpretability tooling | Predictions aren't individually explainable | Add SHAP values for feature-level explanation per prediction |
| Trained on 2006–2010 Ames, Iowa data | Won't generalize to other markets or time periods without retraining | Out of scope for this project; would require new labeled data |

---

## 9. Reproducibility

```bash
git clone https://github.com/Mnabil-11/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

The trained model (`models/xgboost_model.pkl`) and its exact expected
feature order (`models/feature_columns.json`) are committed to the
repository, so retraining from the notebooks is optional — the API runs
against the already-trained artifact out of the box.

---

## 10. Conclusion

This project demonstrates a complete, defensible ML workflow: a documented
data-handling rationale (not just default imputation), a fair multi-model
comparison with both winner and runner-up reported honestly, a
production-minded serving layer with explicit failure modes, and a
containerization strategy that accounts for real deployment constraints
(image size, build caching). The remaining gap — production deployment and
automated testing — is clearly scoped rather than left implicit.
