# Technical Report: House Price Prediction System

**Author:** Mohammed Nabil A. Alyousefi
**Project repository:** [github.com/Mnabil-11/house-price-prediction](https://github.com/Mnabil-11/house-price-prediction)
**Type:** End-to-end regression system (EDA → modeling → serving → containerization)

---

## 0. Business Framing: What Would a Real Estate Company Gain?

Strip away the ML terminology and the question a real estate business
actually asks is simpler: *if an agency plugged this into their workflow
tomorrow, what would change?*

**Concretely:**

- **Instant first-pass pricing.** An agent listing a new property, or a
  buyer's agent screening 50 comparables, gets a defensible price estimate
  in milliseconds instead of waiting on a formal appraisal (which can take
  days and cost money for every property). This doesn't replace an
  appraisal for closing a deal, but it replaces the informal "gut feel"
  estimate agents currently use to have a first conversation with a client.
- **Consistency across agents.** Two agents at the same firm often price
  similar houses differently based on individual experience and bias. A
  shared model applies the *same* pricing logic to every property, which
  is a defensible, auditable standard rather than "whoever priced it felt
  X."
- **Explainable, not a black box (Section 7.4).** Because every prediction
  comes with its top 5 price drivers, an agent can tell a client *why* a
  number came out the way it did ("your kitchen finish quality is pulling
  the estimate down relative to comparable listings") — turning a number
  into a conversation and a concrete renovation recommendation, not just a
  verdict.
- **Scales without headcount.** The API can screen an entire portfolio or
  a new batch of listings overnight, something that isn't feasible to do
  manually appraisal-by-appraisal.

**What it doesn't replace:** licensed appraisals for financing/legal
purposes, on-the-ground market knowledge of a specific street, or
negotiation. See [Limitations](#8-limitations--next-steps) for where this
tool's judgment should be treated as a starting point, not a final word.

### What actually moves the price? (Top 3 factors, in practice)

The trained model's feature importances (gain-based, from the saved
XGBoost model) and the correlation analysis from EDA (Section 4 /
`notebooks/01_eda.ipynb`) agree on the same three drivers dominating the
prediction:

| Rank | Factor | Why it matters (business translation) |
|---|---|---|
| 1 | **Overall quality/finish level** (`OverallQual`) | By far the single largest lever in the model (~19% of total decision weight — more than the next 3 factors combined) and the strongest raw correlation with price (0.79). **Recommendation:** before listing, a pre-sale inspection focused on visible finish quality (kitchen, bath, materials) is the highest-leverage renovation spend an agency can recommend — it outweighs almost any other single upgrade in this dataset's pricing pattern. |
| 2 | **Total finished living space** (`TotalSF` — basement + 1st + 2nd floor combined) | Second-highest importance, and closely tracks `GrLivArea`'s 0.71 correlation with price. **Recommendation:** an unfinished basement or attic is a quantifiable pricing gap, not just extra storage — flag it to sellers as a specific, costed renovation ROI opportunity, and to buyers as underpriced upside if a comparable listing hasn't finished that space yet. |
| 3 | **Garage capacity** (`GarageCars`) | Consistently among the top predictors in both the correlation analysis (0.64) and the model's importance ranking. **Recommendation:** a 1-car garage in a neighborhood where 2-car is the norm is a specific, identifiable underpricing signal an agent can point to — either as a negotiation lever for buyers or a targeted improvement for sellers, rather than a vague "the market is what it is."|

The common thread across all three: **the model isn't just producing a
number, it's pointing at where renovation or negotiation effort is
actually worth spending**, which is the difference between a pricing tool
and a pricing *decision-support* tool.

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

**Status:** Deployed and live at
[house-price-prediction-obwh.onrender.com](https://house-price-prediction-obwh.onrender.com/docs),
with automated tests running on every push (Section 7) and per-prediction
explainability via SHAP (Section 7.4). See Section 8 for known limitations.

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
docker build -t house-price-api:1.3 .
docker run -d --name house-price-container --restart unless-stopped -p 8000:8000 house-price-api:1.3
```

**Image optimization:** `xgboost`'s standard PyPI wheel pulls in a
~300MB GPU dependency (`nvidia-nccl-cu12`) that is never used by this
CPU-only API. It is installed and removed within the same Docker layer,
reducing the image size by ~700MB with no functional cost, since the
layer caching means the large dependency never persists in the final
image history.

**Build caching:** `requirements.txt` is copied and installed in a layer
before the application code is copied in, so code-only changes during
development don't force a full dependency reinstall on rebuild.

### 7.4 Per-prediction explainability (SHAP)

Every `/predict` response includes a `top_factors` field: the 5 features
that most influenced that specific prediction, computed with `shap.TreeExplainer`
against the trained XGBoost model.

```json
{
  "predicted_price": 158478.44,
  "top_factors": [
    {"feature": "OverallQual", "impact_usd": 24512.68},
    {"feature": "YearsSinceRemodel", "impact_usd": -7060.09}
  ]
}
```

**A methodological caveat worth stating plainly:** the model predicts
`log(SalePrice + 1)`, so raw SHAP values are additive in log-space, not
in dollars. Converting them to a dollar figure requires inverting a
non-linear function (`expm1`), through which SHAP values don't translate
exactly. The dollar impact reported here is a **local linear
approximation** — each SHAP value is scaled by `exp(predicted_log_price)`,
the derivative of `expm1` at the predicted point. This is accurate for
ranking features and estimating rough magnitude, but is not a
mathematically exact decomposition of the dollar prediction, particularly
for features with very large log-space contributions. This tradeoff is
made explicit here and in the API's documentation rather than presenting
an approximation as an exact figure.

---

## 8. Limitations & Next Steps

Being transparent about what's *not* done is part of a credible technical
report:

| Gap | Impact | Status |
|---|---|---|
| SHAP dollar impacts are approximate | `top_factors` uses a local linear approximation to map log-space SHAP values to dollars, not an exact breakdown | Reliable for ranking/rough magnitude; documented as an approximation (Section 7.4) |
| Limited test coverage | The suite covers API/preprocessing correctness, not data drift or performance monitoring over time | Resolved for correctness (Section 7.4/9); monitoring is future work |
| Trained on 2006–2010 Ames, Iowa data | Won't generalize to other markets or time periods without retraining | Out of scope for this project; would require new labeled data |

**Resolved since the initial version of this report:** automated tests (`pytest`), a CI pipeline (GitHub Actions), live deployment (Render), and per-prediction explainability (SHAP) have all since been added -- see Section 7.

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
