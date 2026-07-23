"""Tests for the house price prediction API.

Run with: pytest (from the api/ directory, or `pytest api/` from the repo root
with rootdir configured -- see the GitHub Actions workflow for the exact command).
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from preprocessing import preprocess, FEATURE_COLUMNS
from schemas import HouseFeatures

client = TestClient(app)


# --- Basic endpoints ---------------------------------------------------

def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "predictions_served" in body


# --- /predict happy path -------------------------------------------------

def test_predict_with_defaults_returns_a_price():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price" in body
    assert body["predicted_price"] > 0


def test_predict_includes_top_factors():
    response = client.post("/predict", json={"OverallQual": 8, "GrLivArea": 1800})
    body = response.json()
    factors = body["top_factors"]
    assert len(factors) == 5
    assert {"feature", "impact_usd"} <= factors[0].keys()
    # sorted by absolute impact, largest first
    impacts = [abs(f["impact_usd"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)


def test_predict_higher_quality_house_costs_more():
    """A bigger, higher-quality house should predict a higher price than a
    smaller, lower-quality one -- a basic sanity check that the model responds
    to its inputs in the expected direction."""
    small_house = {"OverallQual": 3, "GrLivArea": 600}
    big_house = {"OverallQual": 9, "GrLivArea": 3000}

    small_price = client.post("/predict", json=small_house).json()["predicted_price"]
    big_price = client.post("/predict", json=big_house).json()["predicted_price"]

    assert big_price > small_price


# --- /predict validation --------------------------------------------------

def test_predict_rejects_overallqual_above_ten():
    response = client.post("/predict", json={"OverallQual": 15})
    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    response = client.post("/predict", json={"OverallQual": "excellent"})
    assert response.status_code == 422


def test_predict_rejects_negative_area():
    response = client.post("/predict", json={"GrLivArea": -100})
    assert response.status_code == 422


# --- rate limiting -----------------------------------------------------------

def test_predict_rate_limit_eventually_kicks_in():
    """/predict is limited to 20 requests/minute per client. This test doesn't
    assume an exact request count (earlier tests in this module already used
    some of the budget against the same shared TestClient/app instance), it
    just keeps calling until a 429 shows up, which it must within a generous
    margin. Placed last among /predict tests since it exhausts the limit."""
    statuses = []
    for _ in range(30):
        response = client.post("/predict", json={"OverallQual": 5})
        statuses.append(response.status_code)
        if response.status_code == 429:
            break

    assert 429 in statuses


# --- preprocessing ---------------------------------------------------------

def test_preprocess_output_matches_trained_feature_columns():
    house = HouseFeatures().model_dump(by_alias=True)
    row = preprocess(house)
    assert list(row.columns) == FEATURE_COLUMNS
    assert row.shape[0] == 1


def test_preprocess_fills_missing_categoricals_as_none():
    house = HouseFeatures().model_dump(by_alias=True)
    row = preprocess(house)
    # With every "None means no such feature" field left unset, none of the
    # PoolQC dummy columns should be active (all fall into the dropped baseline).
    poolqc_cols = [c for c in row.columns if c.startswith("PoolQC_")]
    assert all(row[c].iloc[0] == 0 for c in poolqc_cols)
