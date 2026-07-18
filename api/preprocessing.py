"""Turns one HouseFeatures request into the exact 244-column row the model expects.

This mirrors notebooks/02_data_cleaning.ipynb and notebooks/03_feature_engineering.ipynb,
but for a single house instead of the whole training set.
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

with open(MODELS_DIR / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

# Same "NA means the feature doesn't exist" columns as in 02_data_cleaning.ipynb
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]
ZERO_COLS = ["GarageYrBlt", "MasVnrArea"]

# Reasonable fallbacks for the few columns that need a real imputed value
# rather than "None"/0 (matches the cleaned training data).
DEFAULT_LOT_FRONTAGE = 70.0
DEFAULT_ELECTRICAL = "SBrkr"

DROP_COLS = [
    "GarageArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "YearBuilt", "YearRemodAdd", "YrSold",
]


def preprocess(features: dict) -> pd.DataFrame:
    """features: a HouseFeatures.model_dump(by_alias=True) dict for one house."""
    df = pd.DataFrame([features])

    for col in NONE_COLS:
        df[col] = df[col].fillna("None")
    for col in ZERO_COLS:
        df[col] = df[col].fillna(0)

    df["LotFrontage"] = df["LotFrontage"].fillna(DEFAULT_LOT_FRONTAGE)
    df["Electrical"] = df["Electrical"].fillna(DEFAULT_ELECTRICAL)

    # Feature engineering, same as 03_feature_engineering.ipynb
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["YearsSinceRemodel"] = df["YrSold"] - df["YearRemodAdd"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"] +
                        df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    df = df.drop(columns=DROP_COLS)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Align to the exact columns/order the model was trained on: any dummy
    # column not produced for this single house (e.g. a Neighborhood that
    # wasn't this one) is filled with 0, matching a one-hot "not this category".
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df
