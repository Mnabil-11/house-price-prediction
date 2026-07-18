"""Pydantic models describing the API's request and response bodies.

HouseFeatures mirrors the raw columns in data/raw/train.csv (minus Id and
SalePrice) so a caller can send the same kind of data a real house listing
would have. Fields that were "None means the feature doesn't exist" in the
EDA (PoolQC, Alley, GarageType, ...) default to None here too.

Numeric fields carry ge/le bounds matching the min/max actually observed in
the training data. The model (XGBoost) can only interpolate within the range
it was trained on -- a value like OverallQual=15 doesn't raise an error on
its own, but the model has never seen anything above 10 and silently treats
it as if it were 10. Bounding the input here turns that silent, misleading
behavior into an explicit 422 error instead.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class HouseFeatures(BaseModel):
    MSSubClass: int = 20
    MSZoning: str = "RL"
    LotFrontage: Optional[float] = Field(default=None, ge=21, le=313)
    LotArea: int = Field(default=9600, ge=1300, le=215245)
    Street: str = "Pave"
    Alley: Optional[str] = None
    LotShape: str = "Reg"
    LandContour: str = "Lvl"
    Utilities: str = "AllPub"
    LotConfig: str = "Inside"
    LandSlope: str = "Gtl"
    Neighborhood: str = "NAmes"
    Condition1: str = "Norm"
    Condition2: str = "Norm"
    BldgType: str = "1Fam"
    HouseStyle: str = "1Story"
    OverallQual: int = Field(default=5, ge=1, le=10)
    OverallCond: int = Field(default=5, ge=1, le=10)
    YearBuilt: int = Field(default=1970, ge=1872, le=2010)
    YearRemodAdd: int = Field(default=1970, ge=1950, le=2010)
    RoofStyle: str = "Gable"
    RoofMatl: str = "CompShg"
    Exterior1st: str = "VinylSd"
    Exterior2nd: str = "VinylSd"
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = Field(default=None, ge=0)
    ExterQual: str = "TA"
    ExterCond: str = "TA"
    Foundation: str = "PConc"
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: int = Field(default=0, ge=0)
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: int = Field(default=0, ge=0)
    BsmtUnfSF: int = Field(default=0, ge=0)
    TotalBsmtSF: int = Field(default=0, ge=0)
    Heating: str = "GasA"
    HeatingQC: str = "TA"
    CentralAir: str = "Y"
    Electrical: Optional[str] = None
    FirstFlrSF: int = Field(default=856, ge=0, alias="1stFlrSF")
    SecondFlrSF: int = Field(default=0, ge=0, alias="2ndFlrSF")
    LowQualFinSF: int = Field(default=0, ge=0)
    GrLivArea: int = Field(default=856, ge=334, le=5642)
    BsmtFullBath: int = Field(default=0, ge=0, le=3)
    BsmtHalfBath: int = Field(default=0, ge=0, le=2)
    FullBath: int = Field(default=1, ge=0, le=3)
    HalfBath: int = Field(default=0, ge=0, le=2)
    BedroomAbvGr: int = Field(default=2, ge=0, le=8)
    KitchenAbvGr: int = Field(default=1, ge=0, le=3)
    KitchenQual: str = "TA"
    TotRmsAbvGrd: int = Field(default=5, ge=2, le=14)
    Functional: str = "Typ"
    Fireplaces: int = Field(default=0, ge=0, le=3)
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = Field(default=None, ge=1900, le=2010)
    GarageFinish: Optional[str] = None
    GarageCars: int = Field(default=1, ge=0, le=4)
    GarageArea: int = Field(default=250, ge=0)
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str = "Y"
    WoodDeckSF: int = Field(default=0, ge=0)
    OpenPorchSF: int = Field(default=0, ge=0)
    EnclosedPorch: int = Field(default=0, ge=0)
    ThreeSsnPorch: int = Field(default=0, ge=0, alias="3SsnPorch")
    ScreenPorch: int = Field(default=0, ge=0)
    PoolArea: int = Field(default=0, ge=0)
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int = Field(default=0, ge=0)
    MoSold: int = Field(default=6, ge=1, le=12)
    YrSold: int = Field(default=2010, ge=2006, le=2010)
    SaleType: str = "WD"
    SaleCondition: str = "Normal"

    model_config = ConfigDict(populate_by_name=True, json_schema_extra={
        "example": {
            "MSSubClass": 60,
            "MSZoning": "RL",
            "LotFrontage": 70,
            "LotArea": 9600,
            "Neighborhood": "CollgCr",
            "OverallQual": 7,
            "OverallCond": 5,
            "YearBuilt": 2005,
            "YearRemodAdd": 2005,
            "TotalBsmtSF": 900,
            "1stFlrSF": 900,
            "2ndFlrSF": 900,
            "GrLivArea": 1800,
            "FullBath": 2,
            "HalfBath": 1,
            "BedroomAbvGr": 3,
            "TotRmsAbvGrd": 8,
            "GarageCars": 2,
            "GarageArea": 480,
            "YrSold": 2010,
        }
    })


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_log: float
