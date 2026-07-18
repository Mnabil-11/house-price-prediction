"""Pydantic models describing the API's request and response bodies.

HouseFeatures mirrors the raw columns in data/raw/train.csv (minus Id and
SalePrice) so a caller can send the same kind of data a real house listing
would have. Fields that were "None means the feature doesn't exist" in the
EDA (PoolQC, Alley, GarageType, ...) default to None here too.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class HouseFeatures(BaseModel):
    MSSubClass: int = 20
    MSZoning: str = "RL"
    LotFrontage: Optional[float] = None
    LotArea: int = 9600
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
    OverallQual: int = 5
    OverallCond: int = 5
    YearBuilt: int = 1970
    YearRemodAdd: int = 1970
    RoofStyle: str = "Gable"
    RoofMatl: str = "CompShg"
    Exterior1st: str = "VinylSd"
    Exterior2nd: str = "VinylSd"
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str = "TA"
    ExterCond: str = "TA"
    Foundation: str = "PConc"
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: int = 0
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: int = 0
    BsmtUnfSF: int = 0
    TotalBsmtSF: int = 0
    Heating: str = "GasA"
    HeatingQC: str = "TA"
    CentralAir: str = "Y"
    Electrical: Optional[str] = None
    FirstFlrSF: int = Field(default=856, alias="1stFlrSF")
    SecondFlrSF: int = Field(default=0, alias="2ndFlrSF")
    LowQualFinSF: int = 0
    GrLivArea: int = 856
    BsmtFullBath: int = 0
    BsmtHalfBath: int = 0
    FullBath: int = 1
    HalfBath: int = 0
    BedroomAbvGr: int = 2
    KitchenAbvGr: int = 1
    KitchenQual: str = "TA"
    TotRmsAbvGrd: int = 5
    Functional: str = "Typ"
    Fireplaces: int = 0
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[float] = None
    GarageFinish: Optional[str] = None
    GarageCars: int = 1
    GarageArea: int = 250
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str = "Y"
    WoodDeckSF: int = 0
    OpenPorchSF: int = 0
    EnclosedPorch: int = 0
    ThreeSsnPorch: int = Field(default=0, alias="3SsnPorch")
    ScreenPorch: int = 0
    PoolArea: int = 0
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int = 0
    MoSold: int = 6
    YrSold: int = 2010
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
