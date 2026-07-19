"""A small example of calling the API from Python instead of Swagger/curl.

Run the server first (uvicorn main:app --reload), then in another terminal:
    python example_client.py
"""

import requests

API_URL = "http://127.0.0.1:8000"

house = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "Neighborhood": "CollgCr",
    "YearBuilt": 2005,
    "YearRemodAdd": 2005,
    "TotalBsmtSF": 900,
    "1stFlrSF": 900,
    "2ndFlrSF": 900,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 8,
    "GarageCars": 2,
    "GarageArea": 480,
    "YrSold": 2010,
}

response = requests.post(f"{API_URL}/predict", json=house)

print("Status code:", response.status_code)
print("Response body:", response.json())

if response.status_code == 200:
    price = response.json()["predicted_price"]
    print(f"\nPredicted price: ${price:,.2f}")
