# tests/test_api.py
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fashion Recommendation API"}

def test_predict():
    payload = {
        "gender": "women",
        "masterCategory": "apparel",
        "subCategory": "topwear",
        "articleType": "tshirts",
        "baseColour": "red",
        "season": "summer",
        "year": 2023
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    assert "predicted_usage" in response.json()

def test_invalid_year():
    payload = {
        "gender": "women",
        "masterCategory": "apparel",
        "subCategory": "topwear",
        "articleType": "tshirts",
        "baseColour": "red",
        "season": "summer",
        "year": 0
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 422