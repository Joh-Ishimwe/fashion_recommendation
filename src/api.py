# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import joblib
from preprocess import preprocess_new_data
from datetime import datetime
from models import FashionItem
from config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH, LABEL_ENCODER_PATH
from data import get_mongo_collection


app = FastAPI(title="Fashion Recommendation API")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Define valid categories (based on your dataset)
VALID_GENDERS = ["men", "women", "boys", "girls", "unisex"]
VALID_MASTER_CATEGORIES = ["apparel", "accessories", "footwear", "personal care", "free items"]
VALID_SEASONS = ["summer", "winter", "spring", "fall"]


@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion Recommendation API"}


@app.get("/fashion_items/")
async def get_fashion_items(gender: str = None, limit: int = 10):
    try:
        collection = get_mongo_collection()
        query = {"gender": gender} if gender else {}
        items = list(collection.find(query).limit(limit))
        for item in items:
            item["_id"] = str(item["_id"])
        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict/")
async def predict_usage(item: FashionItem):
    """Predict the usage category for a fashion item."""
    try:
        # Convert input to DataFrame with raw columns
        raw_columns = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "year"]
        input_data = pd.DataFrame([item.model_dump()], columns=raw_columns)
        
        # Preprocess the input
        X_new_scaled = preprocess_new_data(input_data, feature_columns, scaler)
        
        # Make prediction
        y_pred = model.predict(X_new_scaled)
        predicted_usage = le.inverse_transform(y_pred)[0]
        
        return {"predicted_usage": predicted_usage}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)