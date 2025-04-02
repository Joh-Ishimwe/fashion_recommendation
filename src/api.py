# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from preprocess import preprocess_new_data  # Reuse your preprocessing function

app = FastAPI(title="Fashion Recommendation API")

# Load the trained model and artifacts
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Define input data model
class FashionItem(BaseModel):
    gender: str
    masterCategory: str
    subCategory: str
    articleType: str
    baseColour: str
    season: str
    year: int

@app.get("/")
async def root():
    return {"message": "Welcome to the Fashion Recommendation API"}

@app.post("/predict/")
async def predict_usage(item: FashionItem):
    """Predict the usage category for a fashion item."""
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([item.dict()], columns=feature_columns)
        
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