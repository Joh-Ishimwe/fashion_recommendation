# src/pipeline.py
import os

import pandas as pd
from data import load_data_from_mongo, upload_csv_to_mongo
from preprocess import preprocess_data, preprocess_new_data
from train import train_and_evaluate
from predict import make_predictions
import joblib

def run_pipeline(data_dir='data/', model_dir='models/', upload=False):
    """Run the full ML pipeline with MongoDB data."""
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Optionally upload CSV to MongoDB (run once)
    if upload:
        print("Uploading data to MongoDB...")
        upload_csv_to_mongo(file_path=f'{data_dir}styles.csv')  # Updated to styles.csv
    
    # Load data from MongoDB
    print("Loading data from MongoDB...")
    df = load_data_from_mongo()
    
    # Preprocess data
    print("Preprocessing data...")
    X_scaled, y_encoded, feature_columns, scaler, le = preprocess_data(df)
    
    # Train and evaluate
    print("Training model...")
    best_model = train_and_evaluate(X_scaled, y_encoded, feature_columns, le, data_dir, model_dir)
    
    # Save artifacts
    joblib.dump(best_model, f'{model_dir}best_model.pkl')
    joblib.dump(scaler, f'{model_dir}scaler.pkl')
    joblib.dump(feature_columns, f'{model_dir}feature_columns.pkl')
    joblib.dump(le, f'{model_dir}label_encoder.pkl')
    print("Artifacts saved to models/ directory")
    
    # Test prediction
    print("Testing prediction...")
    X_new = pd.DataFrame({
        'gender': ['Women'],
        'masterCategory': ['Apparel'],
        'subCategory': ['Topwear'],
        'articleType': ['Tshirts'],
        'baseColour': ['Blue'],
        'season': ['Summer'],
        'year': [2023]
    })
    X_new_scaled = preprocess_new_data(X_new, feature_columns, scaler)
    predicted_categories = make_predictions(f'{model_dir}best_model.pkl', X_new_scaled, le)
    if predicted_categories is not None:
        print("Predicted Usage:", predicted_categories[0])
    else:
        print("Prediction failed.")

if __name__ == "__main__":
    # Set upload=True the first time, then False for subsequent runs
    run_pipeline(upload=False)  # Data is already uploaded, so False