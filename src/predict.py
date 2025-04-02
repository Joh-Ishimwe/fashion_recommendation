# src/predict.py
import pandas as pd
import joblib

def make_predictions(model_path, X, le):
    """Make predictions on new data."""
    try:
        model = joblib.load(model_path)
        predictions = model.predict(X)
        predicted_categories = le.inverse_transform(predictions)
        return predicted_categories
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        return None
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None