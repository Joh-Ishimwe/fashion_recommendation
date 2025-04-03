# src/data.py
import os
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "Fashion-Styles_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Fashion_data")

def get_mongo_collection():
    """Connect to MongoDB and return the collection."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

def upload_csv_to_mongo(file_path: str):
    """Upload a CSV file to MongoDB."""
    try:
        df = pd.read_csv(file_path)
        collection = get_mongo_collection()
        collection.delete_many({})  # Clear existing data
        collection.insert_many(df.to_dict('records'))
        print(f"Uploaded {len(df)} records to MongoDB")
    except Exception as e:
        print(f"Error uploading to MongoDB: {e}")
        raise

def load_data_from_mongo():
    """Load data from MongoDB into a DataFrame."""
    try:
        collection = get_mongo_collection()
        data = list(collection.find())
        df = pd.DataFrame(data)
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
        return df
    except Exception as e:
        print(f"Error loading from MongoDB: {e}")
        raise

# Aliases to match pipeline.py imports
upload_to_mongo = upload_csv_to_mongo
load_from_mongo = load_data_from_mongo