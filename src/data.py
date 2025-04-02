# src/data.py
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration from .env
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

def get_mongo_collection():
    """Connect to MongoDB and return the collection."""
    try:
        if not MONGO_URI or not DB_NAME or not COLLECTION_NAME:
            raise ValueError("MongoDB configuration is missing. Check .env file.")
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print(f"Connected to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        return collection
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")
        raise

def upload_csv_to_mongo(file_path='data/styles.csv'):  # Updated to styles.csv
    """Upload CSV data to MongoDB, skipping bad lines."""
    try:
        # Load CSV into DataFrame, skipping malformed rows
        df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"Loaded {len(df)} records from {file_path} (skipped malformed rows)")
        
        # Convert DataFrame to list of dictionaries
        data = df.to_dict(orient='records')
        
        # Connect to MongoDB
        collection = get_mongo_collection()
        
        # Drop existing collection to avoid duplicates (optional)
        collection.drop()
        print(f"Dropped existing collection '{COLLECTION_NAME}'.")
        
        # Insert data
        collection.insert_many(data)
        print(f"Uploaded {len(data)} records to MongoDB collection '{COLLECTION_NAME}'.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except Exception as e:
        print(f"Error uploading to MongoDB: {str(e)}")
        raise

def load_data_from_mongo():
    """Load data from MongoDB into a DataFrame."""
    try:
        # Connect to MongoDB
        collection = get_mongo_collection()
        
        # Fetch all documents
        data = list(collection.find())
        if not data:
            print("No data found in MongoDB collection.")
            raise ValueError("MongoDB collection is empty.")
        
        # Convert to DataFrame and remove '_id' field
        df = pd.DataFrame(data).drop(columns=['_id'], errors='ignore')
        print(f"Loaded {len(df)} records from MongoDB.")
        return df
    except Exception as e:
        print(f"Error loading data from MongoDB: {str(e)}")
        raise

if __name__ == "__main__":
    # Upload the CSV to MongoDB (run this once)
    upload_csv_to_mongo()
    
    # Test loading from MongoDB
    df = load_data_from_mongo()
    print(df.head())