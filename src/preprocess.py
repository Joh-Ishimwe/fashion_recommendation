# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    """Preprocess the full dataset."""
    categorical_cols = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
    numerical_cols = ['year']
    
    # Handle missing values
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # Encode categorical variables
    for col in categorical_cols:
        df[col] = df[col].astype(str)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df[categorical_cols])
    
    # Combine with numerical data
    X = pd.concat([df[numerical_cols], df_encoded], axis=1)
    feature_columns = X.columns.tolist()  # Include numerical and encoded columns
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['usage'].fillna('Unknown'))
    
    return X_scaled, y_encoded, feature_columns, scaler, le

def preprocess_new_data(df_new, feature_columns, scaler):
    """Preprocess new data for prediction."""
    categorical_cols = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
    numerical_cols = ['year']
    
    # Handle missing values
    df_new[categorical_cols] = df_new[categorical_cols].fillna('Unknown')
    df_new[numerical_cols] = df_new[numerical_cols].fillna(df_new[numerical_cols].median())
    
    # Encode categorical variables
    for col in categorical_cols:
        df_new[col] = df_new[col].astype(str)
    
    # One-hot encoding with training feature columns
    df_encoded = pd.get_dummies(df_new[categorical_cols])
    # Reindex to match feature_columns, excluding numerical_cols (they're already in df_new[numerical_cols])
    encoded_feature_columns = [col for col in feature_columns if col not in numerical_cols]
    df_encoded = df_encoded.reindex(columns=encoded_feature_columns, fill_value=0)
    
    # Combine with numerical data
    X_new = pd.concat([df_new[numerical_cols], df_encoded], axis=1)
    
    # Ensure X_new has the same columns as feature_columns
    X_new = X_new.reindex(columns=feature_columns, fill_value=0)
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    return X_new_scaled

if __name__ == "__main__":
    pass