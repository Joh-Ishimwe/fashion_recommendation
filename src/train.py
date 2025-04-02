import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier 
import joblib

def train_and_evaluate(X, y, feature_columns, le, data_dir, model_dir):
    """Train and evaluate the model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save split data
    pd.DataFrame(X_train, columns=feature_columns).to_csv(f'{data_dir}X_train.csv', index=False)
    pd.DataFrame(X_test, columns=feature_columns).to_csv(f'{data_dir}X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['usage']).to_csv(f'{data_dir}y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['usage']).to_csv(f'{data_dir}y_test.csv', index=False)
    print("Split data saved to data/ directory")
    
    # Define the model with CPU-only usage
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', device='cpu')
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Evaluation with dynamic labels
    test_labels = le.inverse_transform(np.unique(y_test))  # Only classes in y_test
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=test_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_labels, yticklabels=test_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{model_dir}confusion_matrix.png')
    plt.close()
    
    return best_model

if __name__ == "__main__":
    # For standalone testing (assuming preprocessed data is available)
    pass