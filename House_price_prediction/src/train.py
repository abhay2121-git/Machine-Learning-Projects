"""
Training script for house price prediction.
Orchestrates the complete training pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

from data_loader import load_data, inspect_data
from preprocessing import handle_missing_values, encode_categorical_features, scale_features
from model import HousePriceModel

def train_model(data_path: str, target_column: str, model_type: str = 'linear', 
                test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Complete training pipeline.
    
    Args:
        data_path: Path to the dataset
        target_column: Name of target column
        model_type: Type of model to train
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    # Load data
    df = load_data(data_path)
    
    # Preprocessing
    df_clean = handle_missing_values(df)
    df_encoded = encode_categorical_features(df_clean)
    
    # Split features and target
    X = df_encoded.drop(columns=[target_column])
    y = df_encoded[target_column]
    
    # Scale features
    X_scaled, scaler = scale_features(pd.concat([X, y], axis=1), target_column)
    X_scaled = X_scaled.drop(columns=[target_column])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = HousePriceModel(model_type)
    model.train(X_train, y_train)
    
    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    dump(model, 'artifacts/model.joblib')
    dump(scaler, 'artifacts/scaler.joblib')
    
    return {
        'model_type': model_type,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importance': model.get_feature_importance(),
        'feature_names': X.columns.tolist()
    }


if __name__ == "__main__":
    # Example usage
    results = train_model('data/house_prices.csv', 'price', model_type='rf')
    print("Training completed!")
    print(f"Test R²: {results['test_metrics']['r2']:.4f}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
