"""
Training module for house price prediction.
Contains functions for model training and artifact management.
"""

import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional

from data_loader import load_data
from preprocessing import handle_missing_values, encode_categorical_features, scale_features


def train_model(X: pd.DataFrame, y: pd.Series, alpha: float = 0.1, 
                l1_ratio: float = 0.5, test_size: float = 0.2, 
                random_state: int = 42, model_save_path: str = 'artifacts/model.joblib') -> Tuple[ElasticNet, Any]:
    """
    Train ElasticNet regression model and save artifacts.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        alpha: Regularization strength
        l1_ratio: Mix between L1 and L2 regularization
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        model_save_path: Path to save trained model
        
    Returns:
        Tuple of (trained_model, scaler)
    """
    print(f"\nTraining ElasticNet Regression...")
    print(f"  Hyperparameters:")
    print(f"    - Alpha (regularization): {alpha}")
    print(f"    - L1 ratio: {l1_ratio}")
    print(f"    - L1 (Lasso) weight: {l1_ratio * 100:.0f}%")
    print(f"    - L2 (Ridge) weight: {(1-l1_ratio) * 100:.0f}%")
    
    # Handle missing values
    X = handle_missing_values(X, strategy='mean')
    
    # Encode categorical features if any
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        X = encode_categorical_features(X, categorical_columns)
    
    # Scale features
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    X_scaled, scaler = scale_features(X, numerical_columns)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=random_state)
    model.fit(X_train, y_train)
    
    print("  Model trained successfully")
    print(f"\n  Model Coefficients:")
    
    # Display feature importance
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(coef_df.to_string(index=False))
    print(f"\n  Intercept: {model.intercept_:.4f}")
    
    # Save artifacts
    os.makedirs('artifacts', exist_ok=True)
    dump(model, model_save_path)
    dump(scaler, 'artifacts/scaler.joblib')
    
    print(f"\nArtifacts saved to {model_save_path} and artifacts/scaler.joblib")
    
    return model, scaler


def load_model(model_path: str = 'artifacts/model.joblib') -> Any:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    return load(model_path)


def load_scaler(scaler_path: str = 'artifacts/scaler.joblib') -> Any:
    """
    Load fitted scaler from disk.
    
    Args:
        scaler_path: Path to saved scaler
        
    Returns:
        Loaded scaler
    """
    return load(scaler_path)
