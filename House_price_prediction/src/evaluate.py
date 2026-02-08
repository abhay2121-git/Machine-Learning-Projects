"""
Evaluation script for house price prediction.
Provides detailed model analysis and visualization utilities.
"""

import pandas as pd
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any, List
from model import HousePriceModel



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


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = model.predict(X_test)
    
    return {
        'mse': mean_squared_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'mean_prediction': np.mean(predictions),
        'std_prediction': np.std(predictions)
    }


def residual_analysis(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Perform residual analysis.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with residual statistics
    """
    predictions = model.predict(X_test)
    residuals = y_test - predictions
    
    return {
        'residuals': residuals,
        'residual_mean': np.mean(residuals),
        'residual_std': np.std(residuals),
        'predictions': predictions,
        'actual': y_test.values
    }


def feature_importance_analysis(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance
    """
    importance = model.get_feature_importance()
    
    if len(importance) == 0:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def predict_new_data(model, scaler, new_data: pd.DataFrame, 
                    feature_columns: List[str]) -> np.ndarray:
    """
    Make predictions on new data.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        new_data: New data to predict
        feature_columns: Expected feature columns
        
    Returns:
        Predictions array
    """
    # Ensure new_data has correct columns
    new_data_aligned = new_data[feature_columns]
    
    # Scale features
    scaled_data = scaler.transform(new_data_aligned)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    
    return predictions


if __name__ == "__main__":
    # Example usage
    model = load_model()
    scaler = load_scaler()
    
    # Load test data (example)
    # test_data = load_data('data/test.csv')
    # X_test = test_data.drop(columns=['price'])
    # y_test = test_data['price']
    
    print("Model evaluation utilities loaded successfully!")