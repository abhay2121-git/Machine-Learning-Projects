"""
Data preprocessing module for house price prediction.
Contains functions for data cleaning, feature engineering, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, List, Tuple


def handle_missing_values(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        data: Input DataFrame
        strategy: Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    print(f"\nHandling missing values with strategy: {strategy}")
    
    if strategy == 'drop':
        # Drop rows with missing values
        return data.dropna()
    elif strategy in ['mean', 'median', 'mode']:
        # Fill with specified strategy
        return data.fillna(data.mean())
    else:
        # Default to mean
        return data.fillna(data.mean())


def encode_categorical_features(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical features using label encoding.
    
    Args:
        data: Input DataFrame
        categorical_columns: List of categorical column names
        
    Returns:
        DataFrame with encoded categorical features
    """
    print(f"\nEncoding categorical features: {categorical_columns}")
    
    data_encoded = data.copy()
    
    for column in categorical_columns:
        if column in data_encoded.columns:
            le = LabelEncoder()
            data_encoded[column] = le.fit_transform(data_encoded[column])
            print(f"  Encoded {column} with {len(le.classes_)} unique values")
    
    return data_encoded


def scale_features(data: pd.DataFrame, numerical_columns: List[str]) -> Tuple[pd.DataFrame, Any]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        data: Input DataFrame
        numerical_columns: List of numerical column names
        
    Returns:
        Tuple of (scaled_data, fitted_scaler)
    """
    print(f"\nScaling {len(numerical_columns)} numerical features")
    
    scaler = StandardScaler()
    data_scaled = data.copy()
    data_scaled[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    return data_scaled, scaler
