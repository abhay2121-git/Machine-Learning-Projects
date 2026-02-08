"""
Data preprocessing module for house price prediction.
Handles cleaning, feature engineering, and data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
from src.data_loader import load_data



def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for missing values ('mean', 'median', 'mode', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    if strategy == 'drop':
        return df_clean.dropna()
    
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if strategy in ['mean', 'median']:
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                fill_value = df_clean[col].mean() if strategy == 'mean' else df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
    
    return df_clean


def encode_categorical_features(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns
    
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    return df_encoded


def scale_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column (not to be scaled)
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df_scaled = df.copy()
    feature_columns = [col for col in df.columns if col != target_column]
    
    scaler = StandardScaler()
    df_scaled[feature_columns] = scaler.fit_transform(df_scaled[feature_columns])
    
    return df_scaled, scaler