"""
Data loading module for house price prediction.
Handles loading and initial data inspection.
"""

import pandas as pd
from typing import Optional


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Basic data inspection.
    
    Args:
        df: DataFrame to inspect
        
    Returns:
        Dictionary with basic info
    """
    return {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict()
    }
