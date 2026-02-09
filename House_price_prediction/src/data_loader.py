"""
Data loading module for house price prediction.
Contains functions for loading and inspecting datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple


def load_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load California Housing dataset.
    
    Args:
        file_path: Path to CSV file (optional)
        
    Returns:
        Tuple of (features_df, target_series)
    """
    from sklearn.datasets import fetch_california_housing
    
    if file_path:
        # Load from CSV file
        data = pd.read_csv(file_path)
        X = data.drop(columns=['Price'])
        y = data['Price']
        return X, y
    else:
        # Load from sklearn
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='Price')
        return X, y


def inspect_data(data: pd.DataFrame, target: pd.Series) -> None:
    """
    Display basic information about the dataset.
    
    Args:
        data: Feature DataFrame
        target: Target Series
    """
    print("\n" + "="*50)
    print("DATA INSPECTION")
    print("="*50)
    
    print(f"Dataset Shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print(f"Target: {target.name}")
    
    print(f"\nFirst 5 rows:")
    print(pd.concat([data.head(), target.head()], axis=1))
    
    print(f"\nData Types:")
    print(data.dtypes)
    
    print(f"\nMissing Values:")
    print(data.isnull().sum())
    
    print(f"\nBasic Statistics:")
    print(data.describe())
    print("="*50)
    print()


def get_feature_info() -> Dict[str, Any]:
    """
    Get information about dataset features.
    
    Returns:
        Dictionary with feature descriptions and statistics
    """
    from sklearn.datasets import fetch_california_housing
    
    california = fetch_california_housing()
    
    feature_info = {
        'MedInc': {
            'name': 'Median Income',
            'description': 'Median income in block group',
            'units': 'tens of thousands of dollars ($10k)',
            'range': '0.5 to 15.0',
            'importance': 'High'
        },
        'HouseAge': {
            'name': 'House Age',
            'description': 'Median house age in block group',
            'units': 'years',
            'range': '1.0 to 52.0',
            'importance': 'Medium'
        },
        'AveRooms': {
            'name': 'Average Rooms',
            'description': 'Average number of rooms',
            'units': 'count',
            'range': '0.8 to 141.0',
            'importance': 'Medium'
        },
        'AveBedrms': {
            'name': 'Average Bedrooms',
            'description': 'Average number of bedrooms',
            'units': 'count',
            'range': '0.3 to 34.0',
            'importance': 'Medium'
        },
        'Population': {
            'name': 'Population',
            'description': 'Block group population',
            'units': 'count',
            'range': '3.0 to 35682.0',
            'importance': 'Medium'
        },
        'AveOccup': {
            'name': 'Average Occupancy',
            'description': 'Average number of people per household',
            'units': 'count',
            'range': '0.7 to 1243.0',
            'importance': 'Medium'
        },
        'Latitude': {
            'name': 'Latitude',
            'description': 'Geographic coordinate - latitude',
            'units': 'degrees',
            'range': '32.5 to 42.0',
            'importance': 'High'
        },
        'Longitude': {
            'name': 'Longitude',
            'description': 'Geographic coordinate - longitude',
            'units': 'degrees',
            'range': '-124.3 to -114.3',
            'importance': 'High'
        }
    }
    
    return feature_info
