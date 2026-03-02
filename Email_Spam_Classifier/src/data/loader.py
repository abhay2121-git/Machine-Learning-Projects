"""
Data loader module for email spam classifier.
Handles loading of CSV datasets with error handling and basic validation.
"""

import pandas as pd
from typing import Optional
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid CSV or is empty
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        print(f"Dataset shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nDataset info:")
        print(df.info())
        
        return df
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty or corrupted")
        raise ValueError("CSV file is empty or corrupted")
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise
