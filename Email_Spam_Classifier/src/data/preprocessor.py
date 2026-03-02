"""
Data preprocessing module for email spam classifier.
Handles data cleaning and preprocessing for pre-processed word frequency data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess the word frequency dataset and split into train and test sets.
    
    Args:
        df (pd.DataFrame): Raw dataset with word frequencies and Prediction column
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        X_train, X_test, y_train, y_test
        
    Raises:
        ValueError: If required columns are not found in the dataset
    """
    print("Starting data preprocessing for word frequency data...")
    
    # Check for required columns
    if 'Prediction' not in df.columns:
        raise ValueError("Prediction column not found in dataset")
    if 'Email No.' in df.columns:
        print("Found Email No. column - will be excluded from features")
    
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Step 1: Remove identifier column if present
    if 'Email No.' in df_clean.columns:
        df_clean = df_clean.drop('Email No.', axis=1)
        print("Removed Email No. column")
    
    # Step 2: Check for missing values
    missing_values = df_clean.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values - filling with 0")
        df_clean = df_clean.fillna(0)
    else:
        print("No missing values found")
    
    # Step 3: Separate features and target
    X = df_clean.drop('Prediction', axis=1)
    y = df_clean['Prediction']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Step 4: Check label distribution
    label_counts = y.value_counts()
    print(f"Label distribution: Ham (0): {label_counts.get(0, 0)}, Spam (1): {label_counts.get(1, 0)}")
    print(f"Label percentages: Ham (0): {label_counts.get(0, 0)/len(y)*100:.1f}%, Spam (1): {label_counts.get(1, 0)/len(y)*100:.1f}%")
    
    # Step 5: Basic feature filtering (remove very sparse features)
    # Remove columns that are all zeros or have very low variance
    zero_cols = (X == 0).all()
    if zero_cols.any():
        print(f"Removing {zero_cols.sum()} columns that are all zeros")
        X = X.loc[:, ~zero_cols]
    
    # Optional: Remove features with very low frequency (appear in less than 1% of emails)
    min_freq = max(1, int(0.01 * len(X)))  # At least 1% of documents or 1 document
    low_freq_cols = (X > 0).sum() < min_freq
    if low_freq_cols.any():
        print(f"Removing {low_freq_cols.sum()} low-frequency features (< {min_freq} documents)")
        X = X.loc[:, ~low_freq_cols]
    
    print(f"Final feature shape after filtering: {X.shape}")
    
    # Step 6: Optional feature scaling (helps with some algorithms)
    print("Applying feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame for consistency
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Step 7: Train-test split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training label distribution: {y_train.value_counts().to_dict()}")
    print(f"Test label distribution: {y_test.value_counts().to_dict()}")
    
    # Save the scaler for future use
    import joblib
    import os
    scaler_path = "outputs/models/scaler.pkl"
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    print("Data preprocessing completed successfully!")
    
    return X_train.values, X_test.values, y_train.values, y_test.values
