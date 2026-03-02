"""
Feature vectorizer module for email spam classifier.
Handles pre-processed word frequency features (no text vectorization needed).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
from typing import Tuple
from sklearn.feature_selection import VarianceThreshold

def vectorize(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process pre-processed word frequency features.
    
    Args:
        X_train (np.ndarray): Training features (word frequencies)
        X_test (np.ndarray): Test features (word frequencies)
        y_train (np.ndarray): Training labels (needed for feature selection)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed training and test features
        
    Note:
        Since data is already vectorized, this function applies feature selection
        to reduce dimensionality and improve model performance.
    """
    print("Processing pre-processed word frequency features...")
    
    print(f"Input training data shape: {X_train.shape}")
    print(f"Input test data shape: {X_test.shape}")
    
    # Step 1: Feature selection to reduce dimensionality
    # Select top k features based on ANOVA F-score
    n_features = min(2000, X_train.shape[1])  # Use up to 2000 best features
    print(f"Applying feature selection to keep top {n_features} features...")
    
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Training data shape after selection: {X_train_selected.shape}")
    print(f"Test data shape after selection: {X_test_selected.shape}")
    
    # Step 2: Additional feature filtering (remove features with very low variance)
    
    
    # Remove features with variance threshold
    variance_selector = VarianceThreshold(threshold=0.01)  # Remove near-constant features
    X_train_final = variance_selector.fit_transform(X_train_selected)
    X_test_final = variance_selector.transform(X_test_selected)
    
    print(f"Training data shape after variance filtering: {X_train_final.shape}")
    print(f"Test data shape after variance filtering: {X_test_final.shape}")
    
    # Step 3: Save the feature selectors for future use
    selector_path = "outputs/models/feature_selector.pkl"
    variance_path = "outputs/models/variance_selector.pkl"
    os.makedirs(os.path.dirname(selector_path), exist_ok=True)
    
    try:
        joblib.dump(selector, selector_path)
        joblib.dump(variance_selector, variance_path)
        print(f"Feature selectors saved to: {selector_path} and {variance_path}")
    except Exception as e:
        print(f"Warning: Could not save selectors - {e}")
    
    # Step 4: Display feature selection statistics
    selected_scores = selector.scores_[selector.get_support()]
    print(f"Feature selection statistics:")
    print(f"  - Mean F-score: {np.mean(selected_scores):.4f}")
    print(f"  - Max F-score: {np.max(selected_scores):.4f}")
    print(f"  - Min F-score: {np.min(selected_scores):.4f}")
    
    print("Feature processing completed successfully!")
    
    return X_train_final, X_test_final
