"""
Naive Bayes classifier module for email spam classification.
Handles training and prediction using Multinomial Naive Bayes.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import RobustScaler
import joblib
import os
from typing import Tuple


def train_naive_bayes(X_train: np.ndarray, X_test: np.ndarray, 
                     y_train: np.ndarray) -> Tuple[MultinomialNB, np.ndarray]:
    """
    Train a Multinomial Naive Bayes classifier for spam detection.
    
    Args:
        X_train (np.ndarray): Training features (word frequencies)
        X_test (np.ndarray): Test features (word frequencies)
        y_train (np.ndarray): Training labels
        
    Returns:
        Tuple[MultinomialNB, np.ndarray]: Trained model and predictions
        
    Note:
        Uses RobustScaler to handle negative values while preserving data characteristics.
        Then converts to non-negative for MultinomialNB compatibility.
    """
    print("Training Naive Bayes classifier...")
    
    # Use RobustScaler to handle negative values while preserving relationships
    # RobustScaler uses median and IQR, less sensitive to outliers
    scaler_nb = RobustScaler()
    X_train_scaled = scaler_nb.fit_transform(X_train)
    X_test_scaled = scaler_nb.transform(X_test)
    
    # Convert to non-negative for MultinomialNB (shift by minimum value)
    min_val = X_train_scaled.min()
    if min_val < 0:
        X_train_nb = X_train_scaled - min_val  # Shift to make non-negative
        X_test_nb = X_test_scaled - min_val
        print(f"Applied RobustScaler with shift ({min_val:.3f}) for non-negative values")
    else:
        X_train_nb = X_train_scaled
        X_test_nb = X_test_scaled
        print(f"Applied RobustScaler (no shift needed)")
    
    print(f"Training data range: [{X_train_nb.min():.3f}, {X_train_nb.max():.3f}]")
    print(f"Test data range: [{X_test_nb.min():.3f}, {X_test_nb.max():.3f}]")
    
    # Initialize and train model
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    
    print(f"Model parameters:")
    print(f"  - Alpha (smoothing): {model.alpha}")
    print(f"  - Training data shape: {X_train_nb.shape}")
    print(f"  - Training labels shape: {y_train.shape}")
    
    # Train model
    model.fit(X_train_nb, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_nb)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test_nb)
    
    print(f"Training completed successfully!")
    print(f"Predictions shape: {y_pred.shape}")
    print(f"Prediction probabilities shape: {y_pred_proba.shape}")
    
    # Display model insights
    print(f"Class priors (learned): {model.class_log_prior_}")
    print(f"Feature count shape: {model.feature_count_.shape}")
    
    # Save model and scaler for consistent preprocessing
    model_path = "outputs/models/naive_bayes_model.pkl"
    scaler_path = "outputs/models/naive_bayes_scaler.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler_nb, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
    except Exception as e:
        print(f"Warning: Could not save model - {e}")
    
    return model, y_pred


def load_naive_bayes_model():
    """
    Load trained Naive Bayes model and its scaler.
    
    Returns:
        Tuple[MultinomialNB, RobustScaler]: Model and scaler
    """
    try:
        model = joblib.load("outputs/models/naive_bayes_model.pkl")
        scaler = joblib.load("outputs/models/naive_bayes_scaler.pkl")
        print("Naive Bayes model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading Naive Bayes model: {e}")
        return None, None
