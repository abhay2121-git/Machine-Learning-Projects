"""
Naive Bayes classifier module for email spam classification.
Handles training and prediction using Multinomial Naive Bayes.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
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
        Since MultinomialNB requires non-negative values, we'll use the original
        unscaled data for this model.
    """
    print("Training Naive Bayes classifier...")
    
    # Check for negative values and handle them
    if np.any(X_train < 0):
        print("Warning: Negative values found in training data for Naive Bayes")
        print("Converting to absolute values for MultinomialNB compatibility...")
        X_train_nb = np.abs(X_train)
        X_test_nb = np.abs(X_test)
    else:
        X_train_nb = X_train
        X_test_nb = X_test
    
    # Initialize and train the model
    model = MultinomialNB(alpha=1.0)  # Laplace smoothing
    
    print(f"Model parameters:")
    print(f"  - Alpha (smoothing): {model.alpha}")
    print(f"  - Training data shape: {X_train_nb.shape}")
    print(f"  - Training labels shape: {y_train.shape}")
    
    # Train the model
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
    
    # Save the model
    model_path = "outputs/models/naive_bayes_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Warning: Could not save model - {e}")
    
    return model, y_pred
