"""
Model trainer module for email spam classifier.
Coordinates training of both Naive Bayes and Logistic Regression models.
"""

import numpy as np
from typing import Tuple
from .naive_bayes import train_naive_bayes
from .logistic_regression import train_logistic_regression


def train_models(X_train: np.ndarray, X_test: np.ndarray, 
                y_train: np.ndarray) -> Tuple[object, object, np.ndarray, np.ndarray]:
    """
    Train both Naive Bayes and Logistic Regression models.
    
    Args:
        X_train (np.ndarray): Vectorized training features
        X_test (np.ndarray): Vectorized test features
        y_train (np.ndarray): Training labels
        
    Returns:
        Tuple[object, object, np.ndarray, np.ndarray]: 
        Naive Bayes model, Logistic Regression model, 
        Naive Bayes predictions, Logistic Regression predictions
    """
    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    # Train Naive Bayes
    print("\n1. Training Naive Bayes Classifier...")
    nb_model, nb_predictions = train_naive_bayes(X_train, X_test, y_train)
    
    # Train Logistic Regression
    print("\n2. Training Logistic Regression Classifier...")
    lr_model, lr_predictions = train_logistic_regression(X_train, X_test, y_train)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)
    print(f"Naive Bayes predictions: {len(nb_predictions)} samples")
    print(f"Logistic Regression predictions: {len(lr_predictions)} samples")
    
    return nb_model, lr_model, nb_predictions, lr_predictions
