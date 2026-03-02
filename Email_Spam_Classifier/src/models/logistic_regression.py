"""
Logistic Regression classifier module for email spam detection.
Implements Logistic Regression with training and prediction functionality.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
from typing import Tuple


def train_logistic_regression(X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Train a Logistic Regression classifier and make predictions.
    
    Args:
        X_train (np.ndarray): Training features (word frequencies)
        X_test (np.ndarray): Test features (word frequencies)
        y_train (np.ndarray): Training labels
        
    Returns:
        Tuple[LogisticRegression, np.ndarray]: Trained model and predictions
        
    Note:
        The trained model is saved to outputs/models/logistic_regression_model.pkl
    """
    print("Training Logistic Regression classifier...")
    
    # Initialize Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        C=1.0,  # Regularization parameter
        penalty='l2'  # L2 regularization
    )
    
    print(f"Model parameters:")
    print(f"  - Max iterations: 1000")
    print(f"  - Random state: 42")
    print(f"  - Solver: lbfgs")
    print(f"  - Regularization (C): 1.0")
    print(f"  - Penalty: l2")
    print(f"  - Training data shape: {X_train.shape}")
    print(f"  - Training labels shape: {y_train.shape}")
    
    # Train the model
    lr_model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions on test set
    y_pred = lr_model.predict(X_test)
    
    # Get prediction probabilities for ROC curve
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    print(f"Predictions made on {len(y_pred)} test samples")
    
    # Save the trained model
    model_path = "outputs/models/logistic_regression_model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    try:
        joblib.dump(lr_model, model_path)
        print(f"Model saved to: {model_path}")
    except Exception as e:
        print(f"Warning: Could not save model - {e}")
    
    # Display model information
    print(f"Number of iterations completed: {lr_model.n_iter_[0]}")
    print(f"Model intercept: {lr_model.intercept_[0]:.4f}")
    
    print("Logistic Regression classifier completed successfully!")
    
    return lr_model, y_pred
