"""
Email Spam Prediction Script
Classifies new emails as spam or ham using trained models.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Tuple


def load_models() -> Tuple[object, object, object, object, object]:
    """Load all trained models and preprocessors."""
    print("Loading trained models...")
    
    # Load models
    nb_model = joblib.load("outputs/models/naive_bayes_model.pkl")
    lr_model = joblib.load("outputs/models/logistic_regression_model.pkl")
    
    # Load preprocessors
    scaler = joblib.load("outputs/models/scaler.pkl")
    feature_selector = joblib.load("outputs/models/feature_selector.pkl")
    variance_selector = joblib.load("outputs/models/variance_selector.pkl")
    
    print("Models loaded successfully!")
    return nb_model, lr_model, scaler, feature_selector, variance_selector


def preprocess_email(email_features: dict, feature_selector, variance_selector, scaler) -> np.ndarray:
    """
    Preprocess new email features for prediction.
    
    Args:
        email_features: Dictionary of word frequencies
        feature_selector: Loaded feature selector
        variance_selector: Loaded variance selector  
        scaler: Loaded scaler
        
    Returns:
        Processed feature array
    """
    # Load original dataset and apply same filtering as in training
    df = pd.read_csv("data/dataset.csv")
    if 'Email No.' in df.columns:
        df = df.drop('Email No.', axis=1)
    if 'Prediction' in df.columns:
        df = df.drop('Prediction', axis=1)
    
    # Apply same filtering as in preprocessing
    # Remove zero columns
    zero_cols = (df == 0).all()
    if zero_cols.any():
        df = df.loc[:, ~zero_cols]
    
    # Remove low-frequency features (same threshold as training)
    min_freq = max(1, int(0.01 * len(df)))
    low_freq_cols = (df > 0).sum() < min_freq
    if low_freq_cols.any():
        df = df.loc[:, ~low_freq_cols]
    
    # Create feature array with filtered features
    feature_array = np.zeros(len(df.columns))
    
    # Fill in provided features
    for word, count in email_features.items():
        if word in df.columns:
            feature_array[df.columns.get_loc(word)] = count
    
    # Apply preprocessing
    feature_array = feature_array.reshape(1, -1)
    
    # Apply feature selection
    feature_array = feature_selector.transform(feature_array)
    feature_array = variance_selector.transform(feature_array)
    
    # Apply scaling
    feature_array = scaler.transform(feature_array)
    
    return feature_array


def predict_email(email_features: dict, model_type: str = "logistic") -> Tuple[str, float]:
    """
    Predict if an email is spam or ham.
    
    Args:
        email_features: Dictionary of word frequencies
        model_type: "naive" or "logistic"
        
    Returns:
        Tuple of (prediction, confidence)
    """
    # Load models
    nb_model, lr_model, scaler, feature_selector, variance_selector = load_models()
    
    # Preprocess email
    processed_features = preprocess_email(email_features, feature_selector, variance_selector, scaler)
    
    # Choose model
    if model_type.lower() == "naive":
        model = nb_model
        # Naive Bayes needs absolute values
        processed_features = np.abs(processed_features)
    else:
        model = lr_model
    
    # Make prediction
    prediction = model.predict(processed_features)[0]
    probabilities = model.predict_proba(processed_features)[0]
    
    # Get confidence
    confidence = max(probabilities) * 100
    
    # Convert to readable format
    result = "Spam" if prediction == 1 else "Ham"
    
    return result, confidence


def interactive_prediction():
    """Interactive email prediction."""
    print("=" * 60)
    print("EMAIL SPAM CLASSIFIER - INTERACTIVE PREDICTION")
    print("=" * 60)
    print()
    
    while True:
        print("\nEnter email features (word frequencies):")
        print("Format: word:count (e.g., 'free:5 offer:3 money:2')")
        print("Type 'quit' to exit")
        print()
        
        user_input = input("Enter features: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some features.")
            continue
        
        try:
            # Parse input
            features = {}
            for item in user_input.split():
                if ':' in item:
                    word, count = item.split(':', 1)
                    try:
                        features[word] = int(count)
                    except ValueError:
                        print(f"Invalid count for {word}: {count}")
                        continue
            
            if not features:
                print("No valid features entered.")
                continue
            
            # Choose model
            print("\nChoose model:")
            print("1. Logistic Regression (Recommended - 96.52% accuracy)")
            print("2. Naive Bayes (88.12% accuracy)")
            
            model_choice = input("Enter choice (1 or 2): ").strip()
            model_type = "logistic" if model_choice == "1" else "naive"
            
            # Make prediction
            result, confidence = predict_email(features, model_type)
            
            print(f"\nPREDICTION RESULT:")
            print(f"   Classification: {result}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Model Used: {model_type.title()} Regression")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your input format.")


def batch_prediction():
    """Predict multiple emails from a file."""
    print("Batch prediction feature coming soon!")
    print("For now, use interactive prediction.")


if __name__ == "__main__":
    print("Email Spam Classifier - Prediction Module")
    print("1. Interactive Prediction")
    print("2. Batch Prediction (CSV)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        interactive_prediction()
    elif choice == "2":
        batch_prediction()
    else:
        print("Invalid choice. Running interactive prediction...")
        interactive_prediction()
