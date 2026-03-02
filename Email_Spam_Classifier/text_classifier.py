"""
Text-based Email Spam Classifier
Processes raw email text and classifies as spam or ham.
"""

import numpy as np
import pandas as pd
import joblib
import os
import re
from typing import Tuple, Dict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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


def get_feature_vocabulary() -> list:
    """Get the vocabulary of features that the model was trained on."""
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
    
    return df.columns.tolist()


def clean_text(text: str) -> str:
    """Clean and preprocess email text."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation (keep letters, numbers, and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return ' '.join(filtered_words)


def extract_word_frequencies(text: str, vocabulary: list) -> Dict[str, int]:
    """
    Extract word frequencies from text based on model vocabulary.
    
    Args:
        text: Cleaned email text
        vocabulary: List of words the model was trained on
        
    Returns:
        Dictionary of word frequencies
    """
    words = text.split()
    word_freq = {}
    
    for word in words:
        if word in vocabulary:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    return word_freq


def preprocess_text_features(word_freq: Dict[str, int], feature_selector, variance_selector, scaler) -> np.ndarray:
    """
    Preprocess word frequencies for prediction.
    
    Args:
        word_freq: Dictionary of word frequencies
        feature_selector: Loaded feature selector
        variance_selector: Loaded variance selector  
        scaler: Loaded scaler
        
    Returns:
        Processed feature array
    """
    # Get vocabulary
    vocabulary = get_feature_vocabulary()
    
    # Create feature array with all vocabulary features
    feature_array = np.zeros(len(vocabulary))
    
    # Fill in word frequencies
    for word, count in word_freq.items():
        if word in vocabulary:
            idx = vocabulary.index(word)
            feature_array[idx] = count
    
    # Apply preprocessing
    feature_array = feature_array.reshape(1, -1)
    
    # Apply feature selection
    feature_array = feature_selector.transform(feature_array)
    feature_array = variance_selector.transform(feature_array)
    
    # Apply scaling
    feature_array = scaler.transform(feature_array)
    
    return feature_array


def classify_email_text(text: str, model_type: str = "logistic") -> Tuple[str, float]:
    """
    Classify email text as spam or ham.
    
    Args:
        text: Raw email text
        model_type: "naive" or "logistic"
        
    Returns:
        Tuple of (prediction, confidence)
    """
    # Load models
    nb_model, lr_model, scaler, feature_selector, variance_selector = load_models()
    
    # Clean text
    cleaned_text = clean_text(text)
    print(f"Cleaned text: {cleaned_text[:100]}...")
    
    # Get vocabulary
    vocabulary = get_feature_vocabulary()
    print(f"Model vocabulary size: {len(vocabulary)} words")
    
    # Extract word frequencies
    word_freq = extract_word_frequencies(cleaned_text, vocabulary)
    print(f"Found {len(word_freq)} matching words in vocabulary")
    
    if not word_freq:
        print("Warning: No words from email found in model vocabulary!")
        return "Ham", 50.0  # Default to ham if no matches
    
    # Preprocess features
    processed_features = preprocess_text_features(word_freq, feature_selector, variance_selector, scaler)
    
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


def interactive_text_classification():
    """Interactive email text classification."""
    print("=" * 80)
    print("EMAIL SPAM CLASSIFIER - TEXT-BASED CLASSIFICATION")
    print("=" * 80)
    print()
    
    while True:
        print("\nEnter your email text (or 'quit' to exit):")
        print("-" * 50)
        
        # Multi-line input
        lines = []
        while True:
            line = input()
            if line.strip() == 'quit':
                return
            if line.strip() == '' and lines:  # Empty line ends input
                break
            lines.append(line)
        
        if not lines:
            continue
        
        email_text = '\n'.join(lines)
        
        if len(email_text.strip()) < 10:
            print("Please enter a longer email text.")
            continue
        
        print(f"\nProcessing email ({len(email_text)} characters)...")
        
        try:
            # Choose model
            print("\nChoose model:")
            print("1. Logistic Regression (Recommended - 96.52% accuracy)")
            print("2. Naive Bayes (88.12% accuracy)")
            
            model_choice = input("Enter choice (1 or 2, default=1): ").strip()
            model_type = "logistic" if model_choice != "2" else "naive"
            
            # Classify
            result, confidence = classify_email_text(email_text, model_type)
            
            print(f"\n" + "=" * 50)
            print(f"CLASSIFICATION RESULT:")
            print(f"  Result: {result}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Model: {model_type.title()} Regression")
            print("=" * 50)
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your input and try again.")


def classify_from_file():
    """Classify email from a text file."""
    file_path = input("Enter file path: ").strip()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            email_text = f.read()
        
        result, confidence = classify_email_text(email_text, "logistic")
        
        print(f"\nFile Classification Result:")
        print(f"  File: {file_path}")
        print(f"  Classification: {result}")
        print(f"  Confidence: {confidence:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")


if __name__ == "__main__":
    print("Email Spam Classifier - Text-Based Classification")
    print("1. Interactive Text Input")
    print("2. Classify from File")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "2":
        classify_from_file()
    else:
        interactive_text_classification()
