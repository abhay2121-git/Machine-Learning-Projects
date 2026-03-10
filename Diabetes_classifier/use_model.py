"""
Example: How to use the saved model and scaler for predictions
"""

import joblib
import pandas as pd
import numpy as np

def load_model_and_scaler():
    """
    Load the saved model and scaler
    """
    model = joblib.load('models/diabetes_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, scaler

def predict_diabetes(patient_data):
    """
    Make predictions on new patient data
    
    Args:
        patient_data (dict): Dictionary with patient measurements
        
    Returns:
        tuple: (prediction, probability)
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Convert to DataFrame with correct column order
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 
                      'SkinThickness', 'Insulin', 'BMI', 
                      'DiabetesPedigreeFunction', 'Age']
    
    # Create DataFrame
    df = pd.DataFrame([patient_data], columns=feature_columns)
    
    # Handle zero values (same as training)
    zero_sensitive_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI']
    
    for col in zero_sensitive_cols:
        if df[col].iloc[0] == 0:
            df[col] = df[col].replace(0, np.nan)
            # Use median values from training (these would be stored)
            median_values = {
                'Glucose': 112.50,
                'BloodPressure': 72.00,
                'SkinThickness': 30.50,
                'Insulin': 129.00,
                'BMI': 31.40
            }
            df[col] = df[col].fillna(median_values[col])
    
    # Scale features using the saved scaler
    scaled_data = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0]
    
    return prediction, probability

def example_predictions():
    """
    Example usage with sample patient data
    """
    print("DIABETES PREDICTION EXAMPLES")
    print("="*50)
    
    # Example 1: High-risk patient
    patient1 = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,  # Will be filled with median
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    pred1, prob1 = predict_diabetes(patient1)
    print(f"\nPatient 1 (High Risk):")
    print(f"  Prediction: {'Diabetic' if pred1 == 1 else 'Non-Diabetic'}")
    print(f"  Confidence: {max(prob1):.2%}")
    print(f"  Probabilities: Non-Diabetic={prob1[0]:.2%}, Diabetic={prob1[1]:.2%}")
    
    # Example 2: Low-risk patient
    patient2 = {
        'Pregnancies': 1,
        'Glucose': 85,
        'BloodPressure': 66,
        'SkinThickness': 29,
        'Insulin': 0,  # Will be filled with median
        'BMI': 26.6,
        'DiabetesPedigreeFunction': 0.351,
        'Age': 31
    }
    
    pred2, prob2 = predict_diabetes(patient2)
    print(f"\nPatient 2 (Low Risk):")
    print(f"  Prediction: {'Diabetic' if pred2 == 1 else 'Non-Diabetic'}")
    print(f"  Confidence: {max(prob2):.2%}")
    print(f"  Probabilities: Non-Diabetic={prob2[0]:.2%}, Diabetic={prob2[1]:.2%}")

if __name__ == "__main__":
    example_predictions()
