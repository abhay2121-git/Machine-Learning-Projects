"""
Example: How to use the Email Spam Classifier for prediction
"""

from predict import predict_email

# Example 1: Spam email features
spam_features = {
    'free': 5,
    'offer': 3,
    'money': 2,
    'click': 4,
    'winner': 2,
    'congratulations': 1
}

# Example 2: Ham email features
ham_features = {
    'meeting': 2,
    'project': 3,
    'deadline': 1,
    'team': 2,
    'report': 1
}

print("=" * 50)
print("EMAIL SPAM CLASSIFIER - EXAMPLE PREDICTIONS")
print("=" * 50)

# Test spam email
print("\nTesting SPAM-like email:")
print(f"Features: {spam_features}")
result, confidence = predict_email(spam_features, "logistic")
print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")

# Test ham email
print("\nTesting HAM-like email:")
print(f"Features: {ham_features}")
result, confidence = predict_email(ham_features, "logistic")
print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")

print("\n" + "=" * 50)
print("To test your own emails, run: python predict.py")
print("=" * 50)
