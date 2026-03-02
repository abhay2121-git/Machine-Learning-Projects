"""
Example: How to use the Text-based Email Spam Classifier
"""

from text_classifier import classify_email_text

# Example 1: Spam email text
spam_email = """
CONGRATULATIONS! You have won a FREE vacation to Hawaii!
Click here to claim your prize: www.free-vacation.com
Limited time offer! Don't miss this amazing opportunity!
Money back guaranteed! Winner winner chicken dinner!
"""

# Example 2: Ham email text
ham_email = """
Hi team,

Just wanted to remind everyone about tomorrow's project meeting.
We'll discuss the quarterly report and upcoming deadlines.
Please prepare your status updates and any questions you might have.

Meeting details:
- Time: 10:00 AM
- Location: Conference Room B
- Duration: 1 hour

Best regards,
John
"""

print("=" * 60)
print("EMAIL SPAM CLASSIFIER - TEXT-BASED EXAMPLES")
print("=" * 60)

# Test spam email
print("\nTesting SPAM email:")
print(f"Text: {spam_email[:100]}...")
result, confidence = classify_email_text(spam_email, "logistic")
print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")

# Test ham email
print("\nTesting HAM email:")
print(f"Text: {ham_email[:100]}...")
result, confidence = classify_email_text(ham_email, "logistic")
print(f"Prediction: {result} (Confidence: {confidence:.2f}%)")

print("\n" + "=" * 60)
print("To classify your own emails, run: python text_classifier.py")
print("=" * 60)
