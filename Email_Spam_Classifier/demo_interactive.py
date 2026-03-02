"""
Demo script to show how interactive classification works
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from text_classifier import classify_email_text

def demo_classification():
    """Demonstrate email classification with sample emails."""
    print("=" * 60)
    print("EMAIL SPAM CLASSIFIER - DEMO")
    print("=" * 60)
    
    # Sample emails
    spam_email = """
CONGRATULATIONS! You have won a FREE vacation to Hawaii!
Click here to claim your prize: www.free-vacation.com
Limited time offer! Don't miss this amazing opportunity!
Money back guaranteed! Winner winner chicken dinner!
"""
    
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
    
    print("\n1. Testing SPAM email:")
    print("-" * 30)
    print(f"Email: {spam_email.strip()[:100]}...")
    
    try:
        result, confidence = classify_email_text(spam_email, "logistic")
        print(f"Classification: {result}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n2. Testing HAM email:")
    print("-" * 30)
    print(f"Email: {ham_email.strip()[:100]}...")
    
    try:
        result, confidence = classify_email_text(ham_email, "logistic")
        print(f"Classification: {result}")
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("For interactive classification, run:")
    print("python text_classifier.py")
    print("(in an interactive terminal)")
    print("=" * 60)

if __name__ == "__main__":
    demo_classification()
