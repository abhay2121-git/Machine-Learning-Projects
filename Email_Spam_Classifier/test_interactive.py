"""
Test script to demonstrate the interactive email classification in main.py
"""

import subprocess
import sys

def test_interactive_mode():
    """Test the interactive mode with sample email text."""
    print("Testing Interactive Email Classification")
    print("=" * 50)
    
    # Sample email text to test
    test_email = """CONGRATULATIONS! You have won a FREE vacation to Hawaii!
Click here to claim your prize: www.free-vacation.com
Limited time offer! Don't miss this amazing opportunity!
Money back guaranteed! Winner winner chicken dinner!"""
    
    print("Sample email to classify:")
    print(test_email)
    print()
    
    # Simulate user input for interactive mode
    input_data = "2\n" + test_email + "\n\n1\n"
    
    try:
        # Run main.py with simulated input
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        stdout, stderr = process.communicate(input=input_data)
        
        print("Output:")
        print(stdout)
        
        if stderr:
            print("Errors:")
            print(stderr)
            
    except Exception as e:
        print(f"Error running interactive mode: {e}")

if __name__ == "__main__":
    test_interactive_mode()
