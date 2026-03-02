#!/usr/bin/env python3
"""
Test script to verify main.py interactive mode works
"""

import subprocess
import sys
import time

def test_interactive_mode():
    """Test the interactive mode with a simple email."""
    print("Testing main.py interactive mode...")
    
    # Prepare the input sequence
    input_data = "2\nCONGRATULATIONS! You won a FREE vacation!\nClick here now!\n\n1\n3\n"
    
    try:
        # Run main.py with the prepared input
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="."
        )
        
        stdout, stderr = process.communicate(input=input_data, timeout=30)
        
        print("STDOUT:")
        print(stdout)
        
        if stderr:
            print("STDERR:")
            print(stderr)
            
        # Check if classification worked
        if "Classification:" in stdout or "Spam" in stdout or "Ham" in stdout:
            print("✅ Interactive mode test PASSED!")
        else:
            print("❌ Interactive mode test FAILED!")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out!")
        process.kill()
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_interactive_mode()
