"""
Test suite for data loader module.
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from src.data.loader import load_data


class TestLoader(unittest.TestCase):
    """Test cases for the load_data function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.valid_csv_path = os.path.join(self.temp_dir, "valid_data.csv")
        self.empty_csv_path = os.path.join(self.temp_dir, "empty_data.csv")
        self.nonexistent_path = os.path.join(self.temp_dir, "nonexistent.csv")
        
        # Create valid test data
        valid_data = pd.DataFrame({
            'message': ['Hello world', 'Spam message', 'Good morning'],
            'label': ['ham', 'spam', 'ham']
        })
        valid_data.to_csv(self.valid_csv_path, index=False)
        
        # Create empty CSV
        empty_data = pd.DataFrame()
        empty_data.to_csv(self.empty_csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_valid_data(self):
        """Test loading a valid CSV file."""
        df = load_data(self.valid_csv_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('message', df.columns)
        self.assertIn('label', df.columns)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_data(self.nonexistent_path)
    
    def test_load_empty_file(self):
        """Test loading an empty CSV file raises ValueError."""
        with self.assertRaises(ValueError):
            load_data(self.empty_csv_path)


if __name__ == '__main__':
    unittest.main()
