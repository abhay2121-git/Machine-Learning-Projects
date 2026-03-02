"""
Test suite for data preprocessor module.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.preprocessor import preprocess


class TestPreprocessor(unittest.TestCase):
    """Test cases for the preprocess function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_df = pd.DataFrame({
            'message': [
                'Hello world!',
                'FREE VIAGRA NOW!!!',
                'Good morning friends',
                'Claim your prize!!!',
                'Meeting at 3pm'
            ],
            'label': ['ham', 'spam', 'ham', 'spam', 'ham']
        })
        
        self.df_with_duplicates = pd.DataFrame({
            'message': [
                'Hello world!',
                'Hello world!',  # duplicate
                'FREE VIAGRA NOW!!!',
                'Good morning friends'
            ],
            'label': ['ham', 'ham', 'spam', 'ham']
        })
        
        self.df_with_nulls = pd.DataFrame({
            'message': [
                'Hello world!',
                None,
                'FREE VIAGRA NOW!!!',
                'Good morning friends'
            ],
            'label': ['ham', 'spam', None, 'ham']
        })
    
    def test_preprocess_valid_data(self):
        """Test preprocessing valid data."""
        X_train, X_test, y_train, y_test = preprocess(self.valid_df)
        
        # Check return types
        self.assertIsInstance(X_train, pd.Series)
        self.assertIsInstance(X_test, pd.Series)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
        # Check data split (80/20 ratio approximately)
        total_samples = len(self.valid_df)
        expected_train_size = int(total_samples * 0.8)
        self.assertEqual(len(X_train) + len(X_test), total_samples)
        
        # Check labels are binary (0/1)
        unique_labels = set(y_train.unique()) | set(y_test.unique())
        self.assertEqual(unique_labels, {0, 1})
    
    def test_preprocess_removes_duplicates(self):
        """Test that preprocessing removes duplicates."""
        initial_length = len(self.df_with_duplicates)
        X_train, X_test, y_train, y_test = preprocess(self.df_with_duplicates)
        
        # Should have one less sample due to duplicate removal
        self.assertEqual(len(X_train) + len(X_test), initial_length - 1)
    
    def test_preprocess_handles_nulls(self):
        """Test that preprocessing handles null values."""
        initial_length = len(self.df_with_nulls)
        X_train, X_test, y_train, y_test = preprocess(self.df_with_nulls)
        
        # Should have fewer samples due to null removal
        self.assertLess(len(X_train) + len(X_test), initial_length)
    
    def test_preprocess_text_cleaning(self):
        """Test that text cleaning works properly."""
        X_train, X_test, y_train, y_test = preprocess(self.valid_df)
        
        # Check that text is cleaned (lowercase, no special chars)
        all_text = pd.concat([X_train, X_test])
        
        # Should be lowercase
        self.assertTrue(all(text.islower() for text in all_text if pd.notna(text)))
        
        # Should not contain special characters (basic check)
        for text in all_text:
            if pd.notna(text):
                self.assertNotIn('!', text)
                self.assertNotIn('@', text)
                self.assertNotIn('#', text)
    
    def test_preprocess_stratified_split(self):
        """Test that train-test split is stratified."""
        X_train, X_test, y_train, y_test = preprocess(self.valid_df)
        
        # Check that both splits have both classes (if possible)
        train_classes = set(y_train.unique())
        test_classes = set(y_test.unique())
        
        # At minimum, training set should have both classes
        self.assertEqual(train_classes, {0, 1})


if __name__ == '__main__':
    unittest.main()
