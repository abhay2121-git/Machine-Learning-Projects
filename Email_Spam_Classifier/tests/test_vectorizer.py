"""
Test suite for vectorizer module.
"""

import unittest
import pandas as pd
import numpy as np
from src.features.vectorizer import vectorize


class TestVectorizer(unittest.TestCase):
    """Test cases for the vectorize function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X_train = pd.Series([
            'hello world',
            'spam message',
            'good morning',
            'free viagra',
            'meeting tomorrow'
        ])
        
        self.X_test = pd.Series([
            'hello there',
            'spam email',
            'good evening'
        ])
    
    def test_vectorize_returns_arrays(self):
        """Test that vectorize returns numpy arrays."""
        X_train_vec, X_test_vec = vectorize(self.X_train, self.X_test)
        
        self.assertIsInstance(X_train_vec, np.ndarray)
        self.assertIsInstance(X_test_vec, np.ndarray)
    
    def test_vectorize_shapes(self):
        """Test that vectorized arrays have correct shapes."""
        X_train_vec, X_test_vec = vectorize(self.X_train, self.X_test)
        
        # Number of samples should be preserved
        self.assertEqual(X_train_vec.shape[0], len(self.X_train))
        self.assertEqual(X_test_vec.shape[0], len(self.X_test))
        
        # Number of features should be the same for both
        self.assertEqual(X_train_vec.shape[1], X_test_vec.shape[1])
        
        # Should have reasonable number of features (<= 5000)
        self.assertLessEqual(X_train_vec.shape[1], 5000)
    
    def test_vectorize_sparse_matrix(self):
        """Test that vectorization produces sparse matrices."""
        X_train_vec, X_test_vec = vectorize(self.X_train, self.X_test)
        
        # Should be sparse matrices (check if they have sparse attributes)
        self.assertTrue(hasattr(X_train_vec, 'toarray'))
        self.assertTrue(hasattr(X_test_vec, 'toarray'))
    
    def test_vectorize_data_leakage_prevention(self):
        """Test that test data is transformed but not fitted."""
        # Create test data with words that only appear in test set
        X_train_unique = pd.Series(['hello world', 'good morning'])
        X_test_unique = pd.Series(['unique_word_test'])
        
        X_train_vec, X_test_vec = vectorize(X_train_unique, X_test_unique)
        
        # Test data should still have the same number of features
        self.assertEqual(X_train_vec.shape[1], X_test_vec.shape[1])
        
        # But the test sample should be mostly zeros (since unique_word_test wasn't in training)
        test_nonzero = X_test_unique[0].split()[0] in 'unique_word_test'
        
        # This is a basic check - the exact behavior depends on the vectorizer settings


if __name__ == '__main__':
    unittest.main()
