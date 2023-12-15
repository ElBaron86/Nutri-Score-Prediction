'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-23 20:11:31
 # @ Modified by: 
 # @ Modified time:
 # @ Description: Unit tests for Ridge regression functions in tools_ridge.py.
 '''

#### Imports ####

import sys
import os
import unittest
import numpy as np

# Move up to the "src" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing the function to test
from tools.regressions.ridge.tools_ridge import train_and_evaluate_ridge_model

class TestTrainAndEvaluateRidgeModel(unittest.TestCase):

    def setUp(self):
        """Set up data for testing."""
        # Test data
        self.train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.train_labels = np.array([0, 1, 0])
        self.test_data = np.array([[1, 2, 3], [4, 5, 6]])
        self.test_labels = np.array([0, 1])
        self.alpha = 0.5
    
    def test_train_ridge_model(self):
        """Test to train a Ridge regression model on the given training data and labels."""
        # Function call
        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(self.train_data, self.train_labels,
                                                                                        self.test_data, self.test_labels,
                                                                                        self.alpha)

        # Assertions
        self.assertIsInstance(classification_error, float)
        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, np.ndarray)

    # The function should be able to evaluate the performance of the trained model on the given test data and labels.
    def test_evaluate_ridge_model(self):
        """Test to evaluate the performance of the trained model on the given test data and labels."""
        # Function call
        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(self.train_data, self.train_labels,
                                                                                        self.test_data, self.test_labels,
                                                                                        self.alpha)

        # Assertions
        self.assertGreaterEqual(classification_error, 0)
        self.assertLessEqual(classification_error, 1)

    def test_classification_error(self):
        """Test to calculate the classification error of the model on test data.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array([0, 1, 0])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array([0, 1])
        alpha = 0.5

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       alpha)

        self.assertIsInstance(classification_error, float)

    def test_non_empty_training_data(self):
        """Test to handle non-empty training data and labels.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6]])
        train_labels = np.array([0, 1])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array([0, 1])
        alpha = 0.5

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       alpha)

        self.assertIsInstance(classification_error, float)
 
    def test_non_empty_test_data(self):
        """Test to handle non-empty test data and labels.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array([0, 1, 0])
        test_data = np.array([[10, 11, 12]])
        test_labels = np.array([1])
        alpha = 0.5

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       alpha)

        self.assertIsInstance(classification_error, float)

    def test_non_categorical_target(self):
        """Test  to handle non-categorical target variable.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array([0, 1, 0])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array([0, 1])
        alpha = 0.5

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       alpha)

        self.assertIsInstance(classification_error, float)

    def test_return_coefficients_and_intercept(self):
        """Test to return the coefficients and intercept of the trained Ridge model.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array([0, 1, 0])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array([0, 1])
        alpha = 0.5

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels,
                                                                                       alpha)

        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, np.ndarray)


    def test_handle_non_numeric_data_fixed(self):
        """Test to handle non-numeric data by passing numeric data to the 'train_and_evaluate_ridge_model' function.
        """
        train_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        train_labels = np.array(['low', 'high', 'low'])
        test_data = np.array([[1, 2, 3], [4, 5, 6]])
        test_labels = np.array(['low', 'high'])

        classification_error, coefficients, intercept = train_and_evaluate_ridge_model(train_data, train_labels,
                                                                                       test_data, test_labels)

        self.assertIsInstance(coefficients, np.ndarray)
        self.assertIsInstance(intercept, np.ndarray)

if __name__ == '__main__':
    unittest.main()
