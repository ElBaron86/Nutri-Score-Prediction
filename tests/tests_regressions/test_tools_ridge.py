'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-23 20:11:31
 # @ Modified by: 
 # @ Modified time:
 # @ Description: Unit tests for Ridgr egression functions in tools_ridge.py.
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

class TestRidgeModel(unittest.TestCase):
    """
    A test case for the train_and_evaluate_ridge_model function.

    This test case checks various scenarios to ensure the correct behavior of the Ridge regression model.

    Methods:
        - setUp: Set up common data for tests.
        - test_train_and_evaluate_ridge_model: Test the functionality of the model on provided data.
        - test_empty_data_and_labels: Test when provided with empty data and labels.
        - test_non_numeric_data: Test when provided with non-numeric data.
        - test_non_categorical_target: Test when provided with non-categorical target.
    """
    def setUp(self):
        """
        Set up common data for tests.
        """
        self.train_data = np.random.rand(100, 5)
        self.train_labels = np.random.randint(0, 3, 100)
        self.test_data = np.random.rand(50, 5)
        self.test_labels = np.random.randint(0, 3, 50)

    def test_train_and_evaluate_ridge_model(self):
        """
        Test the functionality of the model on provided data.

        This test checks if the error is within the expected range and if coefficients and intercept are not None.
        """
        error, coefficients, intercept = train_and_evaluate_ridge_model(
            self.train_data, self.train_labels, self.test_data, self.test_labels
        )

        # Check if the error is within the expected range
        self.assertGreaterEqual(error, 0)
        self.assertLessEqual(error, 1)

        # Check if coefficients and intercept are not None
        self.assertIsNotNone(coefficients)
        self.assertIsNotNone(intercept)

    def test_empty_data_and_labels(self):
        """
        Test when provided with empty data and labels.

        This test checks if the function raises a ValueError when given empty data and labels.
        """
        with self.assertRaises(ValueError):
            train_and_evaluate_ridge_model(np.array([]), np.array([]), np.array([]), np.array([]))

    def test_non_numeric_data(self):
        """
        Test when provided with non-numeric data.

        This test checks if the function raises a ValueError when given non-numeric data.
        """
        non_numeric_data = ['a', 'b', 'c']
        with self.assertRaises(ValueError):
            train_and_evaluate_ridge_model(
                non_numeric_data, self.train_labels, self.test_data, self.test_labels
            )

    def test_non_categorical_target(self):
        """
        Test when provided with non-categorical target.

        This test checks if the function raises a ValueError when given a non-categorical target.
        """
        non_categorical_labels = np.random.rand(100)
        with self.assertRaises(ValueError):
            train_and_evaluate_ridge_model(
                self.train_data, non_categorical_labels, self.test_data, self.test_labels
            )

if __name__ == '__main__':
    unittest.main()
