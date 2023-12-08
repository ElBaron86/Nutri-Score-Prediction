'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-05 22:10:12
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Unit tests for function in tools_data_cleaning.py.
 '''

# Imports #

import sys
import os
import unittest
import pandas as pd

# Recovery of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Back to the project directory 
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

# import function for tests
from src.tools.tools_data_cleaning import clean_data  

class TestCleanDataFunction(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'product_name': ['Product1', 'Product2', 'Product3', 'Product4', 'Product5'],
            'nutrient_100g_1': [10, 20, 30, 40, 50],
            'nutrient_100g_2': [5, 15, 25, 35, 45],
            'nutrition-score-fr_100g': [15, 10, 25, 5, 20],
            'nutrition_grade_fr': ['a', 'b', 'c', 'a', 'b']
        }
        self.sample_data = pd.DataFrame(data)

    def test_clean_data_shape(self):
        # Test if the shape of the returned DataFrames is as expected
        balanced_data, train_data, test_data = clean_data(self.sample_data)
        self.assertEqual(balanced_data.shape[0], len(self.sample_data['nutrition_grade_fr'].unique()))
        self.assertEqual(train_data.shape[0] + test_data.shape[0], balanced_data.shape[0])

    def test_clean_data_duplicates(self):
        # Test if the balanced_data DataFrame has no duplicates
        balanced_data, _, _ = clean_data(self.sample_data)
        self.assertFalse(balanced_data.duplicated().any())

    def test_clean_data_train_test_split(self):
        # Test if the train-test split is working correctly
        _, train_data, test_data = clean_data(self.sample_data)
        self.assertGreater(train_data.shape[0], 0)
        self.assertGreater(test_data.shape[0], 0)

if __name__ == '__main__':
    unittest.main()
