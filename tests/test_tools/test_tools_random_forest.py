'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-22 21:19:18
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Testing src/tools/make_random_forest.py script functions 
 '''

import sys
import os
# Recovery of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Back to the project directory 
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Import of essential modules to tests
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.tools.tools_random_forest import hyper_params_search, make_prediction, hyper_params_search_with_feature_elimination

class TestMyFunctions(unittest.TestCase):

    def setUp(self):
        # Initialization of data for tests
        self.X_train = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 12], 'feature_2': [4, 5, 6, 9, 10, 12, 33, 44, 2, 34, 45, 37]})
        self.y_train = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        self.X_test = pd.DataFrame({'feature_1': [2, 3, 4], 'feature_2': [5, 6, 7]})
        self.y_test = pd.Series([1, 0, 1])

    def test_hyper_params_search(self):
        # Test of the hyper_params_search function

        # Make sure that no error is raised
        try:
            model = hyper_params_search(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Unknown error : {e}")

        # Make sure that the model is a RandomForestClassifier instance
        self.assertIsInstance(model, RandomForestClassifier)

    def test_make_prediction(self):
        # Make_Prediction function test

        # Make sure that no error is raised
        try:
            model = RandomForestClassifier()  # Create a model for the test
            model.fit(self.X_train, self.y_train)
            y_pred = make_prediction(model, self.X_test, self.y_test)
        except Exception as e:
            self.fail(f"Unknown error : {e}")

        # Make sure that y_pred is a numpy.array instance
        self.assertIsInstance(y_pred, np.ndarray)

    def test_hyper_params_search_with_feature_elimination(self):
        """Testing function hyper_params_search_with_feature_elimination."""
        # Create a model to use for the test
        actual_model = RandomForestClassifier().fit(self.X_train, self.y_train)

        # Make sure that no mistake is lifted
        try:
            model, features = hyper_params_search_with_feature_elimination(
                self.X_train, self.y_train, self.X_test, self.y_test, actual_model=actual_model)
            # ...
        except Exception as e:
            self.fail(f"Unknown error: {e}")

        # Make sure that the model is a RandomForestClassifier instance
        self.assertIsInstance(model, RandomForestClassifier)

        # Make sure that features is a list
        self.assertIsInstance(features, list)


    def tearDown(self):
        # Clean after tests if necessary
        pass

if __name__ == '__main__':
    unittest.main()
