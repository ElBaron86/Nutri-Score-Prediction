'''
 # @ Author: Jaurès Ememaga
 # @ Create Time: 2023-11-22 21:19:18
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Tests des fonctions du script make_random_forest.py dans src/tools
 '''

import sys
import os
# récupération du répertoire actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# retour au répertoire du projet 
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

# Importation des modules essentiels aux tests
import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.tools.tools_random_forest import hyper_params_search, make_prediction, hyper_params_search_with_feature_elimination

class TestMyFunctions(unittest.TestCase):

    def setUp(self):
        # Initialisation des donnees pour les tests
        self.X_train = pd.DataFrame({'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 12], 'feature_2': [4, 5, 6, 9, 10, 12, 33, 44, 2, 34, 45, 37]})
        self.y_train = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1])
        self.X_test = pd.DataFrame({'feature_1': [2, 3, 4], 'feature_2': [5, 6, 7]})
        self.y_test = pd.Series([1, 0, 1])

    def test_hyper_params_search(self):
        # Test de la fonction hyper_params_search

        # S'assurer qu'aucune erreur n'est levee
        try:
            model = hyper_params_search(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Erreur non attendue : {e}")

        # S'assurer que le modele est une instance de RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)

    def test_make_prediction(self):
        # Test de la fonction make_prediction

        # S'assurer qu'aucune erreur n'est levee
        try:
            model = RandomForestClassifier()  # Créer un modèle pour le test
            model.fit(self.X_train, self.y_train)
            y_pred = make_prediction(model, self.X_test, self.y_test)
        except Exception as e:
            self.fail(f"Erreur non attendue : {e}")

        # S'assurer que y_pred est une instance de numpy.array
        self.assertIsInstance(y_pred, np.ndarray)

    def test_hyper_params_search_with_feature_elimination(self):
        """Test de la fonction hyper_params_search_with_feature_elimination."""
        # Créer un modèle à utiliser pour le test
        actual_model = RandomForestClassifier().fit(self.X_train, self.y_train)

        # S'assurer qu'aucune erreur n'est levée
        try:
            model, features = hyper_params_search_with_feature_elimination(
                self.X_train, self.y_train, self.X_test, self.y_test, actual_model=actual_model)
            # ...
        except Exception as e:
            self.fail(f"Erreur non attendue : {e}")

        # S'assurer que le modele est une instance de RandomForestClassifier
        self.assertIsInstance(model, RandomForestClassifier)

        # S'assurer que features est une liste
        self.assertIsInstance(features, list)


    def tearDown(self):
        # Nettoyer apres les tests si necessaire
        pass

if __name__ == '__main__':
    unittest.main()
