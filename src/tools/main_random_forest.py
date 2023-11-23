'''
 # @ Author: Jaurès Ememaga
 # @ Create Time: 2023-11-23 08:25:37
 # @ Modified by: 
 # @ Modified time: 
 # @ Description: Ce script permet de lancer la construction des modeles random forest producteur ou consommateur
 '''

import sys
import os
# récupération du répertoire actuel
current_dir = os.path.dirname(os.path.abspath(__file__))

# retour au répertoire du projet 
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

#### Importations ####

# bibliothèque standard
import time
from typing import (List, Tuple)
import pickle

# Importations de bibliothèques tierces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score,
                            classification_report, ConfusionMatrixDisplay
                            )
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import mplcyberpunk
plt.style.use('cyberpunk')

# Importation des fonctions de make_random_forest
from src.tools.tools_random_forest import hyper_params_search, make_prediction, hyper_params_search_with_feature_elimination

#### Importation des bases ####

train, test = pd.read_csv(r'src\data\train.csv'), pd.read_csv(r'src\data\test.csv')

# Numerisation des scores
train['score'] = train['score'].map({"E":0, "D":1, "C":2, "B":3, "A":4})
test['score'] = test['score'].map({"E":0, "D":1, "C":2, "B":3, "A":4})

# Choix sur le modele a construire 

choice = input("Construire le modele 'Producteurs' ( Entrez P ) ou construire le modele 'Consommateurs' ( Entrez C )")
if choice.lower() == 'p':
    X_train, X_test = train.drop(labels=['score'], axis=1), test.drop(labels=['score'], axis=1)
    
elif choice.lower() == 'c':
    X_train, X_test = train[['energy_100g',
                                'fat_100g',
                                'saturated-fat_100g',
                                'carbohydrates_100g',
                                'sugars_100g',  'proteins_100g',
                                'salt_100g']], test[['energy_100g',
                                'fat_100g',
                                'saturated-fat_100g',
                                'carbohydrates_100g',
                                'sugars_100g',  'proteins_100g',
                                'salt_100g']]
                                
else :
    print("Saisie incorrecte, veuillez saisir P ou C")
    
y_train, y_test = train['score'], test['score']


# On propose un premier modele optimal avec toutes les variables de base selon le profil producteur ou consommateur

best_rf = hyper_params_search(X_train=X_train, y_train=y_train, n_iter = 30)

# On lance la recherche d'un modele optimal
king_model, best_features = hyper_params_search_with_feature_elimination(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, actual_model = best_rf, 
                                                                        n_cross_val=5, n_iter=30, 
                                                                        n_estimators_min=15, n_estimators_max=300,
                                                                        max_depth_min=1, max_depth_max=20)

        
save_choice = input("Voulez-vous sauvegarder le modèle entraîné ? (O/N)")
if save_choice.lower() == 'o':
    if choice.lower() == 'p':
        # Enregistrement du modèle (cas producteurs) optimal dans un pickle
        with open('random_forest_prod.pickle', 'wb') as content:
            pickle.dump(king_model, content)
        # Enregistrement de la liste des variables utilisées pour le modèle optimal    
        with open('features_random_forest_prod.txt', 'w') as l:
            for line in best_features:
                l.write(f"{line}\n")
                
        print("#### Modèle producteur enregistré avec succès ! ####")
    else:
        # Enregistrement du modèle (cas consommateurs) optimal dans un pickle
        with open('random_forest_conso.pickle', 'wb') as content:
            pickle.dump(king_model, content)
        # Enregistrement de la liste des variables utilisées pour le modèle optimal
        with open('features_random_forest_conso.txt', 'w') as l:
            for line in best_features:
                l.write(f"{line}\n")
                
        print("#### Modèle consommateur enregistré avec succès ! ####")

        
print("#### Programme Termine ####")

