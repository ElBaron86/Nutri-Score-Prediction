'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-22 18:11:59
 # @ Modified by: Jaures Ememaga
 # @ Modified time: 2023-11-22 18:12:05
 # @ Description: Ce script rassemble toutes les fonctionnalites pour construire des modeles randomforest consommateur et producteur
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



#### Fonction pour la recherche d'hyperparametres ####

def hyper_params_search(X_train : pd.DataFrame, y_train : pd.Series, n_cross_val : int = 5, n_iter : int = 30, n_estimators_min : int = 20,
                        n_estimators_max : int = 300, max_depth_min : int = 1, max_depth_ax : int = 20) -> RandomForestClassifier: 
                        
    """La fonction `hyper_params_search` effectue une recherche d'hyperparametres pour un classificateur RandomForest.
    
    Args:
        X_train (pd.DataFrame) : Variables explicatives d'entrainement.
        y_train (pd.Series) : Labels d'entrainement.
        n_cross_val (int): Nombre de divisions des donnees dans la validation croisee. Par defaut 5
        n_iter (int): Nombre d'iterations du processus de recherche de parametres = nombre de combinaisons d'hyperparametres qui vont être testees.
        Plus ce nombre est grand plus le processus prendra du temps, donc attention au nombre fixe. Par defaut 30
        n_estimators_min (int): Nombre minimum d'arbres dans l'intervalle de recherche. Par defaut 20.
        n_estimators_max (int): Nombre maximum d'arbres dans l'intervalle de recherche. Par defaut 300.
        max_depth_min (int): Nombre minimum des niveaux (profondeur) des arbres dans l'intervalle de recherche. Par defaut 1.
        max_depth_max (int): Nombre maximum des niveaux (profondeur) des arbres dans l'intervalle de recherche. Par defaut 20.
        
    Returns:
        best_rf (RandomForestClassifier): Meilleur modele obtenu
    """
    time_start = time.time()
    
    print("#### Recherche d'hyper parametres en cours... ####")
    print()
    
    # Dictionnaire avec les distributions des valeurs d'hyperparametres a chercher
    param_dist = {'n_estimators': randint(n_estimators_min, n_estimators_max),
              'max_depth': randint(max_depth_min, max_depth_ax)}
    
    rf = RandomForestClassifier()

    # Recherche aleatoire des hyperparametres
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=n_iter, 
                                    cv=n_cross_val)

    # Ajustement du modele aux donnees
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_ # ici le meilleur model est celui ayant obtenu un meilleure precision moyenne sur l'ensemble des splits de la cross validation
    
    time_end = time.time()
    print(f"Temps d'execution : {time_end-time_start} secondes")
    print()
    print('Meilleurs hyperparametres:',  rand_search.best_params_)
    return best_rf



#### Fonction pour faire des prediction et afficher des performances detaillees au test ####

def make_prediction(model, X_test : pd.DataFrame, y_test : pd.Series, plot_conf_mat : bool = False) -> np.array:
    """Fonction pour faire des predictions avec un modele de random frorest.

    Args:
        model : Modele a utiliser pour la prediction
        X_test (pd.): Variables explicatives de test
        y_test (pd.Series) : labels des donnees de test
        plot_conf_mat (bool, optional): Afficher la matrice de confusion au format heatmap.

    Returns:
        y_pred (np.array) : Matrice contenant les classes predites par model
    """
    
    # Faire une prediction
    y_pred = model.predict(X_test)
    
    # Prendre la precision
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Precision globale : {accuracy}")
    print()
    print(classification_report(y_test, y_pred)) # permet d'avoir des informations plus completes sur la prediction effectuee
    
    if plot_conf_mat:
        # Affichage de la matrice de confusion au format heatmap
        cm = confusion_matrix(y_test, y_pred)
        labels = ['E', 'D', 'C', 'B', 'A']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matrice de Confusion')
        plt.xlabel('Valeurs Predites')
        plt.ylabel('Valeurs Reelles')
        plt.show()
    return y_pred


#### Recherche de modele optimal avec reduction de variables (backward_selection) et recherche d'hyperparametres #### 

def hyper_params_search_with_feature_elimination(X_train, y_train, X_test, y_test, actual_model : RandomForestClassifier,
                                                n_cross_val=5, n_iter=30, 
                                                n_estimators_min=20, n_estimators_max=300, 
                                                max_depth_min=1, max_depth_max=20) -> Tuple[RandomForestClassifier, List[str]]:
    
    """Fonction pour faire une selectioun de modeles en enlevant progressivement les variables avec la plus faible importance.
    
    Args:
        X_train (pd.DataFrame): Variables explicatives de train.
        y_train (pd.Series): Labels de train.
        X_test (pd.DataFrame): Variables explicatives de test.
        y_test (pd.Series): Labels de test.
        actual_model (RandomForestClassifier): Meilleur modèle actuel
        n_cross_val (int): Nombre de splits a fairte dans la validation croisee.
        n_iter (int): Nombre d'iterations du processus de recherche de parametres = nombre de combinaisons d'hyperparametres qui vont être testees.
        n_estimators_min (int): Nombre minimum d'arbres dans l'intervalle de recherche. Par defaut 20.
        n_estimators_max (int): Nombre maximum d'arbres dans l'intervalle de recherche. Par defaut 300.
        max_depth_min (int): Nombre minimum des niveaux (profondeur) des arbres dans l'intervalle de recherche. Par defaut 1.
        max_depth_max (int): Nombre maximum des niveaux (profondeur) des arbres dans l'intervalle de recherche. Par defaut 20.
        
    Returns:
        Tuple[RandomForestClassifier, List[str]]: Meilleur modele obtenu et liste des variables importantes
    """
    time_start = time.time()
    
    current_X_train = X_train.copy()
    current_X_test = X_test.copy()
    
    # Initialiser le meilleur modele avec le modele entraine que vous avez deja
    best_model = actual_model
    
    # Initialiosation des variables pour les iterations
    best_accuracy = accuracy_score(y_test, best_model.predict(current_X_test)) # meilleure precision connue au depart(celle de precedent best_rf)
    evolution_accuracy = [best_accuracy] # liste des accuracy
    removed_features = [] # liste qui va contenir les variables retirees
    best_features = list(current_X_train.columns) # liste des meilleures variables (actuellement celles du meilleur model connu)
    
    print("#### Recherche de modele optimal par reduction de variables + recherche d'hyper parametres en cours... ####")
    print()
    while current_X_train.shape[1] > 2:
        feature_importances = best_model.feature_importances_ # on recupere les importances des variables du meilleur modele
        weakest_feature_index = np.argmin(feature_importances) # on repere celle avec la plus faible importance pour la retirer
        
        if weakest_feature_index < current_X_train.shape[1]:
            removed_feature = current_X_train.columns[weakest_feature_index] # on retire la variables la moins importante
            
            print(f"Variable retiree : {removed_feature}")
            
            current_X_train = current_X_train.drop(removed_feature, axis=1) # maj des variables en retirant la moins importante
            current_X_test = current_X_test.drop(removed_feature, axis=1) # maj aussi sur les donnees de test
            
            # On cree un nouveau modele avec les meilleures caracteristiques et on l'entraine sur le train maj
            new_best_model = hyper_params_search(current_X_train, y_train, n_cross_val, n_iter, 
                                                 n_estimators_min, n_estimators_max, 
                                                 max_depth_min, max_depth_max)
            new_best_model.fit(current_X_train, y_train)
            
            new_accuracy = accuracy_score(y_test, new_best_model.predict(current_X_test)) # on recupere sa precision
            
            # Mettre a jour le meilleur modele et les meilleures caracteristiques si seulement la precision est meilleure
            if new_accuracy > best_accuracy:
                best_model = new_best_model
                best_accuracy = new_accuracy
                best_features = list(current_X_train.columns)
            
            evolution_accuracy.append(new_accuracy)
            removed_features.append(removed_feature)
        else:
            print("Erreur: weakest_feature_index est hors des limites.") # au cas ou on pourrait retirer plus de var que prevu on stoppe
            break
    
    time_end = time.time()
    print(f"Temps d'execution : {time_end - time_start} seconds")
    print('Meilleurs hyperparametres :', best_model.get_params())
    print()
    
    plt.plot(range(1, len(evolution_accuracy) + 1), evolution_accuracy, marker='o')
    for i, txt in enumerate(removed_features):
        plt.annotate(txt, (i + 1, evolution_accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center', rotation=45, color='red')
    plt.xlabel('Nombre de variables retirees')
    plt.ylabel('Precision')
    plt.show()
    
    return best_model, best_features

