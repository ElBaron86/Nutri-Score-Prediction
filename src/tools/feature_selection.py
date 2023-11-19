'''
 # @ Author: KPADONOU Carlos
 # @ Create Time: 2023-11-15 18:14:14
 # @ Modified by: KPADONOU Carlos
 # @ Modified time: 2023-11-19 17:50:36
 # @ Description: Ce code regroupe des fonctions pour effectuer la sélection de variable dans le cadre de
 la regression logistique ordinale : Backward
 '''

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
#from sklearn.preprocessing import RobustScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
import itertools
import statsmodels.api as sm
from tqdm.auto import tqdm
import time
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score




# Identifier la variable à retirer dans le cadre d'un Backward-critère p-value


def variable_selection_ordered_model_aic(X: pd.DataFrame, y: pd.Series):
    
    """ Cette fonction nous permet d'identifier les variables permettant de minimiser
    le critère AIC en utilisant une itération de type  backward.

    Args:
        X (pd.DataFrame): un dataframe constitué des variables explicatives
        normalisées
        Y (pd.Series) : un vecteur numpy ou un dataframe à une colonne contenant des variables ordonnées
        Ex: valeur numérique ou de type categoriel ordonné 

    Returns:
        Tuple:
            best_variables(list): Cette liste contient l'ensemble des variables contenus dans le modèle 
            qui minimise le critère AIC
            aic_values(list): Cette liste contient l'ensemble des critères obtenu à chaque itération 
            unused_variables(list): Cette liste contient l'ensemble des variables qui ont été retirés du modèles de base 
            pour obtenir le modèle optimal
    """
    # Récupérer le nombre de variables
    num_variables = X.shape[1]
    
    # Créer une liste de toutes les variables
    all_variables = list(X.columns)
    
    # Copier la liste des variables restantes et des meilleures variables
    remaining_variables = all_variables.copy()
    best_variables = all_variables.copy()
    
    # Initialiser une liste pour stocker les valeurs AIC
    aic_values = []
    
    # Initialiser des listes pour stocker les variables non utilisées et supprimées
    unused_variables = []
    removed_variable_names = []

    # Mesurer le temps d'exécution
    time_start = time.time()

    # Calculer le premier AIC avec toutes les variables
    min_aic = OrderedModel(y, X, distr='probit').fit(method='bfgs', maxiter=1000).aic
    aic_values.append(min_aic)
    
    # Utiliser tqdm pour une boucle plus informative
    for a in range(num_variables - 1):
        best_combination = None

        # Essayer toutes les combinaisons possibles de variables
        for subset in itertools.combinations(remaining_variables, len(remaining_variables) - 1):
            # Ajuster le modèle avec la sous-ensemble actuel
            model = OrderedModel(y,
                                 X[list(subset)],
                                 distr='probit').fit(method='bfgs', maxiter=1000)
            current_aic = model.aic

            # Vérifier si le nouvel AIC est plus bas que le minimum actuel
            if current_aic < min_aic:
                min_aic = current_aic
                best_combination = subset

        # Vérifier s'il y a une meilleure combinaison
        if best_combination is not None:
            # Identifier la variable non utilisée
            unused_var = set(best_variables) - set(best_combination)
            unused_variable_name = unused_var.pop()
            
            # Ajouter la variable non utilisée aux listes
            unused_variables.append(unused_variable_name)
            removed_variable_names.append(unused_variable_name)

            # Mettre à jour les listes de variables restantes et meilleures variables
            remaining_variables = list(best_combination)
            best_variables.remove(unused_variable_name)
            
            # Ajouter le nouveau AIC à la liste
            aic_values.append(min_aic)
        else:
            # Sortir de la boucle s'il n'y a pas d'amélioration possible
            break
            
    # Mesurer le temps d'exécution
    time_end = time.time()

    print(f"Temps d'exécution : {time_end - time_start}")

    # Créer un graphique AIC
    plt.plot(range(len(aic_values)), aic_values, marker='o', linestyle='-', color='b')
    plt.title("Evolution de l'AIC à chaque itération")
    plt.xlabel("Itération")
    plt.ylabel("AIC")

    # Ajout des annotations sur le graphique
    for i, var_name in enumerate(removed_variable_names):
        plt.annotate(var_name, (i+1, aic_values[i]), textcoords="offset points", xytext=(0, 1), ha='center')

    # Afficher le graphique
    plt.show()
    
    # Sauvegarder le graphique en tant qu'image PNG
    plt.savefig("aic_graphe.png", dpi=300)

    return best_variables, aic_values, unused_variables


# Sélection backward selon BIC

def variable_selection_ordered_model_bic(X: pd.DataFrame, y: pd.Series):
    """
    Sélection de variables basée sur le critère BIC (Bayesian Information Criterion).

    Parameters:
    - X (pd.DataFrame): Le dataframe contenant les variables indépendantes.
    - y (pd.Series): La série contenant la variable dépendante. Elle doit être ordonnée.

    Returns:
    - Tuple: Un tuple contenant les meilleures variables, les valeurs BIC à chaque itération, et les variables non utilisées.
    """

    # Récupérer le nombre de variables
    num_variables = X.shape[1]
    
    # Créer une liste de toutes les variables
    all_variables = list(X.columns)
    
    # Copier la liste des variables restantes et des meilleures variables
    remaining_variables = all_variables.copy()
    best_variables = all_variables.copy()
    
    # Initialiser une liste pour stocker les valeurs BIC
    bic_values = []
    
    # Initialiser des listes pour stocker les variables non utilisées et supprimées
    unused_variables = []
    removed_variable_names = []

    # Mesurer le temps d'exécution
    time_start = time.time()

    # Calculer le premier BIC avec toutes les variables
    min_bic = OrderedModel(y, X, distr='probit').fit(method='bfgs', maxiter=1000).bic
    bic_values.append(min_bic)
    
    # Utiliser tqdm pour une boucle plus informative
    for _ in tqdm(range(num_variables - 1)):
        best_combination = None

        # Essayer toutes les combinaisons possibles de variables
        for subset in itertools.combinations(remaining_variables, len(remaining_variables) - 1):
            # Ajuster le modèle avec la sous-ensemble actuel
            model = OrderedModel(y,
                                 X[list(subset)],
                                 distr='probit').fit(method='bfgs', maxiter=1000)
            current_bic = model.bic

            # Vérifier si le nouveau BIC est plus bas que le minimum actuel
            if current_bic < min_bic:
                min_bic = current_bic
                best_combination = subset

        # Vérifier s'il y a une meilleure combinaison
        if best_combination is not None:
            # Identifier la variable non utilisée
            unused_var = set(best_variables) - set(best_combination)
            unused_variable_name = unused_var.pop()
            
            # Ajouter la variable non utilisée aux listes
            unused_variables.append(unused_variable_name)
            removed_variable_names.append(unused_variable_name)

            # Mettre à jour les listes de variables restantes et meilleures variables
            remaining_variables = list(best_combination)
            best_variables.remove(unused_variable_name)
            
            # Ajouter le nouveau BIC à la liste
            bic_values.append(min_bic)
        else:
            # Sortir de la boucle s'il n'y a pas d'amélioration possible
            break
            
    # Mesurer le temps d'exécution
    time_end = time.time()

    print(f"Temps d'exécution : {time_end - time_start}")

    # Créer un graphique BIC
    plt.plot(range(len(bic_values)), bic_values, marker='o', linestyle='-', color='b')
    plt.title("Evolution de l'BIC à chaque itération")
    plt.xlabel("Itération")
    plt.ylabel("BIC")

    # Ajout des annotations sur le graphique
    for i, var_name in enumerate(removed_variable_names):
        plt.annotate(var_name, (i+1, bic_values[i]), textcoords="offset points", xytext=(0, 2), ha='center')

    # Afficher le graphique
    plt.show()
    
    # Sauvegarder le graphique en tant qu'image PNG
    plt.savefig("C:/Users/edson/OneDrive - URCA/Documents/M2 SEP/Projet_digital_pers/bic_graphe.png", dpi=300)

    return best_variables, bic_values, unused_variables

# Adaptation de la cross validation pour le calcule de l'erreur théorique des modèles obtenus après backward AIC,BIC


def cross_validation_(k: int, X: pd.DataFrame, y: pd.Series):
    """
    Effectue une validation croisée k-fold et retourne la moyenne des erreurs de classification.

    Parameters:
    - k (int): Le nombre de folds pour la validation croisée.
    - X (pd.DataFrame): Le dataframe contenant les variables indépendantes.
    - y (pd.Series): La série contenant la variable dépendante.

    Returns:
    - float: La moyenne des erreurs de classification sur tous les folds.
    """

    # Initialisation de la liste pour stocker les erreurs moyennes par split de la validation croisée
    mse_per_split = []

    # Configuration de la validation croisée avec k splits
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    # Boucle à travers chaque split de la validation croisée
    for i, (train_idx, valid_idx) in enumerate(cv.split(X)):
        X_fold_train, y_fold_train = X.iloc[train_idx], y.iloc[train_idx]
        X_fold_valid, y_fold_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # Ajuster le modèle aux données d'entraînement du fold
        ordinal_model_fold = OrderedModel(y_fold_train, X_fold_train, distr='probit').fit(method='bfgs', maxiter=1000)

        # Prédire les probabilités pour le fold de validation
        y_prob_valid = ordinal_model_fold.predict(X_fold_valid)

        # Convertir les probabilités en classes prédites
        y_pred_valid = np.argmax(y_prob_valid.values, axis=1)

        # Calculer l'erreur moyenne pour le fold de validation
        mse = 1 - accuracy_score(y_fold_valid, y_pred_valid)

        # Stocker l'erreur dans la liste
        mse_per_split.append(mse)

    # Retourner la moyenne des erreurs de classification sur tous les folds
    return np.mean(mse_per_split)