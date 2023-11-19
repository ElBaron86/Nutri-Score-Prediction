'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-15 13:14:14
 # @ Modified by: Jaures Ememaga
 # @ Modified time: 2023-11-15 13:16:24
 # @ Description: Ce code regroupe des fonctions pour faire une validation croisée afin d'obtenir les merilleurs parametres d'un modele de régression
    logistique ordinale
 '''

# Importation des modules

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import time
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
import pickle
import mplcyberpunk
plt.style.use('cyberpunk')


# Chargement des ensembles de données d'entraînement (train.csv) et de test (test.csv)
train = pd.read_csv("src/data/train.csv")
test = pd.read_csv("src/data/test.csv")

# Conversion des catégories en ordinales
score_type = CategoricalDtype(categories=['E', 'D', 'C', 'B', 'A'], ordered=True)
train['score']=train['score'].astype(score_type)

X_train = train.drop('score', axis=1)
y_train = train['score']


import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_data(data : pd.DataFrame, scaler_path: str = 'src/tools/scaler.pkl') -> pd.DataFrame:
    """
    La fonction `scale_data` prend un DataFrame pandas et un chemin vers un fichier pickle de scaler,
    charge le scaler à partir du fichier et transforme les données à l'aide du scaler.
    
    Args:
      data (pd.DataFrame): Le paramètre `data` est un DataFrame pandas qui contient les données que vous
    souhaitez mettre à l'échelle. Il doit contenir des valeurs numériques que vous souhaitez transformer
    à l'aide du scaler.
      scaler_path (str): Le paramètre `scaler_path` est une chaîne qui représente le chemin du fichier
    pickle qui contient l'objet scaler. Cet objet scaler est utilisé pour transformer les données.
    Par défaut src/tools/scaler.pkl
    
    Returns:
      les données transformées en tant que DataFrame pandas.
    """
    
    # Charger le scaler depuis le fichier pickle
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)

    # Transformer les données avec le scaler
    transformed_data = pd.DataFrame(scaler.transform(data), columns=data.columns)
    return transformed_data

X_train_scaled = scale_data(X_train)

# modèle initial
#ordinal_base_model = OrderedModel(y_train,
#                            X_train,
#                            distr='probit')
# entrainement du modèle initial avec paramètres de base
#ordinal_base_model_trained = ordinal_base_model.fit(method='bfgs', maxiter=1000)


def cross_validation_selection(X_train: pd.DataFrame, y_train : pd.Series, k: int, plot_results : bool = False, save_path : str = "") -> Tuple[OrderedModel, List[float]]:
    """Effectue une validation croisée k fois pour sélectionner les meilleurs paramètres d'un modèle de régression logistique ordinale.

    Args:
        X_train (pd.DataFrame): Données contenant les variables indépendantes. Ces données doivent au prealable etre normalisées avec un StandardScaler
        y_train (pd.Series): Données contenant la variable dépendante déjà ordonnée
        k (int): Nombre de splits (divisions) à effectuer sur les données.
        plot_results (bool): Afficher la courbe d'évolution de l'erreur en fonction des splits. Par défaut false.
        save_path (str): Chemin du fichier pickle pour enregistrerle modele optimal.

    Returns:
        Tuple[OrderedModel, List[float]]: Meilleur modèle sélectionné et performances par split.
    """
    time_start = time.time()
    
    # Initialisation de la liste pour stocker performances moyennes par split de la validation croisée
    performance_per_split = []
    # Liste pour stocker les modèles de la validation croisée
    models = []

    # Définition de la validation croisée à k plis
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    # Boucle pour effectuer la validation croisée
    for i, (train_idx, valid_idx) in enumerate(cv.split(X_train)):
        X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_fold_valid, y_fold_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        
        # Ajuster le modèle aux données d'entraînement du fold
        ordinal_model_fold = OrderedModel(y_fold_train,
                                          X_fold_train,
                                          distr='probit')
        ordinal_model_fold_trained = ordinal_model_fold.fit(method='bfgs', maxiter=2000)

        # Prédire les probabilités pour le fold de validation
        y_prob_valid = ordinal_model_fold_trained.predict(X_fold_valid)

        # Convertir les probabilités en classes prédites
        y_pred_valid = np.argmax(y_prob_valid.values, axis=1) 
        
        # Calculer la performance pour le fold de validation
        performance = mean_squared_error(y_fold_valid.cat.codes.to_numpy(), y_pred_valid)      

        # Stocker la performance dans la liste
        performance_per_split.append(performance)
        models.append(ordinal_model_fold_trained)
        
    best_model_index = np.argmin(performance_per_split)
    best_model = models[best_model_index]
        
    time_end = time.time()
    print(f"Temps d'exécution : {time_end - time_start} secondes")
    print()
    print(best_model.summary())
    
    if save_path != "":
        with open(save_path, 'wb') as file:
            pickle.dump(best_model, file)

    
    if plot_results:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        
        # Plot 1: Evolution de l'erreur
        axes[0].plot(range(1, k+1), np.array(performance_per_split), marker='o', label="MSE")
        axes[0].axvline(x=best_model_index + 1, color='green', linestyle='--', label='Meilleur modèle')
        axes[0].set_xlabel('Split')
        axes[0].set_ylabel('MES')
        axes[0].set_title("Evolution de l'erreur (mse) par split de la validation croisée")
        axes[0].legend()
        mplcyberpunk.make_lines_glow()

        # Recuperatiojn de la liste des variables utilisees
        feature_names = X_train.columns
        # Recuperation des coefficients pour attribuer l'importance des variables
        feature_importances = np.abs(best_model.params[:len(feature_names)])
        # Tri par importance des coefficients
        sorted_idx = np.argsort(feature_importances)
        
        # Dictionnaire pour traduire les variables en français
        var_dict = {
            "energy_100g": "Energie",
            "fat_100g": "Matieres grasses",
            "saturated-fat_100g": "Graisses saturees",
            "trans-fat_100g": "Graisses trans",
            "cholesterol_100g": "Cholesterol",
            "carbohydrates_100g": "Glucides",
            "sugars_100g": "Sucres",
            "fiber_100g": "Fibres",
            "proteins_100g": "Proteines",
            "salt_100g": "Sel",
            "sodium_100g": "Sodium",
            "vitamin-a_100g": "Vitamine A",
            "vitamin-c_100g": "Vitamine C",
            "calcium_100g": "Calcium",
            "iron_100g": "Fer"
        }
        
        # Liste des variables triees dans l'odre d'importance dans les meilleur modele
        sorted_features = [feature_names[sorted_idx]]
        
        # Plot 2: Diagramme à barres pour les variables par importance dans le meilleur modèle
        axes[1].barh(range(len(sorted_idx)), feature_importances[sorted_idx])
        axes[1].set_yticks(range(len(sorted_idx)))
        axes[1].set_yticklabels([var_dict[i] for i in sorted_features[0]]) # Utilisation de sorted_features[0] pour obtenir la liste des noms de variables
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Importance des variables dans le meilleur modèle')

        plt.tight_layout()
        plt.show()
    return best_model, performance_per_split

cross_validation_selection(X_train=X_train_scaled, y_train=y_train, k=12, plot_results = True)