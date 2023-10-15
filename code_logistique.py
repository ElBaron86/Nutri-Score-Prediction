"https://pythonhosted.org/mord/" #doc du module de régression ordinale en python

# installation du fameux module
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mord'])
print("Hello")


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mord import LogisticAT
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from typing import Tuple, Dict, List, Callable, Any




def data_preparation(path : str = None, dep_var : str = None, indep_var : List[str] = None, toy_data = None) -> Tuple[np.array, np.array]:
    """Fonction pour préparer les données afin d'extraire les données d'entrainement et de test qui serviront à préparer le modèle de régression
        On suppose que toutes les variables explicatives sont quanti, sinon je vais rajouter un traitemen spécifique
    Args:
        path (str): chemin du fichier de données
        dep_var (str): nom de la variable d'intérêt (variable dépendante) parmis les colonnes des données
        indep_var (List[str], optional): variables sélectionnées (columns of source file) pour créer le modèle. !!! Attention à ne pas inscrirele nom de la variable d'intérêt dedans Defaults to "".

    Returns:
        Tuple[X, Y]: X contient les variables sélectionnées pour la régression et Y contient les labels 
    """
    if toy_data != None: # Dans le cas où on utilise les données intégrées de python 
        
        data = toy_data
        X = data.data
        Y = data.target
 
    else : 
        data = pd.read_csv(path) # on lit tranquilement le fichier csv sinon
    
    
        # Si les variables explicatives ne sont pas précisées on prend tout le jeu de données 
        if (indep_var == None):
            data.drop_duplicates(inplace=True)
            data.dropna(inplace=True)
                
            X = data.drop(labels=[dep_var], axis=1).values
            
            # transformation de la variable d'intéret si elle est quali
            
            if data[dep_var].dtypes == object:
                classes = {quali : i for i, quali in enumerate(list(data[dep_var].unique()))}
                data[dep_var] = data[dep_var].apply(lambda x : classes[x])
                Y = data[dep_var].values
            else :
                Y = data[dep_var].values
                
            """Je vais rajouter des trucs pour des traitements spécifiques des données
            comme de la standardisation, mise à l'échelle, dummies...
            """
        else:
            
            data = data[indep_var.append(dep_var)]
            data.drop_duplicates(inplace=True)
            data.dropna(inplace=True)
            
            X = data[indep_var].values
            
            # transformation de la variable d'intéret si elle est quali
            
            if data[dep_var].dtypes == object :
                classes = {quali : i for i, quali in enumerate(list(data[dep_var].unique()))}
                data[dep_var] = data[dep_var].apply(lambda x : classes[x])
                 
            else :
                Y = data[dep_var].values           

            
    return X, Y
        

def model_training(X , Y, train_size : float = 0.75) -> LogisticAT:
    """Fonction qui va retourner le modèle de régression logistique ordinale après l'avoir entrainé sur les données et montré les performances évaluées sur les données de test

    Args:
        X (_type_): Variables explicatives
        Y (_type_): labels
        train_size (float, optional):proportion des données utilisées pour l'entrainement du modèle. Defaults to 0.75.

    Returns:
        LogisticRegression: modèle final de régression logistique
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_size, random_state=42)
    model = LogisticAT()
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de test
    predictions = model.predict(X_test)
    
    # Évaluation de la précision du modèle
    accuracy = accuracy_score(y_test, predictions)
    
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)

    # Affichage des métriques
    print("Précision : {:.2f}%".format(accuracy * 100))
    print("Matrice de Confusion :\n", conf_matrix)
    print("Rapport de Classification :\n", class_report)
  
    
    return model

######### Je teste avec les données Iris #########

X, Y = data_preparation(toy_data=load_iris())

model = model_training(X, Y)
