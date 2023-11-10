"""
Ce code permet d'entrainer le modèle de régression ordinale sur de nouvelles données. Le modène est mis à jour uniquement si la précision au test est >75% 
"""

# Importation des modules
import pandas as pd
from mord import LogisticAT
from sklearn.metrics import accuracy_score
import pickle

# Chargement des données existantes
variables = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'cholesterol_100g', 
             'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 
             'sodium_100g', 'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g', 'nutrition_grade_fr']

nutri_code = {'A' : 4, 'B' : 3, 'C' : 2, 'D' : 1, 'E' : 0}
df_train = pd.read_csv("C:/Users/jaure/OneDrive/Documents/M2 SEP/projet_digital/Projet_digital/data/train.csv")
df_test = pd.read_csv("C:/Users/jaure/OneDrive/Documents/M2 SEP/projet_digital/Projet_digital/data/test.csv")

check = input("Avez-vous de nouvelles données à ajouter ? (O/N) ")

if check.upper() == 'N':
    print("Ok! Fin du programme.")
elif check.upper() == 'O':
    path = input("Entrez le chemin de votre fichier (csv) : ")
    path = path.encode('unicode_escape').decode() # correction de l'encodage pour éviter les erreurs de lecture liées au chemin

    # Fonction pour nettoyer et prétraiter les nouvelles données
    def clean_data(path):
        if "score" in list(pd.read_csv(path).columns):
            variables[-1] = "score"
            new_df = pd.read_csv(path, usecols=variables)
            new_df.dropna(inplace=True)
        else:
            new_df = pd.read_csv(path, usecols=variables)
            new_df.rename(columns={"nutrition_grade_fr": "score"}, inplace=True)
            new_df['score'] = new_df['score'].apply(lambda x: nutri_code[x.upper()])
            print("########## Nouvelles données nettoyées et traitées ###########")
        return new_df

    # On applique le chargement et nettoyage des nouvelles données
    new_data = clean_data(path)

    # Fusionn des nouvelles données avec l'ensemble d'entraînement existant
    merged_data = pd.concat([df_train, new_data], ignore_index=True, sort=False)
    merged_data.drop_duplicates(keep='first', inplace=True)

    # Entraînement du modèle de régression logistique ordinale
    model = LogisticAT(alpha=0)
    X_train = merged_data.drop(labels=["score"], axis=1).values
    y_train = merged_data["score"].values
    model.fit(X_train, y_train)

    # Prédictions sur l'ensemble de test connu
    X_test = df_test.drop(labels=["score"], axis=1).values
    y_test = df_test["score"].values
    preds = model.predict(X_test)

    # Calcul de la précision
    acc = accuracy_score(y_test, preds)
    print("Précision du modèle : {:.2f}%".format(acc * 100))

    # Sauvegarde du modèle si la précision est supérieure à 73%
    if acc > 0.75:
        with open("modele_ordinal.pkl", 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Modèle sauvegardé.")
    else:
        print("Le modèle n'a pas atteint la précision requise et n'a pas été sauvegardé.")
else:
    print("Saisie incorrecte, veuillez répondre par O ou N")
