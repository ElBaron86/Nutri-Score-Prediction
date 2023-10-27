import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np
import pandas as pd


# Chargement modèle de régression logistique depuis le fichier Pickle
with open('../model/modele_ordinal.pkl', 'rb') as model_file:
    modele = pickle.load(model_file)
    
    


liste_variables = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'salt_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g']

dic_score = {0 : 'E', 1 : 'D', 2:'C', 3:'B', 4:'A'}


# Fonction pour faire des prédictions avec le modèle
def faire_prediction():
    # Récupérer les valeurs des widgets
    valeurs = []
    for entry in entries:
        try:
            valeur = float(entry.get())
            valeurs.append(valeur)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques.")
            return
    
    # Vérifier s'il y a bien 12 valeurs
    if len(valeurs) != 12:
        messagebox.showerror("Erreur", "Veuillez entrer 12 valeurs.")
        return
    
    # Convertir la liste en un tableau NumPy
    valeurs_array = np.array(valeurs).reshape(1, -1)

    # Faire la prédiction avec le modèle
    resultat = modele.predict(valeurs_array)
    messagebox.showinfo("Résultat de la prédiction", f"La prédiction est : {dic_score[resultat[0]]}")

# Créer l'interface utilisateur
root = tk.Tk()
root.title("Prédiction avec Régression Logistique")

# Créer les widgets pour entrer les valeurs
entries = []
for i in range(12):
    label = tk.Label(root, text=f"Valeur {i + 1}:")
    label.grid(row=i, column=0, padx=10, pady=10)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=10)
    entries.append(entry)

# Bouton pour faire la prédiction
predict_button = tk.Button(root, text="Faire la prédiction", command=faire_prediction)
predict_button.grid(row=12, columnspan=2, padx=10, pady=10)

# Lancement de l'interface utilisateur
root.mainloop()
