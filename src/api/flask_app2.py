import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/test/', methods=['POST'])
def prediction():
    try:
        # Charger les modèles depuis les fichiers pickle 
        with open("mysite/random_forest_conso.pickle", "rb") as file:
            model_conso = pickle.load(file)
        with open("mysite/random_forest_prod.pickle", "rb") as file:
            model_prod = pickle.load(file)

        # Récupérer les données JSON de la requête
        data = request.json

        # Extraire les variables d'entrée
        energy = data.get('Energie')
        fat = data.get('Mat_gras')
        saturated_fat = data.get('Mat_gras_sat')
        trans_fat = data.get('trans_fat')
        cholesterol = data.get('Choles')
        carbohydrates = data.get('carb')
        sugars = data.get('Sucre')
        proteins = data.get('prot')
        salt = data.get('Sel')
        fiber = data.get('Fibre')
        sodium = data.get('sodium')
        vita_A = data.get('vita_A')
        vita_C = data.get('Vita_C')
        cal = data.get('Cal')
        iron = data.get('Fer')

        # Liste des variables selon le modèle
        var_conso = [energy, fat, saturated_fat, carbohydrates, sugars, proteins, salt]
        var_prod = [energy, fat, saturated_fat, trans_fat, cholesterol, carbohydrates, sugars, fiber, sodium, vita_A,
                    vita_C, cal, iron]

        # Condition pour sélectionner le modèle (décommenter et ajuster si nécessaire)
        if data.get('C') == 10:
            df = np.array(var_conso).reshape(1, 7)
            model = model_conso
        else:
            df = np.array(var_prod).reshape(1, 13)
            model = model_prod

        # Effectuer une prédiction en utilisant le modèle et les données reçues.
        if all(x is not None for x in df):
            # Prédire des probabilités et les stockés
            result = model.predict_proba(df)
            prob_E, prob_D, prob_C, prob_B, prob_A = result[0]
            # Construire le JSON de sortie 
            return jsonify({"prob_E": prob_E, "prob_D": prob_D, "prob_C": prob_C, "prob_B": prob_B, "prob_A": prob_A})
        else:
            return jsonify({"error": "Missing data in the request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
