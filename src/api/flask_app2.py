import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/test/', methods=['POST'])
def prediction():
   """
    Args:
      requête (json): Elle doit être de cette forme("{""Energie"": 377,""Mat_gras"": 1.3,
      ""Mat_gras_sat"": 1.0,""trans_fat"": 0.007,""Choles"": 0.004,""carb"": 18.0,
      ""Sucre"": 16.0,""prot"": 0.6,""Sel"": 0.0,""Fibre"": 1.6,""sodium"": 0.0,""vita_A"":0.0 ,
      ""Vita_C"":0.0016 ,""Cal"": 0.12587,""Fer"":0.0036 ,""C"": 10}"
                )
    Aucune valeur ne doit être vide et toutes les clés doivent figurer
    
    Returns:
      reponse(json) Elle est de la forme ({"prob_E": prob_E, "prob_D": prob_D, "prob_C": prob_C, "prob_B": prob_B, "prob_A": prob_A})
      la somme de toutes les probabilités est égale à 1
    """
    
    
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
