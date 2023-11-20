
import numpy as np
import pickle
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier


app = Flask(__name__)

@app.route('/test/', methods=['POST'])


def prediction():
    try:
        with open("..\\tools\\random_forest_conso.pickle","rb") as file :
            model_conso=pickle.load(file)
            
        with open("..\\tools\\random_forest_prod.pickle","rb") as file :
            model_prod=pickle.load(file)

        data = request.json


        energy = data.get('Energie')
        fat = data.get('Mat_gras')
        saturated_fat = data.get('Mat_gras_sat')
        trans_fat=data.get('trans_fat')
        cholesterol = data.get('Choles')
        carbohydrates = data.get('carb')
        sugars = data.get('Sucre')
        proteins= data.get('prot')
        salt= data.get('Sel')
        fiber = data.get('Fibre')
        sodium= data.get('Fibre')
        vita_A= data.get('Fibre')
        vita_C = data.get('Vita_C')
        cal = data.get('Cal')
        iron = data.get('Fer')

        # liste des variables selon le modèle
        var_conso=[energy,fat,saturated_fat,carbohydrates,sugars,proteins,salt]
        var_prod=[energy,fat,saturated_fat,trans_fat,cholesterol,carbohydrates,sugars,fiber,sodium,vita_A,
                  vita_C,cal,iron]

        # Condition pour sélectionner le modèle
        if data.get('conso') == 1:
            df=np.array(var_conso).reshape(1,7)
            model=model_conso
        else:
            df=np.array(var_prod).reshape(1,13)
            model=model_prod
        # Effectue une prédiction en utilisant le modèle et les données reçues.

        if all(x is not None for x in df):
            result=model.predict_proba(df)
            prob_E=result[0][0]
            prob_D=result[0][1]
            prob_C=result[0][2]
            prob_B=result[0][3]
            prob_A=result[0][4]

  # Construire le json de sortie.

            return jsonify({"prob_E": prob_E,"prob_D": prob_D,"prob_C": prob_C,"prob_B": prob_B,"prob_A": prob_A})
        else:
            return jsonify({"error": "Missing data in the request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()


