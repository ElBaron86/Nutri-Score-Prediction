
import numpy as np
import pickle
from flask import Flask, request, jsonify
import mord


app = Flask(__name__)

@app.route('/test/', methods=['POST'])


def prediction():
    try:
        with open("../tools/radom_forest_conso.pickle","rb") as file :
            model=pickle.load(file)

        data = request.json


        Energie = data.get('Energie')
        Mat_gras = data.get('Mat_gras')
        Mat_gras_sat = data.get('Mat_gras_sat')
        prot = data.get('prot')
        sucre = data.get('sucre')
        glu = data.get('glu')
        Sel = data.get('Sel')

        df=np.array([Energie,Mat_gras,Mat_gras_sat,prot,sucre,glu,Sel])

 # Effectue une prédiction en utilisant le modèle et les données reçues.

        if all(x is not None for x in df):
            reponse=model.predict(df)
            if reponse[0] == 0:
                resultat = "E"
            elif reponse[0] == 1:
                resultat = "D"
            elif reponse[0] == 2:
                resultat = "C"
            elif reponse[0] == 3:
                resultat = "B"
            else:
                resultat = "A"

  # Mappe la sortie de la prédiction à une valeur de lettre.

            return jsonify({"predicted_value": resultat})
        else:
            return jsonify({"error": "Missing data in the request"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()


