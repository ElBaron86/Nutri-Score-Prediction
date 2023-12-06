""" 
The module contains the API function which will be hosted on a remote server. 
The API fetches pickles from the tools\\cunsumer and tools\\producer directories

"""
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/test/', methods=['POST'])
def prediction():
    """
    Args:
      request (json): It must be of this form("{""Energie"": 377,""Mat_gras"": 1.3,
      ""Mat_gras_sat"": 1.0,""trans_fat"": 0.007,""Choles"": 0.004,""carb"": 18.0,
      ""Sucre"": 16.0,""prot"": 0.6,""Sel"": 0.0,""Fibre"": 1.6,""sodium"": 0.0,""vita_A"":0.0 ,
      ""Vita_C"":0.0016 ,""Cal"": 0.12587,""Fer"":0.0036 ,""C"": 10}")
      No values must be empty and all keys must be present
    Returns:
      response(json) It must be of this form ({"prob_E": prob_E, "prob_D": prob_D,
      "prob_C": prob_C, "prob_B": prob_B, "prob_A": prob_A})
      the sum of all probabilities is equal to 1
    """
    try:
        #Load models from pickle files 
        with open("mysite/random_forest_conso.pickle", "rb") as file:
            model_conso = pickle.load(file)
        with open("mysite/random_forest_prod.pickle", "rb") as file:
            model_prod = pickle.load(file)

        #Get JSON data from request
        data = request.json

        #Assign the values of each key to a variable
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
        vita_a = data.get('vita_A')
        vita_c = data.get('Vita_C')
        cal = data.get('Cal')
        iron = data.get('Fer')

        #Create explanatory variables according to the type of model that will be used
        var_conso = [energy, fat, saturated_fat, carbohydrates, sugars, proteins, salt]
        var_prod = [energy, fat, saturated_fat, trans_fat, cholesterol, carbohydrates,
                    sugars, fiber, sodium, vita_a,vita_c, cal, iron]

        #select the model according to the choice made by the user in the
        #check box on excel
        if data.get('C') == 10:
            df = np.array(var_conso).reshape(1, 7)
            model = model_conso
        else:
            df = np.array(var_prod).reshape(1, 13)
            model = model_prod

        #Perform predictions if no value is missing
        if all(x is not None for x in df):
            #Predict probabilities and store them
            result = model.predict_proba(df)
            prob_E, prob_D, prob_C, prob_B, prob_A = result[0]
            #Build the output JSON
            return jsonify({"prob_E": prob_E, "prob_D": prob_D, "prob_C": prob_C, 
                            "prob_B": prob_B, "prob_A": prob_A})
        #Return an error if values are missing in the request
        else:
            return jsonify({"error": "Missing data in the request"}), 400
        #Return an error if the code presents another type of error
    except ImportError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
