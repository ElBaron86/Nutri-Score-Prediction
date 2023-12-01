'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-22 18:11:59
 # @ Modified by: Jaures Ememaga
 # @ Modified time: 2023-12-01 17:03:10
 # @ Description: This script brings together all functionalities to build Randomforest consumer and producer models
 '''

#### Imports ####

# standard library
import logging
import time
from typing import (List, Tuple)
import pickle
import sys

# Imports of third -party libraries
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use('cyberpunk')
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.model_selection import RandomizedSearchCV

# Set up logging
logging.basicConfig(filename='src/tools/results_tool.log', level=logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)  # Set the level to ERROR to avoid printing INFO and WARNING messages

# Create a formatter for subsequent log entries
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(console_handler)

# Log the system date and time as the first line in the log file
with open('src/tools/results_tool.log', 'a') as log_file:
    log_file.write(f"System Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
#### Function for the search for hyperparameters ####

def hyper_params_search(X_train : pd.DataFrame, y_train : pd.Series, n_cross_val : int = 5, n_iter : int = 30, n_estimators_min : int = 20,
                        n_estimators_max : int = 300, max_depth_min : int = 1, max_depth_ax : int = 20) -> RandomForestClassifier: 
                        
    """The 'hyper_params_Search' function is looking for hyperparameters for a Randomforest classifier.
    
    Args:
        X_train (pd.DataFrame) : Explanatory training variables.
        y_train (pd.Series) : Training labels.
        n_cross_val (int): Number of data divisions in the crossed validation. D 5efault
        n_iter (int): Number of iterations of the parameter search process = number of combinations of hyperparameters that will be tested.
        The greater this number, the more time will take time, so pay attention to the fixed number. Default 30
        n_estimators_min (int): Minimum number of trees in the research interval. Default 20.
        n_estimators_max (int): Maximum number of trees in the research interval. Default 300.
        max_depth_min (int): Minimum number of levels (depth) of trees in the research interval. Default 1.
        max_depth_max (int): Maximum number of levels (depth) of trees in the search interval. Default 20.
        
    Returns:
        best_rf (RandomForestClassifier): Best model obtained
    """
    time_start = time.time()
    
    logging.info("#### Search for hyper parameters in progress... ####")
    
    # Dictionary with the distributions of hyperparameter values to seek
    param_dist = {'n_estimators': randint(n_estimators_min, n_estimators_max),
              'max_depth': randint(max_depth_min, max_depth_ax)}
    
    rf = RandomForestClassifier()

    # Hyperparameters randomness
    rand_search = RandomizedSearchCV(rf, 
                                    param_distributions = param_dist, 
                                    n_iter=n_iter, 
                                    cv=n_cross_val)

    # Model adjustment to data
    rand_search.fit(X_train, y_train)
    best_rf = rand_search.best_estimator_ # Here the best model is that which has obtained a better average precision on all the Splits of the Cross Validation
    
    time_end = time.time()
    logging.info(f"Execution time : {time_end - time_start} seconds")
    logging.info('Best hyperparameters: %s', rand_search.best_params_)
    return best_rf


#### Function for predicting and displaying detailed performance in the test ####

def make_prediction(model, X_test : pd.DataFrame, y_test : pd.Series, plot_conf_mat : bool = False) -> np.array:
    """Function to make predictions with a Random Frorest model.

    Args:
        model :Model to use for prediction
        X_test (pd.):Explanatory test variables
        y_test (pd.Series) : test labels
        plot_conf_mat (bool, optional): Show the confusion matrix in Heatmap format.

    Returns:
        y_pred (np.array) : Matrix containing the predicted classes by Model
    """
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Accuracy : %s", accuracy)
    logging.info('\n%s', classification_report(y_test, y_pred)) # Allows to have more complete information on the prediction carried out
    
    if plot_conf_mat:
        # Display of the confusion matrix in Heatmap format
        cm = confusion_matrix(y_test, y_pred)
        labels = ['E', 'D', 'C', 'B', 'A']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
    return y_pred


#### Optimal model search with variable reduction (backward_selection) and hyperparameters search #### 

def hyper_params_search_with_feature_elimination(X_train, y_train, X_test, y_test, actual_model : RandomForestClassifier,
                                                n_cross_val=5, n_iter=30, 
                                                n_estimators_min=20, n_estimators_max=300, 
                                                max_depth_min=1, max_depth_max=20) -> Tuple[RandomForestClassifier, List[str]]:
    
    """Function to make a selection of models by gradually removing the variables with the lowest importance.
    
    Args:
        X_train (pd.DataFrame): Train explanatory variables.
        y_train (pd.Series): Train labels.
        X_test (pd.DataFrame): Explanatory test variables.
        y_test (pd.Series): Test labels.
        actual_model (RandomForestClassifier): Best current model
        n_cross_val (int): Number of Splits to do in the crossed validation.
        n_iter (int): Number of iterations of the parameter search process = number of combinations of hyperparameters that will be tested.
        n_estimators_min (int): Minimum number of trees in the research interval. Default 20.
        n_estimators_max (int): Maximum number of trees in the research interval. Default 300.
        max_depth_min (int): Minimum number of levels (depth) of trees in the research interval. Default 1.
        max_depth_max (int): Maximum number of levels (depth) of trees in the search interval. Default 20.
        
    Returns:
        Tuple[RandomForestClassifier, List[str]]: Best model obtained and list of important variables
    """
    time_start = time.time()
    
    current_X_train = X_train.copy()
    current_X_test = X_test.copy()
    
    # Initialize the best model with the model leads to you
    best_model = actual_model
    
    # Initialiosation of variables for iterations
    best_accuracy = accuracy_score(y_test, best_model.predict(current_X_test)) # Best precise accuracy at the start (that of preceding best_rf)
    evolution_accuracy = [best_accuracy] # accuracy list
    removed_features = [] # List that will contain the variables withdrawn
    best_features = list(current_X_train.columns) # List of the best variables (currently those of the best known model)
    
    logging.info("#### Optimal model search by reduction of variables + search for hyper parameters in progress... ####")
    
    while current_X_train.shape[1] > 2:
        feature_importances = best_model.feature_importances_ # We recover the importance of the variables of the best model
        weakest_feature_index = np.argmin(feature_importances) # we repeat the one with the lowest importance to remove it
        
        if weakest_feature_index < current_X_train.shape[1]:
            removed_feature = current_X_train.columns[weakest_feature_index] # The least important variables are removed
            
            logging.info(f"Removed feature: {removed_feature}")
            
            current_X_train = current_X_train.drop(removed_feature, axis=1) # variable update by removing the least important
            current_X_test = current_X_test.drop(removed_feature, axis=1) # Also on test data
            
            # We create a new model with the best characteristics and we train it
            new_best_model = hyper_params_search(current_X_train, y_train, n_cross_val, n_iter, 
                                                 n_estimators_min, n_estimators_max, 
                                                 max_depth_min, max_depth_max)
            new_best_model.fit(current_X_train, y_train)
            
            new_accuracy = accuracy_score(y_test, new_best_model.predict(current_X_test)) 
            
            # Update the best model and the best characteristics if only the precision is better
            if new_accuracy > best_accuracy:
                best_model = new_best_model
                best_accuracy = new_accuracy
                best_features = list(current_X_train.columns)
            
            evolution_accuracy.append(new_accuracy)
            removed_features.append(removed_feature)
        else:
            break
    
    time_end = time.time()
    logging.info(f"Execution time {time_end - time_start} seconds : ")
    logging.info(f"Best hyperparameters : {best_model.get_params()} ")
    
    plt.plot(range(1, len(evolution_accuracy) + 1), evolution_accuracy, marker='o')
    for i, txt in enumerate(removed_features):
        plt.annotate(txt, (i + 1, evolution_accuracy[i]), textcoords="offset points", xytext=(0, 10), ha='center', rotation=45, color='red')
    plt.xlabel('Number of variables withdrawn')
    plt.ylabel('Accuracy')
    plt.show()
    
    return best_model, best_features

