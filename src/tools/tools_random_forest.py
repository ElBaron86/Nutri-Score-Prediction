'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-11-22 18:11:59
 # @ Modified by: Jaures Ememaga
 # @ Modified time: 2023-12-01 17:03:10
 # @ Description: This script brings together all functionalities to build Randomforest consumer and producer models
 '''

#### Imports ####

# standard library
import sys
import logging
import time
from typing import List, Tuple
# Imports of third-party libraries
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use('cyberpunk')

# Set up logging
LOG_FILE_PATH = 'results_tool.log'
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)  # Set the level to ERROR to avoid printing INFO and WARNING messages

# Create a formatter for subsequent log entries
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)

# Add the handler to the root logger
logging.getLogger().addHandler(console_handler)

# Log the system date and time as the first line in the log file
with open(LOG_FILE_PATH, 'a', encoding="utf-8") as log_file:
    log_file.write(f"System Date and Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Function for the search of hyperparameters
def hyper_params_search(x_train: pd.DataFrame, y_train: pd.Series, n_cross_val: int = 5, n_iter: int = 30,
                        n_estimators_max: int = 300) -> RandomForestClassifier:
    """The 'hyper_params_Search' function is looking for the best hyperparameters for a RandomForest classifier."""

    if x_train.empty or y_train.empty:
        raise ValueError("Input dataset is empty.")

    time_start = time.time()

    logging.info("#### Search for hyper parameters in progress... ####")

    # Dictionary with the distributions of hyperparameter values to seek
    param_dist = {'n_estimators': randint(x_train.shape[1], n_estimators_max),
                  'max_depth': randint(1, x_train.shape[1] + 5)}

    rf = RandomForestClassifier()

    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=n_iter, cv=n_cross_val)

    # Model adjustment to data
    rand_search.fit(x_train, y_train)
    best_rf = rand_search.best_estimator_

    time_end = time.time()
    logging.info("Execution time : %s seconds", time_end - time_start)
    logging.info('Best hyperparameters: %s', rand_search.best_params_)
    return best_rf

# Function for predicting and displaying detailed performance in the test
def make_prediction(model, x_test: pd.DataFrame, y_test: pd.Series, plot_conf_mat: bool = False) -> np.array:
    """Function to make predictions with a Random Forest model."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info("Accuracy: %s", accuracy)
    logging.info("\n%s", classification_report(y_test, y_pred))

    if plot_conf_mat:
        cm = confusion_matrix(y_test, y_pred)
        labels = ['E', 'D', 'C', 'B', 'A']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion matrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()

    return y_pred

# Optimal model search with variable reduction (backward_selection) and hyperparameters search
def hyper_params_search_with_feature_elimination(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                                                 y_test: pd.Series, actual_model: RandomForestClassifier,
                                                 n_cross_val: int = 5, n_iter: int = 30,
                                                 n_estimators_max: int = 300,
                                                 plot_graph: bool = False) -> Tuple[RandomForestClassifier, List[str]]:
    """Function to make a selection of models by gradually removing the variables with the lowest importance."""

    if x_train.empty or y_train.empty:
        raise ValueError("Input dataset is empty.")

    time_start = time.time()

    current_x_train = pd.DataFrame(x_train.copy())
    current_x_test = pd.DataFrame(x_test.copy())

    # Initialize the best model with the model leads to you
    best_model = actual_model

    # Initialisation of variables for iterations
    best_accuracy = accuracy_score(y_test, best_model.predict(current_x_test))
    evolution_accuracy = [best_accuracy]
    removed_features = []
    best_features = list(current_x_train.columns)

    logging.info("#### Optimal model search by reduction of variables + search for hyper parameters in progress... ####")

    while current_x_train.shape[1] > 2:
        feature_importances = best_model.feature_importances_
        weakest_feature_index = np.argmin(feature_importances)

        if weakest_feature_index < current_x_train.shape[1]:
            removed_feature = current_x_train.columns[weakest_feature_index]

            # Check if the removed feature is present in the current DataFrame
            if removed_feature in current_x_train.columns:
                logging.info("Removed feature: %s", removed_feature)

                current_x_train = current_x_train.drop(removed_feature, axis=1, errors='ignore')
                current_x_test = current_x_test.drop(removed_feature, axis=1, errors='ignore')

                new_best_model = hyper_params_search(current_x_train, y_train, n_cross_val, n_iter,
                                                     n_estimators_max)
                new_best_model.fit(current_x_train, y_train)

                new_accuracy = accuracy_score(y_test, new_best_model.predict(current_x_test))

                if new_accuracy > best_accuracy:
                    best_model = new_best_model
                    best_accuracy = new_accuracy
                    best_features = list(current_x_train.columns)

                evolution_accuracy.append(new_accuracy)
                removed_features.append(removed_feature)
            else:
                break
        else:
            break

    time_end = time.time()
    logging.info("Execution time: %s seconds", time_end - time_start)
    logging.info("Best hyperparameters: %s", best_model.get_params())

    if plot_graph:
        plt.plot(range(1, len(evolution_accuracy) + 1), evolution_accuracy, marker='o')
        for i, txt in enumerate(removed_features):
            plt.annotate(txt, (i + 1, evolution_accuracy[i]), textcoords="offset points", xytext=(0, 10),
                         ha='center', rotation=45, color='red')
        plt.xlabel('Number of variables withdrawn')
        plt.ylabel('Accuracy')
        plt.show()

    return best_model, best_features
