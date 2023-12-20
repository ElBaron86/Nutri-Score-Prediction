'''
 # @ Author: Your name
 # @ Create Time: 2023-12-15 19:33:15
 # @ Modified by: Your name
 # @ Modified time: 2023-12-15 20:01:57
 # @ Description:
 '''

# Basic modules
import sys
import os

# Modules for logging
import logging
import warnings

# Data manipulation
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
import numpy as np

# Statistical module
from sklearn.metrics import accuracy_score


# Moving up to the "src" directory
while os.path.basename(os.getcwd()) != "src":
    os.chdir("..")
sys.path.append(os.getcwd())

# Importing functions
from tools.regressions.ordinal.tools_ordinal_regression import backward_variable_selection, cross_validation, fit_model

# Configure logging
# Setting up logging configuration
logging.basicConfig(filename='tools/regressions/ordinal/model_results_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Redirecting warnings to the logging system
warnings.showwarning = lambda message, *args, **kwargs: logging.warning(message, stacklevel=2)

# Loading data
df = pd.read_csv("data/data_clean.csv")

# Dropping product names
df.drop(labels=["product_name"], axis=1, inplace=True)

# Getting features names
target_col = 'score'
features = df.columns[df.columns != target_col]

# Ordering labels
categories_order = ['E', 'D', 'C', 'B', 'A']
df[target_col] = pd.Categorical(df[target_col], categories=categories_order, ordered=True)

# Split the data into features (X) and target variable (y)
X = df[features]
y = df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Backward variable selection based on AIC
selected_features_aic, _, _ = backward_variable_selection(X_train, y_train, criterion='aic')

# Backward variable selection based on BIC
selected_features_bic, _, _ = backward_variable_selection(X_train, y_train, criterion='bic')

# Choose the best features based on AIC or BIC
selected_features = selected_features_aic if len(selected_features_aic) < len(selected_features_bic) else selected_features_bic

# Fit the model using cross-validation
k_fold = 5  # Number of folds for cross-validation
average_classification_error = cross_validation(k_fold, X_train, y_train, selected_features)

# Train the final model on the entire training set with selected features
final_model = fit_model(X_train, y_train, selected_features)

# Evaluate the final model on the test set
y_prob_test = final_model.predict(X_test)
y_pred_test = np.argmax(y_prob_test.criterion_values, axis=1)
test_classification_error = 1 - accuracy_score(y_test, y_pred_test)

# Log the results
logging.info(f"Selected Features: {selected_features}")
logging.info(f"Average Classification Error (Cross-Validation): {average_classification_error}")
logging.info(f"Classification Error on Test Set: {test_classification_error}")
