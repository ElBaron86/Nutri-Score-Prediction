'''
 # @ Author: Jaures Ememaga
 # @ Create Time: 2023-12-15 19:10:23
 # @ Modified by: 
 # @ Modified time:
 # @ Description: This script regulates the functions necessary to build an ordinal
 # regression model by assessing its performance according to the AIC and BIC criteria
 '''

# Basic imports
import itertools
from typing import Tuple, List

# Data manipulation
import pandas as pd
import numpy as np

# Statistical modules
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from statsmodels.miscmodels.ordinal_model import OrderedModel

def calculate_aic_bic(x: pd.DataFrame, y: pd.Series, method: str ='bfgs') -> Tuple[float, float]:
    """
    Calculate AIC and BIC criterion values for the given data and model.
    Args:
        x (pd.DataFrame): DataFrame with independent features.
        y (pd.Series): Series with the dependent variable.
        method (str): Optimization method for fitting the model.
    Returns:
        Tuple: AIC and BIC criterion values.
    """
    model = OrderedModel(y, x, distr='probit').fit(method=method, maxiter=1000)
    aic, bic = model.aic, model.bic
    return aic, bic

def fit_model(x: pd.DataFrame, y: pd.Series, features: List[str]) -> OrderedModel:
    """
    Fit the ordinal model with selected features.
    Args:
        x (pd.DataFrame): DataFrame with independent features.
        y (pd.Series): Series with the dependent variable.
        features (list): List of features to include in the model.
    Returns:
        model (OrderedModel): Fitted ordinal model.
    """
    model = OrderedModel(y, x[features], distr='probit').fit(method='bfgs', maxiter=1000)
    return model

def backward_variable_selection(x: pd.DataFrame, y: pd.Series, criterion: str ='aic') -> Tuple[List[str], List[float], List[str]]:
    """
    Perform backward variable selection based on AIC or BIC.
    Args:
        x (pd.DataFrame): DataFrame with independent features.
        y (pd.Series): Series with the dependent variable.
        criterion (str): Criterion for variable selection, either 'aic' or 'bic'.
    Returns:
        Tuple: Selected features, criterion values, and unused features.
    """
    if criterion not in ['aic', 'bic']:
        raise ValueError("Invalid criterion. Use 'aic' or 'bic'.")

    n_features = x.shape[1]
    all_features = list(x.columns)
    remaining_features = all_features.copy()
    best_features = all_features.copy()
    criterion_values = []
    unused_features = []

    min_criterion = calculate_aic_bic(x, y, method='bfgs')[0] if criterion == 'aic' else calculate_aic_bic(x, y, method='bfgs')[1]
    criterion_values.append(min_criterion)

    for _ in range(n_features - 1):
        best_combination = None

        for subset in itertools.combinations(remaining_features, len(remaining_features) - 1):
            current_criterion = calculate_aic_bic(x, y, method='bfgs')[0] if criterion == 'aic' else calculate_aic_bic(x, y, method='bfgs')[1]

            if current_criterion < min_criterion:
                min_criterion = current_criterion
                best_combination = subset

        if best_combination is not None:
            unused_feature = set(best_features) - set(best_combination)
            unused_feature_name = unused_feature.pop()

            unused_features.append(unused_feature_name)
            remaining_features = list(best_combination)
            best_features.remove(unused_feature_name)

            criterion_values.append(min_criterion)
        else:
            break

    return best_features, criterion_values, unused_features

def cross_validation(k: int, x: pd.DataFrame, y: pd.Series, features: List[str]) -> float:
    """
    Perform k-fold cross-validation and return the average classification error.
    Args:
        k (int): Number of folds for cross-validation.
        x (pd.DataFrame): DataFrame with independent features.
        y (pd.Series): Series with the dependent variable.
        features (list): List of features to include in the model.
    Returns:
        float: Average classification error across all folds.
    """
    mse_per_split = []
    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, valid_idx in cv.split(x):
        x_fold_train, y_fold_train = x.iloc[train_idx], y.iloc[train_idx]
        x_fold_valid, y_fold_valid = x.iloc[valid_idx], y.iloc[valid_idx]

        model_fold = fit_model(x_fold_train, y_fold_train, features)
        y_prob_valid = model_fold.predict(x_fold_valid)
        y_pred_valid = np.argmax(y_prob_valid.criterion_values, axis=1)
        mse = 1 - accuracy_score(y_fold_valid, y_pred_valid)

        mse_per_split.append(mse)

    return np.mean(mse_per_split)
