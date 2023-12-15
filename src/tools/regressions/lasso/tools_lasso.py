# Basic modules
from typing import Tuple

# Array manipulation
import numpy as np

# Statistical models
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder

def train_and_evaluate_lasso_model(train_data : np.array, train_labels : np.array,
                                   test_data : np.array, test_labels : np.array,
                                   alpha : float=1.0) -> Tuple[float, np.array, float]:
    """
    Trains a Lasso regression model and evaluates its performance. Target must be categorical.

    Parameters:
        train_data (numpy array): Training data.
        train_labels (numpy array): Training target variable.
        test_data (numpy array): Test data.
        test_labels (numpy array): Test target variable.
        alpha (float): Regularization parameter for Lasso regression.

    Returns:
        classification_error (float): Classification error of the model on test data.
        coefficients (numpy array): Coefficients of the Lasso model.
        intercept (float): Intercept of the Lasso model.
    """
    # Create an OrdinalEncoder object to encode ordinal categories
    ordinal_encoder = OrdinalEncoder()

    # Transform ordinal categories of the target variable for Lasso model
    encoded_target = ordinal_encoder.fit_transform(train_labels.reshape(-1, 1))

    # Create a Lasso regression model
    lasso_model = Lasso(alpha=alpha)

    # Train the Lasso model on our training data
    lasso_model.fit(train_data, encoded_target)

    # Predictions on test data
    predictions = lasso_model.predict(test_data)

    # Transform Lasso predictions into ordinal categories to retrieve scores
    ordinal_predictions = ordinal_encoder.inverse_transform(predictions.reshape(-1, 1))

    # Calculate classification error
    classification_error = 1 - accuracy_score(test_labels, ordinal_predictions)

    # Coefficients of the Lasso model
    coefficients = lasso_model.coef_
    intercept = lasso_model.intercept_

    return classification_error, coefficients, intercept
