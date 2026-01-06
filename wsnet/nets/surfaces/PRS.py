# Polynomial Regression Surface (PRS) Surrogate Model
# Author: Shengning Wang

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from typing import Dict, Any, Tuple, Union, Optional


# Define the type for the model object
PRSModel = Union[Ridge, LinearRegression]


def train_prs_model(X_train: np.ndarray, Y_train: np.ndarray, degree: int = 2,
                    alpha: float = 1.0, use_cross_val: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train a Polynomial Regression Surface (PRS) Surrogate Model

    Args:
    - X_train (np.ndarray): Training feature data (num_samples, num_features)
    - Y_train (np.ndarray): Training target data (num_samples, num_outputs)
    - degree (int): The degree of the polynomial features
    - alpha (float): Regularization strength (lambda) for Ridge regression
                     Set to 0.0 to use standard Linear Regression
    - use_cross_val (bool): Whether to perform cross-validation to assess model stability and performance
    - cv_folds (int): Number of folds for K-fold cross-validation

    Returns:
    - Dict[str, Any]: A dictionary containing the trained model, feature transformer, and cross-validation results
    """

    # Feature Engineering: Generate polynomial and interaction features
    poly_transformer = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_transformer.fit_transform(X_train)  # (num_samples, num_new_features)

    # Model Selection: Choose the regression algorithm (Ridge for regularization)
    if alpha > 0.0:
        # Ridge regression
        model = Ridge(alpha=alpha, solver='auto', random_state=42)
        model_name = 'Ridge PRS Model'
    else:
        # Standard linear regression
        model = LinearRegression()
        model_name = 'Standard PRS Model'

    # Model Training: Fit the model to the polynomial features
    print(f'# Training {model_name} with degree={degree}, alpha={alpha}...')
    model.fit(X_poly, Y_train)
    print(f'# {model_name} training completed')

    # Cross-validation: Assess generalization performance
    cv_results = {}
    if use_cross_val:
        cv_scores = cross_val_score(estimator=model, X=X_poly, y=Y_train, scoring='r2',
                                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42), n_jobs=-1)

        cv_results = {
            'cv_folds': cv_folds,
            'cv_mean_scores': np.mean(cv_scores),
            'cv_std_scores': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }

    # Return the trained model
    return {
        'model_name': model_name,
        'model': model,
        'poly_transformer': poly_transformer,
        'degree': degree,
        'alpha': alpha,
        'cv_results': cv_results
    }


def test_prs_model(trained_model_dict: Dict[str, Any],
                   X_test: np.ndarray, Y_test: Optional[np.ndarray] = None
                   ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Evaluate the trained Polynomial Regression Surface (PRS) Surrogate Model on test data

    Args:
    - trained_model_dict (Dict[str, Any]): Dictionary returned by train_prs_model
    - X_test (np.ndarray): Test feature data (num_samples, num_features)
    - Y_test (np.ndarray, Optional): Test target data (num_samples, num_outputs)

    Returns:
    - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    - If Y_test is available: Returns a Tuple: (Y_pred, metrics)
    - If Y_test is None: Returns only Y_pred

    - Y_pred (np.ndarray): Predicted target values
    - metrics (Dict[str, float]): Dictionary of evaluation metrics
    """

    # Retrieve the trained model
    model_name = trained_model_dict['model_name']
    model: PRSModel = trained_model_dict['model']
    poly_transformer: PolynomialFeatures = trained_model_dict['poly_transformer']

    # Feature Transformer: Apply the PRS transformation to the test data
    X_test_poly = poly_transformer.transform(X_test)

    # Prediction: Generate predictions on the transformed test features
    print(f'# Predicting {model_name}...')
    Y_pred = model.predict(X_test_poly)
    print(f'# {model_name} prediction completed')

    # Return predictions if Y_test is not available (Inference Mode)
    if Y_test is None:
        return Y_pred

    # Metrics Calculation: Evaluate model performance (Evaluation Mode)
    r2 = r2_score(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'r2': r2,
        'mse': mse,
        'rmse': rmse
    }

    # Return predictions and performance metrics
    return Y_pred, metrics


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == '__main__':
    # Simulate data
    np.random.seed(42)
    N = 1000
    X = np.random.rand(N, 2) * 10

    # Branin function
    y1 = 1.0 * (X[:, 1] - 5.1 / (4.0 * np.pi**2) * X[:, 0]**2 + 5.0 / np.pi * X[:, 0] - 6.0)**2 + \
    10.0 * (1 - 1.0 / (8.0 * np.pi)) * np.cos(X[:, 0]) + 10.0

    # Simple interaction
    y2 = X[:, 0] * X[:, 1] + np.sin(X[:, 0]) * 10

    Y = np.stack([y1, y2], axis=1)

    # Split data into training and testing sets (8/2 split)
    split_idx = int(0.8 * N)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    # Train model
    model = train_prs_model(X_train, Y_train, degree=3, alpha=0.5)

    # Show cross-validation results
    print(f'# Cross-validation Mean R2: {model['cv_results']['cv_mean_scores']:.9f}')

    # Test model
    Y_pred, test_metrics = test_prs_model(model, X_test, Y_test)

    # Show testing results
    print(f'# Testing R2: {test_metrics['r2']:.9f}')
