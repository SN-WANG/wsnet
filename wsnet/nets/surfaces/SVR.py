# Support Vector Regression (SVR) Surrogate Model
# Author: Shengning Wang

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from typing import Dict, Any, Tuple, Union, Optional


def train_svr_model(X_train: np.ndarray, Y_train: np.ndarray, kernel: str = 'rbf',
                    C: float = 1.0, epsilon: float = 0.1, gamma: Union[str, float] = 'scale',
                    use_cross_val: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train a Support Vector Regression (SVR) Surrogate Model

    Args:
    - X_train (np.ndarray): Training feature data (num_samples, num_features)
    - Y_train (np.ndarray): Training target data (num_samples, num_outputs)
    - kernel (str): Specifies the kernel type to be used. Common choices: 'rbf', 'linear', 'poly', 'sigmoid'
    - C (float): Regularization parameter. The strength of the regularization is inversely proportional to C
    - epsilon (float): Epsilon-tube width in SVR. Defines the margin of tolerance
                       where no penalty is associated with errors
    - gamma (Union[float, str]): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. Controls the influence radius.
                                 'scale' (default) uses 1 / (num_features * X.var())
    - use_cross_val (bool): Whether to perform cross-validation to assess model stability
    - cv_folds (int): Number of folds for K-fold cross-validation

    Returns:
    - Dict[str, Any]: A dictionary containing the trained model, feature scalar and cross-validation results
    """

    # Preprocessing: Scale features for SVR for better convergence and performance
    # SVR is sensitive to feature scaling, a standard scaler is necessary
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Model Selection
    is_multi_output = Y_train.ndim > 1 and Y_train.shape[1] > 1
    estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
    if is_multi_output:
        model = MultiOutputRegressor(estimator, n_jobs=-1)
    else:
        model = estimator
        Y_train = Y_train.ravel()
    model_name = f'SVR Model ({kernel} kernel)'

    # Model Training: Fit the model to the scaled features
    print(f'# Training {model_name} with C={C}, epsilon={epsilon}, gamma={gamma}, ...')
    model.fit(X_scaled, Y_train)
    print(f'# {model_name} training completed')

    # Cross-validation: Assess generalization performance
    cv_results = []
    if use_cross_val:
        cv_scores = cross_val_score(estimator=model, X=X_scaled, y=Y_train, scoring='r2',
                                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=42), n_jobs=-1)

        cv_results = {
            'cv_folds': cv_folds,
            'cv_mean_scores': np.mean(cv_scores),
            'cv_std_scores': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }

    # Return the trained model
    return {'model_name': model_name,
            'model': model,
            'scaler': scaler,
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': estimator.gamma,
            'cv_results': cv_results
        }


def test_svr_model(trained_model_dict: Dict[str, Any],
                   X_test: np.ndarray, Y_test: Optional[np.ndarray] = None
                   ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Evaluate the trained Support Vector Regression (SVR) Surrogate Model on test data

    Args:
    - trained_model_dict (Dict[str, Any]): Dictionary returned by train_rbf_model
    - X_test (np.ndarray): Test feature data (num_samples, num_features)
    - Y_test (np.ndarray, Optional): True target variable for test data (num_samples, num_outputs)

    Returns:
    - Union[np.ndarray, Tuple[np.array, Dict[str, float]]]:
    - If Y_test is available: Returns a Tuple: (Y_pred, metrics)
    - If Y_test is None: Returns only Y_pred

    - Y_pred (np.ndarray): Predicted target values
    - metrics (Dict[str, float]): Dictionary of evaluation metrics
    """

    # Retrieve the trained model
    model_name = trained_model_dict['model_name']
    model: SVR = trained_model_dict['model']
    scaler: StandardScaler = trained_model_dict['scaler']

    # Preprocessing: Apply the fitted scaler transformation to the test data
    X_scaled = scaler.transform(X_test)

    # Prediction: Generate predictions on the test features
    print(f'# Predicting {model_name}...')
    Y_pred = model.predict(X_scaled)
    print(f'# {model_name} prediction completed')

    # Return predictions if y_test is not available (Inference Mode)
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
    model = train_svr_model(X_train, Y_train, kernel='rbf', C=10.0, epsilon=0.01, gamma='scale')

    # Show cross-validation results
    print(f'# Cross-validation Mean R2: {model['cv_results']['cv_mean_scores']:.9f}')

    # Test model
    Y_pred, test_metrics = test_svr_model(model, X_test, Y_test)

    # Show testing results
    print(f'# Testing R2: {test_metrics['r2']:.9f}')
