# Polynomial Regression Surface (PRS) Surrogate Model
# Author: Shengning Wang

import os
import sys
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Union, Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import wsnet.utils.Engine as E


class PRS:
    """
    Polynomial Regression Surface (PRS) Surrogate Model.

    Utilizes polynomial feature expansion followed by linear (or ridge) regression.

    Attributes:
    - degree (int): The degree of the polynomial features.
    - alpha (float): Regularization strength for Ridge regression (0.0 for OLS).
    - model (Union[Ridge, LinearRegression]): The underlying regressor.
    - poly_trans (PolynomialFeatures): Feature transformer.
    - scaler_x (StandardScaler): Preprocessor for features.
    - scaler_y (StandardScaler): Preprocessor for targets.
    - is_fitted (bool): Status flag.
    """

    def __init__(self, degree: int = 3, alpha: float = 0.0):
        """
        Initializes the PRS model configuration.

        Args:
        - degree (int): The degree of the polynomial features.
        - alpha (float): Regularization strength. If 0.0, uses LinearRegression. If > 0.0, uses Ridge regression.
        """
        self.degree = degree
        self.alpha = alpha

        # Initialize components
        self.poly_trans = PolynomialFeatures(degree=degree)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Select internal regressor based on regularization
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()

        self.is_fitted = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the Polynomial Regression Surface (PRS) Surrogate Model.

        Args:
        - x_train (np.ndarray): Training feature data (num_samples, num_features).
        - y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # 1. Preprocessing: scale train features and targets
        x_scaled = self.scaler_x.fit_transform(x_train)
        y_scaled = self.scaler_y.fit_transform(y_train)

        E.logger.info(f'Training PRS (Degree={self.degree}, Alpha={self.alpha})...')

        # 2. Feature Transformation: Polynomial expansion
        x_poly = self.poly_trans.fit_transform(x_scaled)

        # 3. Model Training
        self.model.fit(x_poly, y_scaled)
        self.is_fitted = True

        E.logger.info(f'PRS training completed.')

    def predict(self, x_test: np.ndarray, y_test: Optional[np.ndarray] = None
                ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Predicts using the trained model.

        Args:
        - x_test (np.ndarray): Test feature data (num_samples, num_features).
        - y_test (Optional[np.ndarray]): Test target data (num_samples, num_outputs).

        Returns:
        - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        - If y_test is available: Returns a Tuple: (y_pred, metrics)
        - If y_test is None: Returns only y_pred

        - y_pred (np.ndarray): Predicted target values
        - metrics (Dict[str, float]): Dictionary of evaluation metrics
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        # 1. Preprocessing: scale test features
        if not self.is_fitted:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        x_test_scaled = self.scaler_x.transform(x_test)

        # 2. Feature Transformation: Polynomial expansion
        x_poly = self.poly_trans.transform(x_test_scaled)

        # 3. Perform predictions: predict on test points
        E.logger.info(f'Predicting PRS (Degree={self.degree}, Alpha={self.alpha})...')
        y_pred_scaled = self.model.predict(x_poly)

        E.logger.info(f'PRS prediction completed.')

        # 4. Inverse Scaling: Convert back to original space
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # 5.1 Inference Mode
        if y_test is None: return y_pred

        # 5.2 Evaluation Mode
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {'r2': r2, 'mse': mse, 'rmse': rmse}

        return y_pred, metrics


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == '__main__':
    # Simulate data
    np.random.seed(42)
    N = 100
    x = np.random.rand(N, 2) * 10

    # Branin function
    y1 = 1.0 * (x[:, 1] - 5.1 / (4.0 * np.pi**2) * x[:, 0]**2 + 5.0 / np.pi * x[:, 0] - 6.0)**2 + \
    10.0 * (1 - 1.0 / (8.0 * np.pi)) * np.cos(x[:, 0]) + 10.0

    # Simple interaction
    y2 = x[:, 0] * x[:, 1] + np.sin(x[:, 0]) * 10

    y = np.stack([y1, y2], axis=1)

    # Split data into training and testing sets (8/2 split)
    split_idx = int(0.8 * N)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Instantiate model
    model = PRS()

    # Train model
    model.fit(x_train, y_train)

    # Test model
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Log results
    E.logger.info(f'Testing R2: {test_metrics['r2']:.9f}')
    E.logger.info(f'Testing MSE: {test_metrics['mse']:.9f}')
    E.logger.info(f'Testing RMSE: {test_metrics['rmse']:.9f}')
