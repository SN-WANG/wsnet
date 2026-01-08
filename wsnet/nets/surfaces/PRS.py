# Polynomial Regression Surface (PRS) Surrogate Model
# Author: Shengning Wang

import logging
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Union, Optional


# Config Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class PRS:
    """
    Polynomial Regression Surface (PRS) Surrogate Model.

    Utilizes polynomial feature expansion followed by linear (or ridge) regression.

    Attributes:
    - degree (int): The degree of the polynomial features.
    - alpha (float): Regularization strength for Ridge regression (0.0 for OLS).
    - model (Union[Ridge, LinearRegression]): The underlying regressor.
    - poly_trans (PolynomialFeatures): Feature transformer.
    - scaler_X (StandardScaler): Preprocessor for features.
    - scaler_Y (StandardScaler): Preprocessor for targets.
    - if_fitted (bool): Status flag.
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
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        # Select internal regressor based on regularization
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()

        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        """
        Trains the Polynomial Regression Surface (PRS) Surrogate Model.

        Args:
        - X_train (np.ndarray): Training feature data (num_samples, num_features).
        - Y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # 1. Preprocessing: scale train features and targets
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)

        logger.info(f'Training PRS (Degree={self.degree}, Alpha={self.alpha})...')

        # 2. Feature Transformation: Polynomial expansion
        X_poly = self.poly_trans.fit_transform(X_scaled)

        # 3. Model Training
        self.model.fit(X_poly, Y_scaled)
        self.is_fitted = True

        logger.info('Training completed')

    def predict(self, X_test: np.ndarray, Y_test: Optional[np.ndarray] = None
                ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Predicts using the trained model.

        Args:
        - X_test (np.ndarray): Test feature data (num_samples, num_features).
        - Y_test (Optional[np.ndarray]): Test target data (num_samples, num_outputs).

        Returns:
        - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        - If Y_test is available: Returns a Tuple: (Y_pred, metrics)
        - If Y_test is None: Returns only Y_pred

        - Y_pred (np.ndarray): Predicted target values
        - metrics (Dict[str, float]): Dictionary of evaluation metrics
        """

        # 1. Preprocessing: scale test features
        if not self.is_fitted:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        X_test_scaled = self.scaler_X.transform(X_test)

        # 2. Feature Transformation: Polynomial expansion
        X_poly = self.poly_trans.transform(X_test_scaled)

        # 3. Perform predictions: predict on test points
        logger.info(f'Predicting PRS (Degree={self.degree}, Alpha={self.alpha})...')
        Y_pred_scaled = self.model.predict(X_poly)

        logger.info(f'Prediction completed')

        # 4. Inverse Scaling: Convert back to original space
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # 5.1. Return predictions if y_test is not available (Inference Mode)
        if Y_test is None:
            return Y_pred

        # 5.2.1 Metrics Calculation: Evaluate model performance (Evaluation Mode)
        r2 = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        }

        # 5.2.2 Return predictions and performance metrics
        return Y_pred, metrics


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == '__main__':
    # Simulate data
    np.random.seed(42)
    N = 100
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

    # Instantiate PRS model
    model = PRS()

    # Train PRS model
    model.fit(X_train, Y_train)

    # Test PRS model
    Y_pred, test_metrics = model.predict(X_test, Y_test)

    # Show testing results
    logger.info(f'Testing R2: {test_metrics['r2']:.9f}')
    logger.info(f'Testing MSE: {test_metrics['mse']:.9f}')
