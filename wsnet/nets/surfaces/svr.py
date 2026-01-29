# Support Vector Regression (SVR) Surrogate Model
# Author: Shengning Wang

import os
import sys
import numpy as np
from sklearn.svm import SVR as SklearnSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Union, Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.utils import sl, logger


class SVR:
    """
    Support Vector Regression (SVR) Surrogate Model.

    Wraps sklearn's SVR in a MultiOutputRegressor to handle multiple output dimensions.

    Attributes:
    - kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
    - gamma (Union[str, float]): Kernel coefficient. Controls the influence radius.
                                 'scale' (default) uses 1 / (num_features * x.var()).
    - C (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
    - epsilon (float): Epsilon-tube width. Defines the margin of tolerance where no penalty is associated with errors.
    - model (MultiOutputRegressor): The multi-output wrapper around SVR.
    - scaler_x (StandardScaler): Preprocessor for features.
    - scaler_y (StandardScaler): Preprocessor for targets.
    - is_fitted (bool): Status flag.
    """

    def __init__(self, kernel: str = 'rbf', gamma: Union[str, float] = 'scale', C: float = 1.0, epsilon: float = 0.1):
        """
        Initializes the SVR model configuration.

        Args:
        - kernel (str): Specifies the kernel type.
        - gamma (Union[str, float]): Kernel coefficient.
        - C (float): Regularization parameter.
        - epsilon (float): Epsilon-tube width.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        self.model = None
        self.is_fitted = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Trains the Support Vector Regression (SVR) Surrogate Model.

        Args:
        - x_train (np.ndarray): Training feature data (num_samples, num_features).
        - y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # 1. Initialize base SVR
        svr_estimator = SklearnSVR(kernel=self.kernel, gamma=self.gamma, C=self.C, epsilon=self.epsilon)
        self.model = MultiOutputRegressor(svr_estimator, n_jobs=-1)

        # 2. Preprocessing: Scale x and y
        x_scaled = self.scaler_x.fit_transform(x_train)
        y_scaled = self.scaler_y.fit_transform(y_train)

        logger.info(f"training SVR (kernel={self.kernel}, gamma={self.gamma}, "
                    f"c={self.C}, epsilon={self.epsilon})...")

        # 3. Model Training
        self.model.fit(x_scaled, y_scaled)
        self.is_fitted = True

        logger.info(f'{sl.g}SVR training completed.{sl.q}')

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

        # 2. Perform predictions: predict on test points
        logger.info(f"predicting SVR (kernel={self.kernel}, gamma={self.gamma}, "
                    f"c={self.C}, epsilon={self.epsilon})...")
        y_pred_scaled = self.model.predict(x_test_scaled)
        logger.info(f'{sl.g}SVR prediction completed.{sl.q}')

        # 3. Inverse Scaling: Convert back to original space
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # 4.1 Inference Mode
        if y_test is None: return y_pred

        # 4.2 Evaluation Mode
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {"r2": r2, "mse": mse, "rmse": rmse}

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
    model = SVR()

    # Train model
    model.fit(x_train, y_train)

    # Test model
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Log results
    logger.info(f'Testing R2: {sl.m}{test_metrics['r2']:.9f}{sl.q}')
    logger.info(f'Testing MSE: {sl.m}{test_metrics['mse']:.9f}{sl.q}')
    logger.info(f'Testing RMSE: {sl.m}{test_metrics['rmse']:.9f}{sl.q}')
