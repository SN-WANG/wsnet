# Support Vector Regression (SVR) Surrogate Model
# Author: Shengning Wang

import logging
import numpy as np
from sklearn.svm import SVR as SklearnSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Union, Optional


# Config Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class SVR:
    """
    Support Vector Regression (SVR) Surrogate Model.

    Wraps sklearn's SVR in a MultiOutputRegressor to handle multiple output dimensions.

    Attributes:
    - kernel (str): Kernel type ('rbf', 'linear', 'poly', 'sigmoid').
    - gamma (Union[str, float]): Kernel coefficient. Controls the influence radius.
                                 'scale' (default) uses 1 / (num_features * X.var()).
    - C (float): Regularization parameter. The strength of the regularization is inversely proportional to C.
    - epsilon (float): Epsilon-tube width. Defines the margin of tolerance where no penalty is associated with errors.
    - model (MultiOutputRegressor): The multi-output wrapper around SVR.
    - scaler_X (StandardScaler): Preprocessor for features.
    - scaler_Y (StandardScaler): Preprocessor for targets.
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

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        # Initialize base SVR
        svr_estimator = SklearnSVR(kernel=self.kernel, gamma=self.gamma, C=self.C, epsilon=self.epsilon)

        # Wrap for multi-output support
        self.model = MultiOutputRegressor(svr_estimator, n_jobs=-1)
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """
        Trains the Support Vector Regression (SVR) Surrogate Model.

        Args:
        - X_train (np.ndarray): Training feature data (num_samples, num_features).
        - Y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # 1. Preprocessing: Scale X and Y
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)

        logger.info(f'Training SVR (Kernel={self.kernel}, gamma={self.gamma}, C={self.C}, epsilon={self.epsilon})...')

        # 2. Model Training
        self.model.fit(X_scaled, Y_scaled)
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

        # 2. Perform predictions: predict on test points
        logger.info(f'Predicting SVR (Kernel={self.kernel}, gamma={self.gamma}, C={self.C}, epsilon={self.epsilon})...')
        Y_pred_scaled = self.model.predict(X_test_scaled)
        logger.info(f'Prediction completed')

        # 3. Inverse Scaling: Convert back to original space
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)

        # 4.1. Return predictions if y_test is not available (Inference Mode)
        if Y_test is None:
            return Y_pred

        # 4.2.1 Metrics Calculation: Evaluate model performance (Evaluation Mode)
        r2 = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)

        metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse
        }

        # 4.2.2 Return predictions and performance metrics
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

    # Instantiate SVR model
    model = SVR()

    # Train SVR model
    model.fit(X_train, Y_train)

    # Test SVR model
    Y_pred, test_metrics = model.predict(X_test, Y_test)

    # Show testing results
    logger.info(f'Testing R2: {test_metrics['r2']:.9f}')
    logger.info(f'Testing MSE: {test_metrics['mse']:.9f}')
