# Radial Basis Function (RBF) Surrogate Model
# Author: Shengning Wang

import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Union, Optional


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.utils import sl, logger


class RBF:
    """
    Radial Basis Function (RBF) Surrogate Model.

    Uses KMeans clustering to determine centers and computes Gaussian RBF features.

    Attributes:
    - num_centers (int): Number of RBF centers (neurons).
    - gamma (float): RBF width parameter.
    - alpha (float): Regularization strength.
    - centers (np.ndarray): Learned cluster centers.
    - model (Union[Ridge, LinearRegression]): The underlying linear weight solver.
    - scaler_x (StandardScaler): Preprocessor for features.
    - scaler_y (StandardScaler): Preprocessor for targets.
    """

    def __init__(self, num_centers: int = 20, gamma: float = 0.1, alpha: float = 0.0):
        """
        Initializes the RBF model configuration.

        Args:
        - num_centers (int): Number of centers to compute via KMeans.
        - gamma (float): Kernel coefficient for RBF (width).
        - alpha (float): Regularization strength for Ridge regression.
        """
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha

        # Initialize components
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.centers: Optional[np.ndarray] = None

        # Select internal regressor based on regularization
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        """
        Initialize centers
        """

        num_samples = x.shape[0]
        pca = PCA(n_components=min(x.shape), random_state=42)
        x_pca = pca.fit_transform(x)

        indices = np.linspace(0, num_samples - 1, self.num_centers, dtype=int)
        scores_selected = x_pca[indices, :]

        centers_init = pca.inverse_transform(scores_selected)

        return centers_init

    def _compute_features(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the RBF feature matrix phi(x).
        Phi_j(x) = exp(-gamma * ||x - c_j||^2)

        Args:
        - x (np.ndarray): Scaled input features (num_samples, num_features).

        Returns:
        - np.ndarray: RBF features (num_samples, num_centers).
        """
        # Distance matrix: (num_samples, num_centers)
        dists = euclidean_distances(x, self.centers, squared=True)
        return np.exp(-self.gamma * dists)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the Radial Basis Function (RBF) Surrogate Model.

        Args:
        - x_train (np.ndarray): Training feature data (num_samples, num_features)
        - y_train (np.ndarray): Training target data (num_samples, num_outputs)
        """
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # 1. Preprocessing: Scale x and y
        x_scaled = self.scaler_x.fit_transform(x_train)
        y_scaled = self.scaler_y.fit_transform(y_train)

        logger.info(f'{sl.g}training RBF (Centers={self.num_centers}, Gamma={self.gamma})...{sl.q}')

        # 2. Center Initialization (KMeans)
        num_samples = x_scaled.shape[0]
        if self.num_centers >= num_samples:
            self.centers = x_scaled
            self.num_centers = num_samples
        else:
            centers_init = self._init_centers(x_scaled)
            kmeans = KMeans(n_clusters=self.num_centers, init=centers_init, n_init=1, max_iter=500, random_state=42)
            kmeans.fit(x_scaled)
            self.centers = kmeans.cluster_centers_

        # [Optional] Set gamma
        if self.gamma is None:
            dist = euclidean_distances(self.centers, self.centers)
            np.fill_diagonal(dist, np.inf)
            min_dists = np.min(dist, axis=1)
            sigma = np.mean(min_dists)
            if sigma <= 1e-10:
                sigma = 1.0
            self.gamma = 1.0 / (2.0 * sigma ** 2)

        # 3. Feature Transformation
        phi = self._compute_features(x_scaled)

        # 4. Model Training
        self.model.fit(phi, y_scaled)

        logger.info(f'{sl.g}RBF training completed.{sl.q}')

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
        if self.centers is None:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        x_test_scaled = self.scaler_x.transform(x_test)

        logger.info(f'{sl.g}predicting RBF (Centers={self.num_centers}, Gamma={self.gamma})...{sl.q}')

        # 2. Feature Transformation
        phi_test = self._compute_features(x_test_scaled)

        # 3. Perform predictions: predict on test points
        y_pred_scaled = self.model.predict(phi_test)

        logger.info(f'{sl.g}RBF prediction completed.{sl.q}')

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
    model = RBF()

    # Train model
    model.fit(x_train, y_train)

    # Test model
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Log results
    logger.info(f'Testing R2: {sl.m}{test_metrics['r2']:.9f}{sl.q}')
    logger.info(f'Testing MSE: {sl.m}{test_metrics['mse']:.9f}{sl.q}')
    logger.info(f'Testing RMSE: {sl.m}{test_metrics['rmse']:.9f}{sl.q}')
