# Radial Basis Function (RBF) Surrogate Model
# Author: Shengning Wang

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Union, Optional


# Config Logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


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
    - scaler_X (StandardScaler): Preprocessor for features.
    - scaler_Y (StandardScaler): Preprocessor for targets.
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
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.centers: Optional[np.ndarray] = None

        # Select internal regressor based on regularization
        if self.alpha > 0:
            self.model = Ridge(alpha=self.alpha)
        else:
            self.model = LinearRegression()

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize centers
        """

        num_samples = X.shape[0]
        pca = PCA(n_components=min(X.shape), random_state=42)
        X_pca = pca.fit_transform(X)

        indices = np.linspace(0, num_samples - 1, self.num_centers, dtype=int)
        scores_selected = X_pca[indices, :]

        centers_init = pca.inverse_transform(scores_selected)

        return centers_init

    def _compute_features(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the RBF feature matrix Phi(X).
        Phi_j(x) = exp(-gamma * ||x - c_j||^2)

        Args:
        - X (np.ndarray): Scaled input features (num_samples, num_features).

        Returns:
        - np.ndarray: RBF features (num_samples, num_centers).
        """
        # Distance matrix: (num_samples, num_centers)
        dists = euclidean_distances(X, self.centers, squared=True)
        return np.exp(-self.gamma * dists)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        """
        Trains the Radial Basis Function (RBF) Surrogate Model.

        Args:
        - X_train (np.ndarray): Training feature data (num_samples, num_features)
        - Y_train (np.ndarray): Training target data (num_samples, num_outputs)
        """
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        # 1. Preprocessing: Scale X and Y
        X_scaled = self.scaler_X.fit_transform(X_train)
        Y_scaled = self.scaler_Y.fit_transform(Y_train)

        logger.info(f'Training RBF (Centers={self.num_centers}, Gamma={self.gamma})...')

        # 2. Center Initialization (KMeans)
        num_samples = X_scaled.shape[0]
        if self.num_centers >= num_samples:
            self.centers = X_scaled
            self.num_centers = num_samples
        else:
            centers_init = self._init_centers(X)
            kmeans = KMeans(n_clusters=self.num_centers, init=centers_init, n_init=1, max_iter=500, random_state=42)
            kmeans.fit(X_scaled)
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
        Phi = self._compute_features(X_scaled)

        # 4. Model Training
        self.model.fit(Phi, Y_scaled)

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
        if self.centers is None:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        X_test_scaled = self.scaler_X.transform(X_test)

        logger.info(f'Predicting RBF (Centers={self.num_centers}, Gamma={self.gamma})...')

        # 2. Feature Transformation
        Phi_test = self._compute_features(X_test_scaled)

        # 3. Perform predictions: predict on test points
        Y_pred_scaled = self.model.predict(Phi_test)

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

    # Instantiate RBF model
    model = RBF()

    # Train RBF model
    model.fit(X_train, Y_train)

    # Test RBF model
    Y_pred, test_metrics = model.predict(X_test, Y_test)

    # Show testing results
    logger.info(f'Testing R2: {test_metrics['r2']:.9f}')
    logger.info(f'Testing MSE: {test_metrics['mse']:.9f}')
