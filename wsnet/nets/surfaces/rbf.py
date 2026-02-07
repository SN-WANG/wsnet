# Radial Basis Function (RBF) Surrogate Model
# Author: Shengning Wang

import numpy as np
from typing import Optional

from wsnet.utils.scaler import StandardScalerNP


class RBF:
    """
    Radial Basis Function (RBF) using K-Means clustering and Gaussian kernels.
    """

    def __init__(self, num_centers: int = 20, gamma: Optional[float] = None, alpha: float = 0.0, max_iter: int = 500):
        """
        Args:
            num_centers (int): Number of RBF centers.
            gamma (Optional[float]): Kernel width parameter.
            alpha (float): Ridge regularization strength.
            max_iter (int): Maximum iterations for KMeans.
        """
        # parameters
        self.num_centers = num_centers
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.centers: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Computes pairwise squared Euclidean distance matrix.

        Args:
            x (np.ndarray): Inputs. Shape: (num_samples, input_dim)
            c (np.ndarray): Centers. Shape: (num_centers, input_dim)

        Returns:
            np.ndarray: Distance matrix. Shape: (num_samples, num_centers)
        """
        x2 = np.sum(x**2, axis=1, keepdims=True)
        c2 = np.sum(c**2, axis=1)
        return x2 + c2 - 2.0 * (x @ c.T)

    # ------------------------------------------------------------------

    def _init_centers(self, x: np.ndarray) -> np.ndarray:
        """
        PCA initialization using SVD.

        Args:
            x (np.ndarray): Inputs. Shape: (num_samples, input_dim)

        Returns:
            np.ndarray: Initial centers. Shape: (num_centers, input_dim)
        """
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        x_pca = u @ np.diag(s)

        idx = np.linspace(0, x.shape[0] - 1, self.num_centers, dtype=int)
        scores = x_pca[idx]

        return scores @ vt

    # ------------------------------------------------------------------

    def _kmeans(self, x: np.ndarray) -> np.ndarray:
        """
        Simple Lloyd KMeans.

        Args:
            x (np.ndarray): Inputs. Shape: (num_samples, spatial_dim)

        Returns:
            np.ndarray: Centers. Shape: (num_centers, spatial_dim)
        """
        centers = self._init_centers(x)

        for _ in range(self.max_iter):
            dists = self._compute_dists(x, centers)
            labels = np.argmin(dists, axis=1)

            new_centers = np.array(
                [
                    x[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else centers[k]
                    for k in range(self.num_centers)
                ]
            )

            if np.allclose(new_centers, centers):
                break

            centers = new_centers

        return centers

    # ------------------------------------------------------------------

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """
        Constructs Gaussian RBF feature matrix.

        Args:
            x (np.ndarray): Inputs. Shape: (num_samples, input_dim)

        Returns:
            np.ndarray: Feature matrix. Shape: (num_samples, num_centers)
        """
        dists = self._compute_dists(x, self.centers)
        return np.exp(-self.gamma * dists)

    # ------------------------------------------------------------------

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Performs model training.

        Args:
            x_train (np.ndarray): Training inputs. Shape: (num_samples, input_dim).
            y_train (np.ndarray): Training targets. Shape: (num_samples, target_dim).
        """
        if x_train.ndim == 1:
            x_train = x_train[:, None]
        if y_train.ndim == 1:
            y_train = y_train[:, None]

        x_scaled = self.scaler_x.fit(x_train, channel_dim=1).transform(x_train)
        y_scaled = self.scaler_y.fit(y_train, channel_dim=1).transform(y_train)

        num_samples = x_scaled.shape[0]
        if self.num_centers >= num_samples:
            self.centers = x_scaled
            self.num_centers = num_samples
        else:
            self.centers = self._kmeans(x_scaled)

        if self.gamma is None:
            x_var = x_scaled.var()
            self.gamma = 1.0 / (x_scaled.shape[1] * x_var) if x_var > 0 else 1.0

        phi = self._build_features(x_scaled)

        xtx = phi.T @ phi
        xty = phi.T @ y_scaled

        if self.alpha > 0.0:
            np.fill_diagonal(xtx, xtx.diagonal() + self.alpha)

        self.weights = np.linalg.solve(xtx, xty)

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Performs model prediction.

        Args:
            x_pred (np.ndarray): Prediction inputs. Shape: (num_samples, input_dim)

        Returns:
            np.ndarray: Prediction targets. Shape: (num_samples, target_dim)
        """
        if self.centers is None or self.weights is None:
            raise RuntimeError("Model has not been fitted.")

        if x_pred.ndim == 1:
            x_pred = x_pred[:, None]

        x_scaled = self.scaler_x.transform(x_pred)
        phi = self._build_features(x_scaled)

        y_scaled = phi @ self.weights
        y_pred = self.scaler_y.inverse_transform(y_scaled)

        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred
