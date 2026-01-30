# Multi-Fidelity Surrogate Model based on Moving Least Squares (MFS-MLS)
# Paper reference: https://doi.org/10.1007/s00158-021-03044-5
# Paper author: Shuo Wang, Yin Liu, Qi Zhou, Yongliang Yuan, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, Union, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets.surfaces import RBF
from wsnet.utils.engine import sl, logger


class MFSMLS:
    """
    Adaptive Multi-Fidelity Surrogate Model based on Moving Least Squares (MFS-MLS).

    Fuses Low-Fidelity (LF) and High-Fidelity (HF) data.
    Approximation: y_hf(x) = rho(x) * y_lf(x) + d(x)
    Solved via Weighted Least Squares with an adaptive influence domain.

    Attributes:
    - lf_model (RBF): Trained LF surrogate model.
    - poly_degree (int): Degree of polynomial basis for discrepancy (1 = Linear, 2 = Quadratic).
    - poly_trans (PolynomialFeatures): Sklearn transformer for polynomial basis expansion.
    - scaler_x (StandardScaler): Normalization processor for input features.
    - scaler_y (StandardScaler): Normalization processor for target responses.
    - x_hf_train_ (np.ndarray): Stored High-Fidelity training inputs (num_hf_samples, num_features).
    - y_hf_train_ (np.ndarray): Stored High-Fidelity training targets (num_hf_samples, num_outputs).
    - p_train_ (np.ndarray): Pre-computed polynomial features for HF inputs.
    - d_max_ (float): The maximum distance among HF samples, used as the base influence radius for the weighting function.
    - is_fitted (bool): Status flag.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None, poly_degree: int = 2):
        """
        Initialize the MFS-MLS model configuration.

        Args:
        - lf_model_params (Optional[Dict]): Dictionary of kwargs to pass to RBF class.
        - poly_degree (int): Degree of the polynomial basis function. 1 for Linear, 2 for Quadratic.
        """
        self.poly_degree = poly_degree

        # Initialize the LF RBF model
        params = lf_model_params if lf_model_params is not None else {}
        self.lf_model = RBF(**params)

        # Polynomial feature generator
        self.poly_trans = PolynomialFeatures(degree=poly_degree, include_bias=True)

        # Scalers for normalization
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Model state
        self.x_hf_train_: Optional[np.ndarray] = None  # (num_hf_train, num_features)
        self.y_hf_train_: Optional[np.ndarray] = None  # (num_hf_train, num_outputs)
        self.p_train_: Optional[np.ndarray] = None  # (num_hf_train, num_basis + 1)
        self.d_max_: float = 0.0
        self.is_fitted: bool = False

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Trains the MFS-MLS model using both LF and HF datasets.

        Args:
        - x_lf (np.ndarray): Low-Fidelity inputs (num_lf_samples, num_features).
        - y_lf (np.ndarray): Low-Fidelity targets (num_lf_samples, num_outputs).
        - x_hf (np.ndarray): High-Fidelity inputs (num_hf_samples, num_features).
        - y_hf (np.ndarray): High-Fidelity targets (num_hf_samples, num_outputs).
        """
        if x_lf.ndim == 1: x_lf = x_lf.reshape(-1, 1)
        if y_lf.ndim == 1: y_lf = y_lf.reshape(-1, 1)
        if x_hf.ndim == 1: x_hf = x_lf.reshape(-1, 1)
        if y_hf.ndim == 1: y_hf = y_hf.reshape(-1, 1)

        logger.info(f'training MFS-MLS (polyDegree={self.poly_degree})...')

        # 1. Train LF model on raw data
        self.lf_model.fit(x_lf, y_lf)

        # 2. Scale HF data
        self.x_hf_train_ = self.scaler_x.fit_transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit_transform(y_hf)

        # 3. Construct augmented basis matrix P = [y_lf(x_hf), 1, x1, x2, ..., x1^2, ...]
        y_lf_at_hf = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf, tuple): y_lf_at_hf = y_lf_at_hf[0]  # For KRG's mse_pred
        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf)

        poly_basis = self.poly_trans.fit_transform(self.x_hf_train_)
        self.p_train_ = np.concatenate([y_lf_at_hf_scaled, poly_basis], axis=1)

        # 4. Calculate adaptive influence radius d_max (max distance in HF set)
        dist_matrix = cdist(self.x_hf_train_, self.x_hf_train_, metric='euclidean')
        self.d_max_ = np.max(dist_matrix)

        self.is_fitted = True
        logger.info(f'{sl.g}MFS-MLS training completed.{sl.q}')

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

        num_samples = x_test.shape[0]
        num_hf_samples, num_basis = self.p_train_.shape
        num_outputs = self.y_hf_train_.shape[1]

        x_test_scaled = self.scaler_x.transform(x_test)

        logger.info(f'predicting MFS-MLS (polyDegree={self.poly_degree})...')

        # 2. Prepare basis for query points
        y_lf_at_test = self.lf_model.predict(x_test)
        y_lf_at_test_scaled = self.scaler_y.transform(y_lf_at_test)
        poly_basis_test = self.poly_trans.transform(x_test_scaled)
        p_test_ = np.concatenate([y_lf_at_test_scaled, poly_basis_test], axis=1)

        # 3. Pre-calculate distances to all HF samples
        dists = cdist(x_test_scaled, self.x_hf_train_, metric='euclidean')

        y_pred_scaled = np.zeros((num_samples, num_outputs))

        # 4. MLS logic per test point: A(x) = (P^T * W(x) * P)^{-1} * P^T * W(x) * Y
        for i in range(num_samples):
            # d_norm = dist / d_max
            di = dists[i] / (self.d_max_ + 1e-12)

            # Weighting function: Cubic Spline or similar local support
            wi = np.zeros(num_hf_samples)
            mask = di <= 1.0
            wi[mask] = 1 - 6 * di[mask]**2 + 8 * di[mask]**3 - 3 * di[mask]**4

            W = np.diag(wi)

            # Weighted Least Square system
            # LHS: (num_basis, num_basis), RHS: (num_basis, num_outputs)
            lhs = self.p_train_.T @ W @ self.p_train_
            rhs = self.p_train_.T @ W @ self.y_hf_train_

            # Solve for coefficients A (num_basis, num_outputs)
            try:
                coeffs = np.linalg.solve(lhs + 1e-8 * np.eye(num_basis), rhs)
                # Prediction: p(x_test) * A
                y_pred_scaled[i, :] = p_test_[i : i + 1, :] @ coeffs
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if singular
                coeffs = np.linalg.pinv(lhs) @ rhs
                y_pred_scaled[i, :] = p_test_[i : i + 1, :] @ coeffs

        logger.info(f'{sl.g}MFS-MLS prediction completed.{sl.q}')

        # 5. Inverse Scaling: Convert back to original space
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        # 6.1 Inference Mode
        if y_test is None: return y_pred

        # 6.2 Evaluation Mode
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        metrics = {'r2': r2, 'mse': mse, 'rmse': rmse}

        return y_pred, metrics


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == '__main__':
    np.random.seed(42)

    def target_hf(x: np.ndarray) -> np.ndarray:
        return np.hstack([x * np.sin(x), 0.1 * x**2 + np.cos(x)])

    def target_lf(x: np.ndarray) -> np.ndarray:
        return 0.8 * target_hf(x) - 0.5 * x

    # Training samples
    x_lf_train = np.linspace(0, 10, 50).reshape(-1, 1)
    y_lf_train = target_lf(x_lf_train)

    x_hf_train = np.linspace(0, 10, 10).reshape(-1, 1)
    y_hf_train = target_hf(x_hf_train)

    # Testing samples
    x_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test = target_hf(x_test)

    # Instantiate model
    model = MFSMLS()

    # Train model
    model.fit(x_lf_train, y_lf_train, x_hf_train, y_hf_train)

    # Test model
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Log results
    logger.info(f'Testing R2: {sl.m}{test_metrics['r2']:.9f}{sl.q}')
    logger.info(f'Testing MSE: {sl.m}{test_metrics['mse']:.9f}{sl.q}')
    logger.info(f'Testing RMSE: {sl.m}{test_metrics['rmse']:.9f}{sl.q}')
