# Efficient Global Optimization (EGO) based on Expected Improvement (EI)
# Paper reference: https://doi.org/10.1023/A:1008306431147
# Paper author: Donald R. Jones, Matthias Schonlau, William J. Welch
# Code author: Shengning Wang

import os
import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from typing import Dict, Tuple, Union, Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets import KRG
from wsnet.utils import sl, logger


class EGO:
    """
    Efficient Global Optimization (EGO) Surrogate Model.

    Implements the Expected Improvement (EI) acquisition function for sequential sampling.
    Uses a Kriging (Gaussian Process) model to estimate the mean and variance of the 
    objective function and proposes new sampling points that balance exploration and exploitation.

    Attributes:
        bounds (np.ndarray): The search space bounds (num_features, 2).
        target_index (int): The index of the output dimension to minimize (for MIMO handling).
        num_restarts (int): Number of restarts for the EI optimization to avoid local optima.
        krg_model (KRG): The underlying Kriging surrogate model.
        y_min_ (float): The current minimum observed value (in the target dimension).
        is_fitted (bool): Status flag.
    """

    def __init__(self, bounds: Union[List[float], np.ndarray], target_index: int = 0,
                 num_restarts: int = 10, krg_params: Optional[Dict] = None):
        """
        Initializes the EGO model configuration.

        Args:
            bounds (Union[List[float], np.ndarray]): Lower and upper bounds for each feature. shape: (num_features, 2).
            target_index (int): Index of the output dimension to be optimized (default is 0).
            num_restarts (int): Number of random restarts for maximizing the EI function.
            krg_params (Optional[Dict]): Dictionary of kwargs to pass to the KRG class.
        """
        self.bounds = np.array(bounds)
        if self.bounds.ndim == 1:
            # handle case where single feature bounds might be passed flattened
            self.bounds = self.bounds.reshape(-1, 2)

        self.target_index = target_index
        self.num_restarts = num_restarts

        # initialize the surrogate model
        params = krg_params if krg_params is not None else {
            "poly": "constant", "kernel": "gaussian", "theta0": 10.0, "theta_bounds": (1e-6, 100.0)
        }
        self.krg_model = KRG(**params)

        # model state
        self.y_min_: float = np.inf
        self.is_fitted: bool = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the underlying Kriging model and updates the current best observation.

        Args:
            x_train (np.ndarray): Training feature data (num_samples, num_features).
            y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        # ensure 2D arrays
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        # check dimensions
        if x_train.shape[1] != self.bounds.shape[0]:
            raise ValueError(f"dimension mismatch: x_train has {x_train.shape[1]} features, "
                             f"but bounds has {self.bounds.shape[0]} dimensions.")
        
        if y_train.shape[1] <= self.target_index:
            raise ValueError(f"target_index {self.target_index} is out of bounds for "
                             f"y_train with {y_train.shape[1]} outputs.")

        logger.info(f"training EGO (target_index={self.target_index})...")

        # 1. train the surrogate model
        # note: KRG handles its own normalization internally
        self.krg_model.fit(x_train, y_train)

        # 2. update current best (minimum) value for the target dimension
        # we work with raw values here because EI scaling usually happens relative to real physics
        self.y_min_ = np.min(y_train[:, self.target_index])

        self.is_fitted = True
        logger.info(f"{sl.g}EGO training completed.{sl.q} Current y_min: {sl.m}{self.y_min_:.6f}{sl.q}")

    def predict(self, x_test: np.ndarray, y_test: Optional[np.ndarray] = None
                ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
        """
        Predicts mean and mean-squared-error using the underlying Kriging model.

        Args:
            x_test (np.ndarray): Test feature data (num_samples, num_features)
            y_test (np.ndarray, Optional): Test target data (num_samples, num_outputs)

        Returns:
            Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
                If y_test is available: Returns a Tuple: (y_pred, mse_pred, metrics)
                If y_test is None: Returns a Tuple: (y_pred, mse_pred)

            y_pred (np.ndarray): Predicted target values
            mse_pred (np.ndarray): Predicted mse values
            metrics (Dict[str, float]): Dictionary of evaluation metrics
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Please call fit() first.")

        # leverage KRG's prediction method
        return self.krg_model.predict(x_test, y_test)

    def compute_ei(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the Expected Improvement (EI) at given points.

        Equation: EI(x) = (y_min - y_hat) * CDF(u) + s * PDF(u)
        where u = (y_min - y_hat) / s

        Args:
            x (np.ndarray): Input points to evaluate (num_samples, num_features).

        Returns:
            np.ndarray: Expected Improvement values (num_samples, 1).
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # 1. get mean and variance from Kriging
        y_pred, mse_pred = self.predict(x)

        # 2. extract target dimension data
        mu = y_pred[:, self.target_index].reshape(-1, 1)
        sigma2 = mse_pred[:, self.target_index].reshape(-1, 1)

        # handle numerical stability for sqrt
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))

        # 3. calculate improvement potential u
        # we want to minimize, so improvement is y_min - prediction
        improvement = self.y_min_ - mu
        u = improvement / sigma

        # 4. compute expected improvement
        cdf_u = norm.cdf(u)
        pdf_u = norm.pdf(u)

        ei = improvement * cdf_u + sigma * pdf_u

        # zero out EI where sigma is effectively zero (observed points)
        ei[sigma < 1e-6] = 0.0

        return ei

    def propose(self, num_points: int = 1) -> np.ndarray:
        """
        Proposes new sampling points by maximizing the Expected Improvement.

        Uses L-BFGS-B optimization with random restarts to handle the multi-modal nature of the EI function.

        Args:
            num_points (int): Number of points to propose.
                              Currently supports sequential proposal (default=1). For batch, it iterates logic.

        Returns:
            np.ndarray: Proposed sample points (num_points, num_features).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Please call fit() first.")

        logger.info(f"optimizing acquisition function (EI)...")

        num_features = self.bounds.shape[0]
        proposed_points = []

        # optimization bounds format for scipy
        scipy_bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        # define negative ei function for minimization
        def min_obj(x_vec):
            # scipy passes 1d array, reshape for batch processing
            x_reshaped = x_vec.reshape(1, -1)
            ei_val = self.compute_ei(x_reshaped)
            return -float(ei_val.flatten()[0])

        for _ in range(num_points):
            best_x = None
            best_ei = -np.inf

            # multi-start optimization to find global max of EI
            for _ in range(self.num_restarts):
                # random starting point within bounds
                x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=num_features)

                res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")

                # invert sign because we minimized negative EI
                current_ei = -res.fun

                if current_ei > best_ei:
                    best_ei = current_ei
                    best_x = res.x

            proposed_points.append(best_x)

        logger.info(f"{sl.g}EGO proposal completed.{sl.q}")

        return np.array(proposed_points)


# ======================================================================
# Example Usage
# ======================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # define objective function
    def branin(x):
        # x shape: (num_samples, 2)
        x1, x2 = x[:, 0], x[:, 1]
        a, b, c, r, s, t = 1.0, 5.1 / (4.0 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8.0 * np.pi)
        return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s

    # define bounds for x1 [-5, 10] and x2 [0, 15]
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])

    # initial random sampling
    x_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(20, 2))
    y_train = branin(x_train)

    # independent test set for metrics evaluation
    x_test = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(100, 2))
    y_test = branin(x_test)

    # instantiate model
    model = EGO(bounds=bounds)

    logger.info(f"{sl.b}--- Phase 1: Baseline Training ---{sl.q}")
    model.fit(x_train, y_train)
    _, _, metrics_initial = model.predict(x_test, y_test)

    logger.info(f"initial R2: {sl.m}{metrics_initial["r2"]:.6f}{sl.q}")
    logger.info(f"initial MSE: {sl.m}{metrics_initial["mse"]:.6f}{sl.q}")
    logger.info(f"initial RMSE: {sl.m}{metrics_initial["rmse"]:.6f}{sl.q}")

    # sequential sampling (the "Infill" step)
    logger.info(f"{sl.b}--- Phase 2: Sequential Sampling (EI) ---{sl.q}")
    x_new = model.propose(num_points=1)
    y_new = branin(x_new)

    # update dataset
    if x_train.ndim == 1: x_train = x_train.reshape(-1, 1)
    if x_new.ndim == 1: x_new = x_new.reshape(-1, 1)
    if y_train.ndim == 1: y_train = y_train.reshape(-1, 1)
    if y_new.ndim == 1: y_new = y_new.reshape(-1, 1)

    x_train_updated = np.vstack([x_train, x_new])
    y_train_updated = np.vstack([y_train, y_new])

    # re-fit and compare metrics
    logger.info(f"{sl.b}--- Phase 3: Post-Update Evaluation ---{sl.q}")
    model.fit(x_train_updated, y_train_updated)
    _, _, metrics_updated = model.predict(x_test, y_test)

    logger.info(f"updated R2: {sl.m}{metrics_updated["r2"]:.6f}{sl.q}")
    logger.info(f"updated MSE: {sl.m}{metrics_updated["mse"]:.6f}{sl.q}")
    logger.info(f"updated RMSE: {sl.m}{metrics_updated["rmse"]:.6f}{sl.q}")
