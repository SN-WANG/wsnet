# Infill Criteria for Sequential Sampling
# Author: Shengning Wang

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from typing import Union, List

from wsnet.models.classical.krg import KRG


class Infill:
    """
    Infill Criteria for Sequential Sampling.

    Provides a suite of acquisition functions (infill criteria) to guide
    sequential sampling strategies using a pre-trained surrogate model.

    Supported Criteria:
        "mse": Mean Squared Error (Pure Exploration / Max Entropy).
        "poi": Probability of Improvement.
        "ei": Expected Improvement (Balances exploration and exploitation).
        "lcb": Lower Confidence Bound.

    Attributes:
        model (KRG): The pre-trained Kriging surrogate model.
        bounds (np.ndarray): Search space bounds (num_features, 2).
        target_index (int): Index of the output dimension to optimize.
        y_min (float): The current best (minimum) observed value.
        criterion_name (str): Name of the active criterion.
        criterion_func (Callable): The active criterion function.
        num_restarts (int): Number of optimizer restarts for proposal.
    """

    def __init__(self, model: KRG, bounds: Union[List[float], np.ndarray], y_train: np.ndarray,
                 criterion: str = "ei", target_index: int = 0, num_restarts: int = 10,
                 kappa: float = 2.0):
        """
        Initializes the Infill strategy configuration.

        Args:
            model (KRG): A fitted Kriging model instance.
            bounds (Union[List[float], np.ndarray]): Search space bounds. shape: (num_features, 2).
            y_train (np.ndarray): The existing training target values, used to find current y_min.
            criterion (str): The acquisition function to use. Options: "mse", "poi", "ei", "lcb".
            target_index (int): Index of the output dimension to optimize (for Multi-output models).
            num_restarts (int): Number of random restarts for the optimizer.
            kappa (float): Exploration parameter for LCB (default=2.0).
        """
        # input validation
        if not hasattr(model, "beta") or model.beta is None:
            raise RuntimeError("provided KRG model is not fitted.")

        self.model = model
        self.bounds = np.array(bounds)
        if self.bounds.ndim == 1:
            self.bounds = self.bounds.reshape(-1, 2)

        self.target_index = target_index
        self.num_restarts = num_restarts
        self.kappa = kappa

        # determine current best solution (minimization)
        self.y_min = np.min(y_train[:, self.target_index])

        # dispatch logic for criterion function
        # all criteria are implemented to return a "utility" to be MAXIMIZED
        criterion_map = {
            "mse": self._crit_mse,
            "poi": self._crit_poi,
            "ei": self._crit_ei,
            "lcb": self._crit_lcb
        }

        if criterion.lower() not in criterion_map:
            raise ValueError(f"unknown criterion: '{criterion}'. available: {list(criterion_map.keys())}")

        self.criterion_name = criterion.lower()
        self.criterion_func = criterion_map[self.criterion_name]

    # ======================================================================
    # Criterion Methods (Acquisition Functions)
    # ======================================================================

    def _crit_mse(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Mean Squared Error (MSE) criterion.
        Equivalent to Maximum Entropy sampling for Gaussian processes.
        Utility = sigma^2

        Args:
            mu (np.ndarray): Predicted mean. shape: (num_samples, 1).
            sigma (np.ndarray): Predicted std dev. shape: (num_samples, 1).

        Returns:
            np.ndarray: Utility values. shape: (num_samples, 1).
        """
        return sigma ** 2

    def _crit_poi(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Probability of Improvement (POI) criterion.
        Utility = P(y < y_min)

        Args:
            mu (np.ndarray): Predicted mean. shape: (num_samples, 1).
            sigma (np.ndarray): Predicted std dev. shape: (num_samples, 1).

        Returns:
            np.ndarray: Utility values. shape: (num_samples, 1).
        """
        with np.errstate(divide="ignore"):
            z = (self.y_min - mu) / sigma

        poi = norm.cdf(z)
        poi[sigma < 1e-9] = 0.0
        return poi

    def _crit_ei(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Expected Improvement (EI) criterion.
        Utility = (y_min - mu) * CDF(z) + sigma * PDF(z)

        Args:
            mu (np.ndarray): Predicted mean. shape: (num_samples, 1).
            sigma (np.ndarray): Predicted std dev. shape: (num_samples, 1).

        Returns:
            np.ndarray: Utility values. shape: (num_samples, 1).
        """
        with np.errstate(divide="ignore"):
            improvement = self.y_min - mu
            z = improvement / sigma

        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def _crit_lcb(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Lower Confidence Bound (LCB) criterion.
        We want to minimize (mu - kappa * sigma).
        To fit the maximization framework, Utility = -1 * (mu - kappa * sigma).

        Args:
            mu (np.ndarray): Predicted mean. shape: (num_samples, 1).
            sigma (np.ndarray): Predicted std dev. shape: (num_samples, 1).

        Returns:
            np.ndarray: Utility values (negated LCB). shape: (num_samples, 1).
        """
        return -1.0 * (mu - self.kappa * sigma)

    # ======================================================================
    # Core Logic
    # ======================================================================

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the active infill criterion at specific points.

        Args:
            x (np.ndarray): Input points. shape: (num_samples, num_features).

        Returns:
            np.ndarray: Utility values. shape: (num_samples, 1).
        """
        # 1. predict using the pre-trained KRG model
        y_pred, mse_pred = self.model.predict(x)

        # 2. extract target dimension statistics
        # shape checks are handled by KRG, but we ensure (n, 1) here
        mu = y_pred[:, self.target_index].reshape(-1, 1)
        var = mse_pred[:, self.target_index].reshape(-1, 1)

        # clamp variance for numerical stability
        sigma = np.sqrt(np.maximum(var, 1e-12))

        # 3. compute utility
        return self.criterion_func(mu, sigma)

    def propose(self) -> np.ndarray:
        """
        Proposes a single new sampling point by maximizing the active criterion.

        Uses L-BFGS-B with random restarts to avoid local optima in the acquisition surface.

        Returns:
            np.ndarray: The proposed sample point. shape: (1, num_features).
        """
        num_features = self.bounds.shape[0]

        # scipy optimization requires 1D arrays for bounds
        scipy_bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        # define objective: minimize negative utility
        def min_obj(x_vec: np.ndarray) -> float:
            x_reshaped = x_vec[None, :]
            utility = self.evaluate(x_reshaped)
            return -float(utility.flatten()[0])

        best_x = None
        best_utility = -np.inf

        # multi-start optimization
        for i in range(self.num_restarts):
            # random initialization within bounds
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=num_features)

            try:
                res = minimize(min_obj, x0=x0, bounds=scipy_bounds, method="L-BFGS-B")

                # invert sign to get actual utility
                curr_utility = -res.fun

                if curr_utility > best_utility:
                    best_utility = curr_utility
                    best_x = res.x
            except Exception as e:
                # catch linalg errors in krg prediction during optimization steps
                continue

        if best_x is None:
             # fallback if optimization fails completely (rare)
            warnings.warn("optimization failed to converge, returning random point.", RuntimeWarning)
            best_x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=num_features)

        return best_x[None, :]
