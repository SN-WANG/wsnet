# Support Vector Regression (SVR) Surrogate Model
# Author: Shengning Wang

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import Tuple, Optional, Literal

from wsnet.utils.scaler import StandardScalerNP


class SVR:
    """
    Support Vector Regression (SVR) using Dual Optimization with epsilon-insensitive loss.
    """

    def __init__(self, kernel: Literal["rbf", "linear"] = "rbf", gamma: Optional[float] = None,
                 C: float = 1.0, epsilon: float = 0.1):
        """
        Initializes the SVR configuration.

        Args:
            kernel (str): Kernel type ("rbf" or "linear").
            gamma (Optional[float]): Kernel coefficient for rbf.
            C (float): Regularization parameter (penalty).
            epsilon (float): Epsilon-tube width (tolerance margin).
        """
        # parameters
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon

        # scalers
        self.scaler_x = StandardScalerNP()
        self.scaler_y = StandardScalerNP()

        # model state
        self.support_vectors_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.fitted = False

    # ------------------------------------------------------------------

    def _build_features(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """
        Constructs SVR feature matrix.

        Args:
            x1 (np.ndarray): Shape (num_samples_1, input_dim)
            x2 (np.ndarray): Shape (num_samples_2, input_dim)

        Returns:
            np.ndarray: Feature matrix. Shape: (num_samples_1, num_samples_2)
        """
        if self.kernel == "linear":
            return x1 @ x2.T
        elif self.kernel == "rbf":
            # pairwise squared euclidean distance
            dists = np.sum(x1**2, axis=1, keepdims=True) + np.sum(x2**2, axis=1) - 2.0 * (x1 @ x2.T)
            return np.exp(-self.gamma * dists)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    # ------------------------------------------------------------------

    def _solve_dual(self, phi: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solves the dual QP problem for a single output dimension.

        Minimize: 0.5 * beta^T @ phi @ beta - beta^T @ y + epsilon * |beta|
        Subject to: sum(beta) = 0, -C <= beta <= C
        Note: beta = alpha - alpha*

        Args:
            phi (np.ndarray): Feature matrix (num_samples, num_samples).
            y (np.ndarray): Target vector (num_samples,).

        Returns:
            Tuple[np.ndarray, float]: Dual coefficients (beta) and intercept (bias).
        """
        num_samples = y.shape[0]

        # optimization objective (using dual form derivation)
        # we optimize x = [alpha; alpha*] to adhere to standard QP form strictly
        # variables: x = [alpha_1, ..., alpha_n, alpha*_1, ..., alpha*_n] (size 2*n)

        def objective(x):
            alpha, alpha_star = x[:num_samples], x[num_samples:]
            beta = alpha - alpha_star

            # quadratic term: 0.5 * beta.T @ phi @ beta
            term1 = 0.5 * beta @ phi @ beta

            # linear term: epsilon * sum(alpha + alpha*) - beta.T @ y
            term2 = self.epsilon * np.sum(alpha + alpha_star) - y @ beta

            return term1 + term2

        # manually providing jacobian to speed up SLSQP by avoiding finite difference
        def jacobian(x):
            alpha, alpha_star = x[:num_samples], x[num_samples:]
            beta = alpha - alpha_star
            grad_base = phi @ beta - y
            grad_alpha = grad_base + self.epsilon
            grad_alpha_star = -grad_base + self.epsilon
            return np.concatenate([grad_alpha, grad_alpha_star])

        # constraints: sum(alpha - alpha*) = 0
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:num_samples] - x[num_samples:])}]

        # bounds: 0 <= alpha, alpha* <= C
        bounds = Bounds(np.zeros(2 * num_samples), np.full(2 * num_samples, self.C))

        # initial guess
        x0 = np.zeros(2 * num_samples)

        # solve QP
        res = minimize(fun=objective, x0=x0, method="SLSQP", jac=jacobian, bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-6, "maxiter": 200})

        if not res.success:
            # warning: optimization might fail on hard problems, fallback or log could be added
            pass

        alpha_final, alpha_star_final = np.split(res.x, 2)
        beta = alpha_final - alpha_star_final

        # threshold small values to zero (sparsity)
        beta[np.abs(beta) < 1e-5] = 0.0

        # compute intercept (bias) using support vectors
        # sv_indices: 0 < alpha < C  OR  0 < alpha* < C
        # robustly: use free support vectors (not at bounds)
        sv_mask = (np.abs(beta) > 1e-5) & (np.abs(beta) < self.C - 1e-5)

        if np.any(sv_mask):
            # b = y - phi @ beta - sign(beta) * epsilon
            biases = y[sv_mask] - phi[sv_mask] @ beta - np.sign(beta[sv_mask]) * self.epsilon

            # take mean for numerical stability
            intercept = np.mean(biases)
        else:
            # fallback if no free support vectors are found
            intercept = 0.0

        return beta, intercept

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

        if self.gamma is None:
            x_var = x_scaled.var()
            self.gamma = 1.0 / (x_scaled.shape[1] * x_var) if x_var > 0 else 1.0

        phi = self._build_features(x_scaled, x_scaled)

        # storage for multi-target
        num_samples, target_dim = y_scaled.shape
        self.dual_coef_ = np.zeros((num_samples, target_dim))
        self.intercept_ = np.zeros(target_dim)
        self.support_vectors_ = x_scaled  # store scaled support vectors

        # fit per target dimension
        for d in range(target_dim):
            beta, bias = self._solve_dual(phi, y_scaled[:, d])
            self.dual_coef_[:, d] = beta
            self.intercept_[d] = bias

        self.fitted = True

    # ------------------------------------------------------------------

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        """
        Performs model prediction.

        Args:
            x_pred (np.ndarray): Prediction inputs. Shape: (num_samples, input_dim).

        Returns:
            np.ndarray: Prediction targets. Shape: (num_samples, target_dim).
        """
        if not self.fitted:
            raise RuntimeError("Model has not been fitted.")

        if x_pred.ndim == 1:
            x_pred = x_pred[:, None]

        x_scaled = self.scaler_x.transform(x_pred)
        phi = self._build_features(x_scaled, self.support_vectors_)

        y_scaled = phi @ self.dual_coef_ + self.intercept_
        y_pred = self.scaler_y.inverse_transform(y_scaled)

        if y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()

        return y_pred
