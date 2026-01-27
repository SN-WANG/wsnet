# Modified Multi-Fidelity Surrogate Model based on RBF with Adaptive Scale Factor (MMFS)
# Paper reference: https://doi.org/10.1186/s10033-022-00742-z
# Paper author: Yin Liu, Shuo Wang, Qi Zhou, Liye Lv, Wei Sun, Xueguan Song
# Code author: Shengning Wang

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from scipy.linalg import pinv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple, Union, Optional, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets import RBF
from wsnet.utils import sl, logger


class MMFS:
    """
    Modified Multi-Fidelity Surrogate Model based on RBF with Adaptive Scale Factor (MMFS).

    Implements the MMFS algorithm which utilizes a Low-Fidelity (LF) RBF model to approximate
    the trend and constructs a High-Fidelity (HF) correction model. The correction involves
    an adaptive scale factor and a deviation term, solved via a comprehensive correction matrix.

    Key Features:
    - Adaptive scaling and shifting based on HF sample locations.
    - Multiquadric (MQ) kernel usage for the HF correction layer.
    - Leave-One-Out Cross-Validation (LOOCV) for optimizing the HF shape parameter (sigma).
    - Generalized Inverse (Minimum Norm Solution) for solving model coefficients.

    Attributes:
    - lf_model (RBF): The Low-Fidelity surrogate model instance.
    - sigma_bounds (Tuple[float, float]): Search bounds for the RBF shape parameter sigma.
    - scaler_x (StandardScaler): Preprocessor for HF features.
    - scaler_y (StandardScaler): Preprocessor for HF targets.
    - x_hf_train_ (np.ndarray): Scaled HF training inputs (num_hf, num_features).
    - y_hf_train_ (np.ndarray): Scaled HF training targets (num_hf, num_outputs).
    - beta_ (List[np.ndarray]): List of coefficient vectors [beta_1, ..., beta_m] for each output dimension.
    - sigma_ (float): Optimized shape parameter for the HF correction RBF.
    - is_fitted (bool): Status flag.
    """

    def __init__(self, lf_model_params: Optional[Dict] = None, sigma_bounds: Tuple[float, float] = (0.01, 10.0)):
        """
        Initializes the MMFS model configuration.

        Args:
        - lf_model_params (Optional[Dict]): Dictionary of kwargs to pass to the base RBF class for the LF model.
        - sigma_bounds (Tuple[float, float]): The lower and upper bounds for optimizing the shape parameter sigma.
        """
        # Initialize the LF RBF model
        params = lf_model_params if lf_model_params is not None else {'num_centers': 50, 'gamma': 0.1, 'alpha': 0.0}
        self.lf_model = RBF(**params)

        self.sigma_bounds = sigma_bounds

        # Scalers for normalization of HF data
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Model state
        self.x_hf_train_: Optional[np.ndarray] = None
        self.y_hf_train_: Optional[np.ndarray] = None
        self.beta_: List[np.ndarray] = []  # Stores beta per output dim
        self.sigma_: float = 1.0
        self.is_fitted: bool = False

    def _multiquadric_kernel(self, dists: np.ndarray, sigma: float) -> np.ndarray:
        """
        Computes the Multiquadric (MQ) basis function: phi(r) = sqrt(r^2 + sigma^2).

        Args:
        - dists (np.ndarray): Euclidean distance matrix (N, M).
        - sigma (float): Shape parameter.

        Returns:
        - np.ndarray: Kernel matrix (N, M).
        """
        return np.sqrt(dists**2 + sigma**2)

    def _construct_expansion_matrix(self, phi: np.ndarray, y_lf: np.ndarray) -> np.ndarray:
        """
        Constructs the expansion matrix H.
        H = [diag(y_lf) * phi, phi]

        Args:
        - phi (np.ndarray): RBF correlation matrix (num_samples, num_centers).
        - y_lf (np.ndarray): Low-fidelity predictions corresponding to samples (num_samples, 1).

        Returns:
        - np.ndarray: Expansion matrix H with shape (num_samples, 2 * num_centers).
        """
        # Element-wise multiplication of column vector y_lf with matrix phi behaves like diag(y_lf) @ phi
        # y_lf shape: (N, 1), phi shape: (N, N_c) -> result: (N, N_c)
        scaled_phi = y_lf * phi
        return np.concatenate([scaled_phi, phi], axis=1)

    def _loocv_error(self, sigma: float, dist_matrix: np.ndarray, y_lf_at_hf: np.ndarray, y_hf: np.ndarray) -> float:
        """
        Calculates the Leave-One-Out Cross-Validation (LOOCV) error for a specific sigma.
        Iterates through all outputs to calculate a summed error.

        Args:
        - sigma (float): The candidate shape parameter.
        - dist_matrix (np.ndarray): Pre-computed distance matrix of HF samples (num_hf, num_hf).
        - y_lf_at_hf (np.ndarray): Scaled LF predictions at HF sites (num_hf, num_outputs).
        - y_hf (np.ndarray): Scaled HF targets (num_hf, num_outputs).

        Returns:
        - float: The total Mean Squared Error across all LOO folds and outputs.
        """
        num_hf, num_outputs = y_hf.shape
        total_error = 0.0

        # Avoid singularity
        if sigma < 1e-6: return 1e10

        # Compute full kernel matrix for this sigma
        phi_full = self._multiquadric_kernel(dist_matrix, sigma)

        # LOOCV Loop
        for i in range(num_hf):
            # 1. Split indices
            train_idx = np.delete(np.arange(num_hf), i)
            test_idx = np.array([i])

            # 2. Extract sub-matrices
            # Training phi: submatrix of shape (num_hf - 1, num_hf - 1)
            phi_train = phi_full[np.ix_(train_idx, train_idx)]

            # Testing phi: distance from test point to training points (1, num_hf - 1)
            phi_test = phi_full[np.ix_(test_idx, train_idx)]

            # 3. Solve for each output dimension
            for m in range(num_outputs):
                y_c_train = y_lf_at_hf[train_idx, m:m+1]
                y_e_train = y_hf[train_idx, m:m+1]

                # Construct H matrix (Eq 8) for training set
                H_train = self._construct_expansion_matrix(phi_train, y_c_train)

                # Solve beta using Generalized Inverse (Eq 9)
                # beta = H.T * (H * H.T)^-1 * Y
                # This corresponds to the minimum norm solution for under-determined systems
                # Or standard least squares for over-determined.
                # Here H is (N-1, 2*(N-1)), so it is 'fat' (under-determined).
                # We use pinv which handles this automatically.
                try:
                    beta = pinv(H_train) @ y_e_train
                except np.linalg.LinAlgError:
                    total_error += 1e10
                    continue

                # 4. Predict validation point
                y_c_test = y_lf_at_hf[test_idx, m:m+1]
                H_test = self._construct_expansion_matrix(phi_test, y_c_test)

                y_pred = H_test @ beta
                y_true = y_hf[test_idx, m:m+1]

                total_error += np.sum((y_pred - y_true) ** 2)

        return total_error / num_hf

    def fit(self, x_lf: np.ndarray, y_lf: np.ndarray, x_hf: np.ndarray, y_hf: np.ndarray) -> None:
        """
        Trains the MMFS model using both Low-Fidelity and High-Fidelity datasets.

        Process:
        1. Train LF model (RBF).
        2. Optimize shape parameter (sigma) via LOOCV on HF data.
        3. Solve for adaptive scale coefficients (beta) using the full HF dataset.

        Args:
        - x_lf (np.ndarray): Low-Fidelity inputs (num_lf, num_features).
        - y_lf (np.ndarray): Low-Fidelity targets (num_lf, num_outputs).
        - x_hf (np.ndarray): High-Fidelity inputs (num_hf, num_features).
        - y_hf (np.ndarray): High-Fidelity targets (num_hf, num_outputs).
        """
        # ensure 2D arrays
        if x_lf.ndim == 1: x_lf = x_lf.reshape(-1, 1)
        if y_lf.ndim == 1: y_lf = y_lf.reshape(-1, 1)
        if x_hf.ndim == 1: x_hf = x_hf.reshape(-1, 1)
        if y_hf.ndim == 1: y_hf = y_hf.reshape(-1, 1)

        logger.info(f"training MMFS (LF=RBF, SigmaBounds={self.sigma_bounds})...")

        # 1. fit LF model
        self.lf_model.fit(x_lf, y_lf)

        # 2. scale HF data
        self.x_hf_train_ = self.scaler_x.fit_transform(x_hf)
        self.y_hf_train_ = self.scaler_y.fit_transform(y_hf)

        # 3. get LF predictions at HF points (Y_c)
        y_lf_at_hf_raw = self.lf_model.predict(x_hf)
        if isinstance(y_lf_at_hf_raw, tuple):
            y_lf_at_hf_raw = y_lf_at_hf_raw[0]
        y_lf_at_hf_scaled = self.scaler_y.transform(y_lf_at_hf_raw)

        # 4. optimize sigma via LOOCV (Step 3 in paper, Eq 10)
        dist_matrix = cdist(self.x_hf_train_, self.x_hf_train_, metric='euclidean')

        logger.info('optimizing MMFS shape parameter (sigma) via LOOCV...')
        res = minimize_scalar(
            fun=self._loocv_error,
            bounds=self.sigma_bounds,
            args=(dist_matrix, y_lf_at_hf_scaled, self.y_hf_train_),
            method='bounded'
        )
        self.sigma_ = res.x
        logger.info(f'{sl.y}optimal sigma found: {self.sigma_:.4f}{sl.q}')

        # 5. compute final coefficients (beta) for each output
        phi_train = self._multiquadric_kernel(dist_matrix, self.sigma_)

        self.beta_ = []
        num_outputs = self.y_hf_train_.shape[1]

        for m in range(num_outputs):
            y_c = y_lf_at_hf_scaled[:, m:m+1]
            y_e = self.y_hf_train_[:, m:m+1]

            # expansion matrix H (Eq 8)
            H = self._construct_expansion_matrix(phi_train, y_c)

            # solve beta (Eq 9: generalized inverse)
            beta_m = pinv(H) @ y_e
            self.beta_.append(beta_m)

        self.is_fitted = True
        logger.info(f'{sl.g}MMFS training completed.{sl.q}')

    def predict(self, x_test: np.ndarray, y_test: Optional[np.ndarray] = None
                ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Predicts using the trained MMFS model.

        Process:
        1. Predict LF response at test points.
        2. Construct cross-correlation matrix between test points and HF training points.
        3. Apply the expansion matrix transformation and correction coefficients.

        Args:
        - x_test (np.ndarray): Test feature data (num_samples, num_features).
        - y_test (Optional[np.ndarray]): Test target data (num_samples, num_outputs).

        Returns:
        - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
            - If y_test is available: Returns (y_pred, metrics).
            - If y_test is None: Returns y_pred.
            - y_pred shape: (num_samples, num_outputs).
        """
        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        if not self.is_fitted:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        logger.info(f'{sl.g}predicting MMFS...{sl.q}')

        num_samples = x_test.shape[0]
        num_outputs = len(self.beta_)

        # 1. Scale Test Inputs
        x_test_scaled = self.scaler_x.transform(x_test)

        # 2. Get LF predictions at Test points (y_c in Eq 11)
        y_lf_at_test_raw = self.lf_model.predict(x_test)
        if isinstance(y_lf_at_test_raw, tuple):
            y_lf_at_test_raw = y_lf_at_test_raw[0]

        # Scale LF predictions to match the training latent space
        y_lf_at_test_scaled = self.scaler_y.transform(y_lf_at_test_raw)

        # 3. Calculate Cross-Distances and Phi Matrix (Step 5 in paper)
        dists_test = cdist(x_test_scaled, self.x_hf_train_, metric='euclidean')
        phi_test = self._multiquadric_kernel(dists_test, self.sigma_)

        # 4. Compute Predictions (Step 6 in paper)
        y_pred_scaled = np.zeros((num_samples, num_outputs))

        for m in range(num_outputs):
            y_c_test = y_lf_at_test_scaled[:, m:m+1]
            beta_m = self.beta_[m]

            # Construct Expansion Matrix H* (Eq 12)
            # H* = [diag(y_c_test) * phi_test, phi_test]
            H_star = self._construct_expansion_matrix(phi_test, y_c_test)

            # Eq 11: Y(x) = H* . beta
            y_pred_scaled[:, m:m+1] = H_star @ beta_m

        # 5. Inverse Scale Predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)

        logger.info(f'{sl.g}MMFS prediction completed.{sl.q}')

        # 6.1 Inference Mode
        if y_test is None:
            return y_pred

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
    # Reproducibility
    np.random.seed(42)

    # --- Benchmark Functions (Forretal Function from paper Eq 14 & 15) ---
    def forretal_hf(x: np.ndarray) -> np.ndarray:
        return (6 * x - 2)**2 * np.sin(12 * x - 4)

    def forretal_lf(x: np.ndarray) -> np.ndarray:
        # Variant A: y_c = 0.5 * y_e + 10(x - 0.5) - 5
        return 0.5 * forretal_hf(x) + 10 * (x - 0.5) - 5

    # --- Data Generation ---
    # Training: Low Fidelity (Dense)
    x_lf_train = np.linspace(0, 1, 50).reshape(-1, 1)
    y_lf_train = forretal_lf(x_lf_train)

    # Training: High Fidelity (Sparse)
    x_hf_train = np.linspace(0, 1, 8).reshape(-1, 1)
    y_hf_train = forretal_hf(x_hf_train)

    # Testing
    x_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_test = forretal_hf(x_test)

    # --- Model Execution ---
    # Instantiate MMFS
    model = MMFS(lf_model_params={'num_centers': 20, 'gamma': 10.0})

    # Train
    model.fit(x_lf_train, y_lf_train, x_hf_train, y_hf_train)

    # Predict
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Output Results
    logger.info(f'--- Forretal Function Results ---')
    logger.info(f'Testing R2: {sl.m}{test_metrics["r2"]:.9f}{sl.q}')
    logger.info(f'Testing MSE: {sl.m}{test_metrics["mse"]:.9f}{sl.q}')
    logger.info(f'Testing RMSE: {sl.m}{test_metrics["rmse"]:.9f}{sl.q}')
