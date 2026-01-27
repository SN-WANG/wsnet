# Two-Stage Adaptive Hybrid Surrogate (T-AHS) Model
# Paper reference: https://doi.org/10.1115/1.4039128
# Paper author: Xueguan Song, Liye Lv, Jieling Li, Wei Sun, Jie Zhang
# Code author: Shengning Wang

import os
import sys
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Union, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets import PRS, RBF, KRG, SVR
from wsnet.utils import sl, logger


class TAHS:
    """
    Two-Stage Adaptive Hybrid Surrogate (T-AHS) Model.

    A robust ensemble method that:
    1. Filters out poor-performing surrogate model using Cross-Validation (Stage 1).
    2. Constructs a weighted average prediction where weights are adaptively determined
       by the local uncertainty estimated via a Gaussian Process (Stage 2).

    Attributes:
    - threshold (float): The normalized CV error threshold [0, 1] for model selection.
                         Models with normalized error > threshold are discarded.
    - models_pool (List[object]): The list of instantiated surrogate model objects.
    - activate_indices_ (List[int]): Indices of models that passed the filtering stage.
    - baseline_index_ (int): Index of the best performing model (baseline).
    - uncertainty_model_ (KRG): A dedicated Kriging model used solely for estimating
                                local process variance (s^2) for weighting.
    - is_fitted (bool): Status flag.
    """

    def __init__(self, threshold: float = 0.5,
                 prs_params: Optional[Dict] = {'degree': 3, 'alpha': 0.0},
                 rbf_params: Optional[Dict] = {'num_centers': 20, 'gamma': 0.1, 'alpha': 0.0},
                 krg_params: Optional[Dict] = {'poly': 'constant', 'kernel': 'gaussian',
                                               'theta0': 1.0, 'theta_bounds': (1e-6, 100.0)},
                 svr_params: Optional[Dict] = {'kernel': 'rbf', 'gamma': 'scale', 'C': 1.0, 'epsilon': 0.1}):
        """
        Initializes the T-AHS model configuration.

        Args:
        - threshold (float): Threshold for filtering models based on normalized CV error. Suggested: 0.2 or 0.5.
        - prs_params (Optional[Dict]): Dictionary of kwargs to pass to PRS class.
        - rbf_params (Optional[Dict]): Dictionary of kwargs to pass to RBF class.
        - krg_params (Optional[Dict]): Dictionary of kwargs to pass to KRG class.
        - svr_params (Optional[Dict]): Dictionary of kwargs to pass to SVR class.
        """
        self.threshold = threshold

        # Initialize the pool of component models
        self.models_pool = [
            PRS(**prs_params),
            RBF(**rbf_params),
            KRG(**krg_params),
            SVR(**svr_params)]
        self.model_names = ['PRS', 'RBF', 'KRG', 'SVR']

        # Dedicated model for variance estimation
        self.uncertainty_model_ = KRG(**krg_params)

        # State attributes
        self.activate_indices_: List[int] = []
        self.baseline_index_: int = -1
        self.is_fitted: bool = False

    def _calculate_loo_error(self, model: object, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates the Leave-One-Out (LOO) Mean Squared Error for a given model.

        Args:
        - model (object): The surrogate model instance to evaluate.
        - x (np.ndarray): Training inputs (num_samples, num_features).
        - y (np.ndarray): Training targets (num_samples, num_outputs).

        Returns:
        - float: The aggregated Mean Squared Error across all samples and outputs.
        """
        loo = LeaveOneOut()
        y_true_all = []
        y_pred_all = []

        for train_index, test_index in loo.split(x):
            x_train_fold, x_test_fold = x[train_index], x[test_index]
            y_train_fold, y_test_fold = y[train_index], y[test_index]

            model.fit(x_train_fold, y_train_fold)

            pred_result = model.predict(x_test_fold)

            if isinstance(pred_result, tuple):
                y_pred_fold = pred_result[0]
            else:
                y_pred_fold = pred_result

            y_true_all.append(y_test_fold)
            y_pred_all.append(y_pred_fold)

        y_true_stack = np.vstack(y_true_all)
        y_pred_stack = np.vstack(y_pred_all)

        return mean_squared_error(y_true_stack, y_pred_stack)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Trains the Two-Stage Adaptive Hybrid Surrogate (T-AHS) Model.

        Stage 1: Fits all candidate models, computes LOO-CV error, and filters based on threshold.
        Stage 2: Retrains active models on full data and trains the uncertainty model (KRG).

        Args:
        - x_train (np.ndarray): Training feature data (num_samples, num_features).
        - y_train (np.ndarray): Training target data (num_samples, num_outputs).
        """
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)

        logger.info(f'training T-AHS (threshold={self.threshold})...')

        # --- Stage 1: Model Selection via LOO-CV ---
        cv_errors = []

        for name, model in zip(self.model_names, self.models_pool):
            logger.info(f'running LOO-CV for component: {name}...')
            error = self._calculate_loo_error(model, x_train, y_train)
            cv_errors.append(error)

        cv_errors = np.array(cv_errors)

        # Normalize CV errors: E_norm = (E - E_min) / (E_max - E_min)
        e_min = np.min(cv_errors)
        e_max = np.max(cv_errors)

        if np.isclose(e_max, e_min):
            cv_norm = np.zeros_like(cv_errors)
        else:
            cv_norm = (cv_errors - e_min) / (e_max - e_min)

        # Filter models
        self.activate_indices_ = np.where(cv_norm <= self.threshold)[0].tolist()

        # Identify Baseline (Best) Model
        self.baseline_index_ = int(np.argmin(cv_errors))

        # Ensure the baseline is strictly in the active set
        if self.baseline_index_ not in self.activate_indices_:
            self.activate_indices_.append(self.baseline_index_)

        selected_names = [self.model_names[i] for i in self.activate_indices_]

        logger.info(f'{sl.g}T-AHS Stage 1 complete. {sl.q}Selected models: {sl.y}{selected_names}{sl.q}. '
                      f'Baseline: {sl.y}{self.model_names[self.baseline_index_]}{sl.q}.')

        # --- Stage 2: Final Fitting ---
        # 1. Fit all ACTIVE models on the FULL training set
        for i in self.activate_indices_:
            self.models_pool[i].fit(x_train, y_train)

        # 2. Fit the Uncertainty Model (KRG)
        logger.info(f'fitting uncertainty model (KRG) for adaptive weighting...')
        self.uncertainty_model_.fit(x_train, y_train)

        self.is_fitted = True
        logger.info(f'{sl.g}T-AHS training completed.{sl.q}')

    def predict(self, x_test: np.ndarray, y_test: Optional[np.ndarray] = None
                ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """
        Predicts using the trained T-AHS model.

        Adaptive Weighting Logic:
        y_hybrid(x) = sum(w_i(x) * y_i(x))
        where w_i(x) ~ P_i(x) / sum(P_j(x))
        P_i(x) = exp( - (y_baseline(x) - y_i(x))^2 / (2 * s^2(x)) )
        where s^2(x) is the predicted variance from the Kriging model.

        Args:
        - x_test (np.ndarray): Test feature data (num_samples, num_features).
        - y_test (Optional[np.ndarray]): Test target data (num_samples, num_outputs).

        Returns:
        - Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
            - y_pred (np.ndarray): Predicted target values (num_samples, num_outputs).
            - metrics (Dict[str, float]): If y_test is provided.
        """
        if not self.is_fitted:
            raise RuntimeError('Model not fitted. Please call fit() first.')

        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        if y_test is not None and y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        num_samples = x_test.shape[0]
        num_outputs = self.uncertainty_model_.beta.shape[1]

        logger.info(f'predicting T-AHS (Active Models: {len(self.activate_indices_)})...')

        # 1. Get Uncertainty Estimate (s^2)
        _, mse_pred_tuple = self.uncertainty_model_.predict(x_test)

        s2 = np.maximum(mse_pred_tuple, 1e-12)

        # 2. Get Baseline Prediction
        def safe_predict(model, x):
            res = model.predict(x)
            return res[0] if isinstance(res, tuple) else res

        y_base = safe_predict(self.models_pool[self.baseline_index_], x_test)

        # 3. Calculate Weighted Predictions
        numerator_sum = np.zeros((num_samples, num_outputs))
        denominator_sum = np.zeros((num_samples, num_outputs))
        for i in self.activate_indices_:
            model = self.models_pool[i]
            y_curr = safe_predict(model, x_test)

            # Calculate Probability Coefficient P_i
            # P_i(x) = exp( - (y_baseline(x) - y_i(x))^2 / (2 * s^2(x)) )
            # If model is baseline, P_i = 1.0 (Maximum weight)
            squared_diff = (y_base - y_curr) ** 2
            exponent = -squared_diff / (2.0 * s2)

            exponent = np.maximum(exponent, -100.0)
            p_i = np.exp(exponent)

            # Accumulate
            # y_hybrid(x) = sum(w_i(x) * y_i(x)) = sum( (P_i / sum_P) * y_i ) = sum(P_i * y_i) / sum(P_i)
            numerator_sum += p_i * y_curr
            denominator_sum += p_i

        # 4. Final aggregation
        denominator_sum = np.maximum(denominator_sum, 1e-12)
        y_pred = numerator_sum / denominator_sum

        logger.info(f'{sl.g}T-AHS prediction completed.{sl.q}')

        # 5.1 Inference mode
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
    N = 50
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
    model = TAHS()

    # Train model
    model.fit(x_train, y_train)

    # Test model
    y_pred, test_metrics = model.predict(x_test, y_test)

    # Log results
    logger.info(f'Testing R2: {sl.m}{test_metrics['r2']:.9f}{sl.q}')
    logger.info(f'Testing MSE: {sl.m}{test_metrics['mse']:.9f}{sl.q}')
    logger.info(f'Testing RMSE: {sl.m}{test_metrics['rmse']:.9f}{sl.q}')
