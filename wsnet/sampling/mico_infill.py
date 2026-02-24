# MICO: Mutual Information and Correlation-based Optimization
# Multi-fidelity sensor placement for digital twins

import numpy as np
from typing import Optional, List

from wsnet.models.classical.krg import KRG
from wsnet.data.scaler import MinMaxScalerNP


class MICOInfill:
    """
    Mutual Information and Correlation-based Optimization (MICO) for 
    multi-fidelity sensor placement in digital twins.

    MICO selects optimal sensor locations from a discrete candidate pool
    (typically low-fidelity simulation nodes) by maximizing mutual information
    between selected and uninstrumented locations, leveraging multi-fidelity
    covariance structure inspired by co-kriging.

    For multi-output problems, per-output selections are computed and then
    clustered using FCM-style aggregation to determine final sensor locations.

    Reference:
        Wang et al. (2024). Optimal sensor placement for digital twin based on 
        mutual information and correlation with multi-fidelity data.
        Engineering with Computers, 40, 1289-1308.

    Attributes:
        model (KRG): Pre-trained Kriging model on low-fidelity data.
        x_lf (np.ndarray): LF candidate locations, shape (num_lf, input_dim).
        y_lf (np.ndarray): LF responses at candidate locations, 
            shape (num_lf, target_dim).
        x_hf (np.ndarray): Initial HF sensor locations, 
            shape (num_hf_init, input_dim).
        y_hf (np.ndarray): HF measurements at initial sensors, 
            shape (num_hf_init, target_dim).
        num_select (int): Total number of sensors to select.
        ratio (float): Balance between MI term and distance term, in [0, 1].
        theta_v (np.ndarray): Correlation length for LF, shape (input_dim,).
        theta_d (np.ndarray): Correlation length for discrepancy, 
            shape (input_dim,).
        rho (np.ndarray): HF/LF scaling factors, shape (target_dim,).
        sigma_sq_v (np.ndarray): LF variance, shape (target_dim,).
        sigma_sq_d (np.ndarray): Discrepancy variance, shape (target_dim,).
        scaler_dist (MinMaxScalerNP): Scaler for distance normalization.
    """

    def __init__(
        self,
        model: KRG,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        num_select: int,
        ratio: float = 0.5,
        theta_v: Optional[np.ndarray] = None,
        theta_d: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        sigma_sq_v: Optional[np.ndarray] = None,
        sigma_sq_d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize MICO multi-fidelity sensor placement.

        Args:
            model: Pre-trained Kriging model on LF data. Must be fitted.
            x_lf: LF candidate locations (simulation nodes), 
                shape (num_lf, input_dim), dtype float64.
            y_lf: LF responses at candidate locations, 
                shape (num_lf, target_dim), dtype float64.
            x_hf: Initial HF sensor locations, may be arbitrary points not in x_lf,
                shape (num_hf_init, input_dim), dtype float64.
            y_hf: HF measurements at initial sensors, 
                shape (num_hf_init, target_dim), dtype float64.
            num_select: Total number of sensors to select (K >= num_hf_init).
            ratio: Weight for MI term vs distance term, in [0, 1]. Default 0.5.
            theta_v: Correlation length for LF. If None, extracted from model.theta.
                shape (input_dim,), dtype float64.
            theta_d: Correlation length for discrepancy. If None, set to theta_v.
                shape (input_dim,), dtype float64.
            rho: HF/LF scaling factors. If None, estimated from data.
                shape (target_dim,), dtype float64.
            sigma_sq_v: LF variance. If None, estimated from data.
                shape (target_dim,), dtype float64.
            sigma_sq_d: Discrepancy variance. If None, estimated from data.
                shape (target_dim,), dtype float64.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If num_select < num_hf_init or invalid dimensions.
        """
        # Validate model is fitted
        if not hasattr(model, "theta") or model.theta is None:
            raise RuntimeError("Provided KRG model is not fitted.")

        # Store references
        self.model = model
        self.x_lf = np.array(x_lf, dtype=np.float64)
        self.y_lf = np.array(y_lf, dtype=np.float64)
        self.x_hf = np.array(x_hf, dtype=np.float64)
        self.y_hf = np.array(y_hf, dtype=np.float64)

        # Dimensions
        self.num_lf, self.input_dim = self.x_lf.shape
        self.num_hf_init = self.x_hf.shape[0]
        self.target_dim = self.y_lf.shape[1] if self.y_lf.ndim > 1 else 1

        # Validate dimensions
        if self.x_hf.shape[1] != self.input_dim:
            raise ValueError(
                f"x_hf dimension {self.x_hf.shape[1]} does not match "
                f"x_lf dimension {self.input_dim}"
            )
        if self.y_hf.shape[0] != self.num_hf_init:
            raise ValueError(
                f"y_hf samples {self.y_hf.shape[0]} does not match "
                f"x_hf samples {self.num_hf_init}"
            )
        if num_select < self.num_hf_init:
            raise ValueError(
                f"num_select {num_select} must be >= initial HF samples "
                f"{self.num_hf_init}"
            )

        self.num_select = num_select
        self.ratio = float(ratio)

        # Extract or set hyperparameters
        self.theta_v = self._initialize_theta(theta_v)
        self.theta_d = self._initialize_theta(theta_d) if theta_d is not None else self.theta_v.copy()

        # Estimate multi-fidelity parameters
        self.rho = self._estimate_rho() if rho is None else np.array(rho, dtype=np.float64)
        self.sigma_sq_v = self._estimate_sigma_sq_v() if sigma_sq_v is None else np.array(sigma_sq_v, dtype=np.float64)
        self.sigma_sq_d = self._estimate_sigma_sq_d() if sigma_sq_d is None else np.array(sigma_sq_d, dtype=np.float64)

        # Ensure target_dim consistency
        if self.target_dim == 1:
            self.rho = np.atleast_1d(self.rho)
            self.sigma_sq_v = np.atleast_1d(self.sigma_sq_v)
            self.sigma_sq_d = np.atleast_1d(self.sigma_sq_d)

        # Map HF locations to LF candidate indices (nearest neighbor)
        self.hf_indices = self._map_hf_to_lf()

        # Initialize distance scaler
        self.scaler_dist = MinMaxScalerNP(feature_range="unit")

    # ------------------------------------------------------------------

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between two point sets.

        Uses the algebraic expansion: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x @ c^T
        for vectorized computation without explicit loops.

        Args:
            x (np.ndarray): Query points of shape (num_queries, num_features), dtype: float64.
            c (np.ndarray): Reference points of shape (num_refs, num_features), dtype: float64.

        Returns:
            np.ndarray: Squared distance matrix of shape (num_queries, num_refs), dtype: float64.
                Entry (i, j) contains the squared Euclidean distance between x[i] and c[j].
        """
        # Compute squared L2 norms for each point set: (num_queries, 1) and (num_refs,)
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)  # Shape: (num_queries, 1)
        c_norm_sq = np.sum(c ** 2, axis=1)                 # Shape: (num_refs,)

        # Compute cross term: 2 * x @ c^T, shape: (num_queries, num_refs)
        cross_term = 2.0 * (x @ c.T)

        # Combine: ||x||^2 + ||c||^2 - 2 * x @ c^T
        # Broadcasting: (num_queries, 1) + (num_refs,) -> (num_queries, num_refs)
        dists_sq = x_norm_sq + c_norm_sq - cross_term

        # Numerical stability: clamp small negative values to zero (floating point errors)
        np.maximum(dists_sq, 0.0, out=dists_sq)

        return dists_sq

    # ------------------------------------------------------------------

    def _initialize_theta(self, theta: Optional[np.ndarray]) -> np.ndarray:
        """
        Initialize correlation length parameters.

        Args:
            theta: User-provided theta or None.

        Returns:
            np.ndarray: Correlation length, shape (input_dim,), dtype float64.
        """
        if theta is not None:
            return np.array(theta, dtype=np.float64).flatten()

        # Extract from KRG model
        if hasattr(self.model, "theta"):
            model_theta = self.model.theta
            if np.isscalar(model_theta) or model_theta.size == 1:
                return np.full(self.input_dim, float(model_theta), dtype=np.float64)
            else:
                return np.array(model_theta, dtype=np.float64).flatten()

        # Default fallback
        return np.ones(self.input_dim, dtype=np.float64)

    # ------------------------------------------------------------------

    def _estimate_rho(self) -> np.ndarray:
        """
        Estimate HF/LF scaling factor rho from initial HF data.

        Returns:
            np.ndarray: Scaling factors, shape (target_dim,), dtype float64.
        """
        # Get LF predictions at HF locations
        y_lf_at_hf, _ = self.model.predict(self.x_hf)

        # Simple least squares estimate: rho = mean(y_hf / y_lf) per output
        rho = np.zeros(self.target_dim, dtype=np.float64)
        for d in range(self.target_dim):
            yl = y_lf_at_hf[:, d]
            yh = self.y_hf[:, d]
            # Avoid division by zero
            mask = np.abs(yl) > 1e-10
            if np.any(mask):
                rho[d] = np.mean(yh[mask] / yl[mask])
            else:
                rho[d] = 1.0

        return rho

    # ------------------------------------------------------------------

    def _estimate_sigma_sq_v(self) -> np.ndarray:
        """
        Estimate LF variance from KRG model.

        Returns:
            np.ndarray: LF variance, shape (target_dim,), dtype float64.
        """
        if hasattr(self.model, "sigma2") and self.model.sigma2 is not None:
            sigma2 = self.model.sigma2
            if np.isscalar(sigma2):
                return np.full(self.target_dim, float(sigma2), dtype=np.float64)
            else:
                return np.array(sigma2, dtype=np.float64).flatten()

        # Estimate from data variance
        return np.var(self.y_lf, axis=0).astype(np.float64)

    # ------------------------------------------------------------------

    def _estimate_sigma_sq_d(self) -> np.ndarray:
        """
        Estimate discrepancy variance from initial HF data.

        Returns:
            np.ndarray: Discrepancy variance, shape (target_dim,), dtype float64.
        """
        # Get LF predictions at HF locations
        y_lf_at_hf, _ = self.model.predict(self.x_hf)

        # Discrepancy = y_hf - rho * y_lf
        sigma_sq_d = np.zeros(self.target_dim, dtype=np.float64)
        for d in range(self.target_dim):
            discrepancy = self.y_hf[:, d] - self.rho[d] * y_lf_at_hf[:, d]
            sigma_sq_d[d] = np.var(discrepancy)

        return sigma_sq_d

    # ------------------------------------------------------------------

    def _map_hf_to_lf(self) -> np.ndarray:
        """
        Map initial HF locations to nearest LF candidate indices.

        Returns:
            np.ndarray: Indices into x_lf for each HF location, 
                shape (num_hf_init,), dtype int64.
        """
        # Compute pairwise distances between HF and LF
        dists = self._compute_dists(self.x_hf, self.x_lf)  # (num_hf, num_lf)

        # Find nearest neighbor
        hf_indices = np.argmin(dists, axis=1)  # (num_hf,)

        return hf_indices.astype(np.int64)

    # ------------------------------------------------------------------

    def _compute_correlation_matrix(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute Gaussian correlation matrix.

        psi_ij = exp(-sum_k theta_k * (x1_ik - x2_jk)^2)

        Args:
            x1: First point set, shape (n1, input_dim), dtype float64.
            x2: Second point set, shape (n2, input_dim), dtype float64.
            theta: Correlation lengths, shape (input_dim,), dtype float64.

        Returns:
            np.ndarray: Correlation matrix, shape (n1, n2), dtype float64.
        """
        # Compute squared Euclidean distances with anisotropic scaling
        # dists[i,j] = sum_k theta_k * (x1[i,k] - x2[j,k])^2
        x1_scaled = x1 * np.sqrt(theta)  # (n1, input_dim)
        x2_scaled = x2 * np.sqrt(theta)  # (n2, input_dim)

        # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2*x1@x2.T
        x1_norm_sq = np.sum(x1_scaled**2, axis=1, keepdims=True)  # (n1, 1)
        x2_norm_sq = np.sum(x2_scaled**2, axis=1)  # (n2,)
        cross_term = 2.0 * np.dot(x1_scaled, x2_scaled.T)  # (n1, n2)

        dists_sq = x1_norm_sq + x2_norm_sq - cross_term  # (n1, n2)
        np.maximum(dists_sq, 0.0, out=dists_sq)  # Numerical stability

        return np.exp(-dists_sq)

    # ------------------------------------------------------------------

    def _compute_multi_fidelity_covariance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        output_dim: int
    ) -> np.ndarray:
        """
        Compute multi-fidelity covariance matrix.

        C_HF = rho^2 * sigma_sq_v * Psi_v + sigma_sq_d * Psi_d

        Args:
            x1: First point set, shape (n1, input_dim), dtype float64.
            x2: Second point set, shape (n2, input_dim), dtype float64.
            output_dim: Output dimension index.

        Returns:
            np.ndarray: Covariance matrix, shape (n1, n2), dtype float64.
        """
        rho_d = self.rho[output_dim]
        sigma_v = self.sigma_sq_v[output_dim]
        sigma_d = self.sigma_sq_d[output_dim]

        psi_v = self._compute_correlation_matrix(x1, x2, self.theta_v)
        psi_d = self._compute_correlation_matrix(x1, x2, self.theta_d)

        return (rho_d**2 * sigma_v) * psi_v + sigma_d * psi_d

    # ------------------------------------------------------------------

    def _select_for_single_output(
        self,
        output_dim: int
    ) -> np.ndarray:
        """
        Execute greedy MICO selection for a single output dimension.

        Args:
            output_dim: Index of output dimension to optimize.

        Returns:
            np.ndarray: Selected sensor indices, shape (num_select,), dtype int64.
        """
        # Initialize selected set with mapped HF indices
        selected = list(self.hf_indices.copy())  # List for dynamic append

        # Candidate set: all LF indices not in selected
        all_indices = np.arange(self.num_lf, dtype=np.int64)

        # Number of additional sensors to select
        num_additional = self.num_select - len(selected)

        if num_additional <= 0:
            return np.array(selected[:self.num_select], dtype=np.int64)

        # Initialize covariance matrices
        # C_AA: covariance among selected points
        x_selected = self.x_lf[selected]  # (num_hf, input_dim)
        c_aa = self._compute_multi_fidelity_covariance(
            x_selected, x_selected, output_dim
        )  # (num_hf, num_hf)

        # Add nugget for numerical stability
        c_aa += np.eye(len(selected), dtype=np.float64) * 1e-6

        # Initial inverse
        try:
            icov_a = np.linalg.inv(c_aa)
        except np.linalg.LinAlgError:
            icov_a = np.linalg.pinv(c_aa)

        # Greedy selection loop
        for _ in range(num_additional):
            # Candidate indices (not selected)
            candidate_mask = np.ones(self.num_lf, dtype=bool)
            candidate_mask[selected] = False
            candidates = all_indices[candidate_mask]  # (num_cand,)

            if len(candidates) == 0:
                break

            x_candidates = self.x_lf[candidates]  # (num_cand, input_dim)

            # Compute distances from candidates to selected set
            # dists[i] = min_j ||x_candidates[i] - x_selected[j]||
            dists_all = self._compute_dists(x_candidates, x_selected)  # (num_cand, num_sel)
            dists = np.min(dists_all, axis=1)  # (num_cand,)

            # Normalize distances using MinMaxScalerNP
            if len(dists) > 1 and np.max(dists) > np.min(dists):
                dists_norm = self.scaler_dist.fit(
                    dists.reshape(-1, 1), channel_dim=0
                ).transform(dists.reshape(-1, 1)).flatten()
            else:
                dists_norm = np.zeros_like(dists)

            # Compute covariance terms for MICO criterion
            # C_yA: covariance between candidates and selected
            c_ya = self._compute_multi_fidelity_covariance(
                x_candidates, x_selected, output_dim
            )  # (num_cand, num_sel)

            # C_Ay: transpose
            c_ay = c_ya.T  # (num_sel, num_cand)

            # Covariance among candidates (for delta_d computation)
            c_vv = self._compute_multi_fidelity_covariance(
                x_candidates, x_candidates, output_dim
            )  # (num_cand, num_cand)

            # Add nugget
            c_vv += np.eye(len(candidates), dtype=np.float64) * 1e-6

            try:
                icov_vv = np.linalg.inv(c_vv)
            except np.linalg.LinAlgError:
                icov_vv = np.linalg.pinv(c_vv)

            # delta_n: numerator term = diag(C_vv - C_yA @ icovA @ C_Ay)
            temp = c_ya @ icov_a @ c_ay  # (num_cand, num_cand)
            delta_n = np.diag(c_vv - temp)  # (num_cand,)

            # delta_d: denominator term = diag(inv(C_vv))
            delta_d = np.diag(icov_vv)  # (num_cand,)

            # Avoid negative or zero values
            delta_n = np.maximum(delta_n, 1e-12)
            delta_d = np.maximum(delta_d, 1e-12)

            # MICO criterion: delta_n * delta_d (from paper Eq. 40)
            delta = delta_n * delta_d  # (num_cand,)

            # Normalize delta
            if len(delta) > 1 and np.max(delta) > np.min(delta):
                delta_min, delta_max = np.min(delta), np.max(delta)
                delta_norm = (delta - delta_min) / (delta_max - delta_min)
            else:
                delta_norm = np.ones_like(delta)

            # Combined criterion
            criterion = self.ratio * delta_norm + (1.0 - self.ratio) * dists_norm  # (num_cand,)

            # Select best candidate
            best_idx_in_candidates = int(np.argmax(criterion))
            best_candidate = int(candidates[best_idx_in_candidates])

            # Update selected set
            selected.append(best_candidate)

            # Update inverse matrix using block inversion formula
            # Extract covariance for new point
            c_y_new = c_ya[best_idx_in_candidates:best_idx_in_candidates+1, :]  # (1, num_sel)
            c_new_y = c_ay[:, best_idx_in_candidates:best_idx_in_candidates+1]  # (num_sel, 1)
            c_new_new = c_vv[best_idx_in_candidates, best_idx_in_candidates]  # scalar

            # Schur complement
            delta_schur = c_new_new - float(c_y_new @ icov_a @ c_new_y)
            delta_schur = max(delta_schur, 1e-12)

            # Block inversion update
            block_22 = 1.0 / delta_schur  # scalar
            block_12 = -block_22 * (icov_a @ c_new_y)  # (num_sel, 1)
            block_21 = -block_22 * (c_y_new @ icov_a)  # (1, num_sel)
            block_11 = icov_a + block_12 @ (c_y_new @ icov_a)  # (num_sel, num_sel)

            # Assemble new inverse
            new_size = len(selected)
            icov_a_new = np.zeros((new_size, new_size), dtype=np.float64)
            icov_a_new[:len(selected)-1, :len(selected)-1] = block_11
            icov_a_new[:len(selected)-1, len(selected)-1] = block_12.flatten()
            icov_a_new[len(selected)-1, :len(selected)-1] = block_21.flatten()
            icov_a_new[len(selected)-1, len(selected)-1] = block_22

            icov_a = icov_a_new
            x_selected = self.x_lf[selected]

        return np.array(selected, dtype=np.int64)

    # ------------------------------------------------------------------

    def _aggregate_selections(
        self,
        selections_per_output: List[np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate per-output selections using FCM-style clustering.

        For multi-output problems, each output dimension produces its own
        optimal sensor set. This method clusters all selected positions to
        determine a single consensus set of sensor locations.

        Args:
            selections_per_output: List of selected indices for each output,
                each shape (num_select,), dtype int64.

        Returns:
            np.ndarray: Final selected indices, shape (num_select,), dtype int64.
        """
        if self.target_dim == 1:
            return selections_per_output[0]

        # Collect all unique selected positions across outputs
        all_selected = np.concatenate(selections_per_output)  # (num_select * target_dim,)
        unique_positions = np.unique(all_selected)

        if len(unique_positions) <= self.num_select:
            # Pad if necessary
            result = np.zeros(self.num_select, dtype=np.int64)
            result[:len(unique_positions)] = unique_positions
            if len(unique_positions) < self.num_select:
                # Fill remaining with random from LF
                remaining = np.setdiff1d(np.arange(self.num_lf), unique_positions)
                if len(remaining) > 0:
                    n_needed = self.num_select - len(unique_positions)
                    result[len(unique_positions):] = np.random.choice(
                        remaining, size=min(n_needed, len(remaining)), replace=False
                    )
            return result

        # Get coordinates of unique positions
        coords = self.x_lf[unique_positions]  # (num_unique, input_dim)

        # Greedy furthest-point sampling for diverse selection
        centroids = [unique_positions[0]]
        for _ in range(1, self.num_select):
            dists_to_centroids = self._compute_dists(coords, self.x_lf[np.array(centroids)])
            min_dists = np.min(dists_to_centroids, axis=1)
            next_idx = int(np.argmax(min_dists))
            centroids.append(unique_positions[next_idx])

        return np.array(centroids, dtype=np.int64)

    # ------------------------------------------------------------------

    def propose(self) -> np.ndarray:
        """
        Execute MICO sensor placement.

        For multi-output problems, performs per-output selection followed
        by clustering aggregation.

        Returns:
            np.ndarray: Selected sensor indices into x_lf, 
                shape (num_select,), dtype int64.
        """
        # Run selection for each output dimension
        selections = []
        for d in range(self.target_dim):
            sel = self._select_for_single_output(d)
            selections.append(sel)

        # Aggregate if multi-output
        final_selection = self._aggregate_selections(selections)

        return final_selection

    # ------------------------------------------------------------------

    def evaluate(self, indices: np.ndarray) -> np.ndarray:
        """
        Evaluate MICO criterion for given candidate indices.

        Computes the mutual information criterion for specified candidates
        without updating the selection. Useful for analysis and debugging.

        Args:
            indices: Candidate indices to evaluate, shape (n_candidates,), 
                dtype int64.

        Returns:
            np.ndarray: Criterion values for each candidate and output,
                shape (n_candidates, target_dim), dtype float64.
                Higher values indicate better candidates.
        """
        indices = np.array(indices, dtype=np.int64)
        n_candidates = len(indices)

        results = np.zeros((n_candidates, self.target_dim), dtype=np.float64)

        for d in range(self.target_dim):
            # Current selected set
            selected = list(self.hf_indices.copy())
            x_selected = self.x_lf[selected]

            # Covariance of selected
            c_aa = self._compute_multi_fidelity_covariance(
                x_selected, x_selected, d
            )
            c_aa += np.eye(len(selected), dtype=np.float64) * 1e-6

            try:
                icov_a = np.linalg.inv(c_aa)
            except np.linalg.LinAlgError:
                icov_a = np.linalg.pinv(c_aa)

            # Evaluate each candidate
            x_candidates = self.x_lf[indices]

            # Covariance terms
            c_ya = self._compute_multi_fidelity_covariance(
                x_candidates, x_selected, d
            )
            c_vv = self._compute_multi_fidelity_covariance(
                x_candidates, x_candidates, d
            )
            c_vv += np.eye(n_candidates, dtype=np.float64) * 1e-6

            try:
                icov_vv = np.linalg.inv(c_vv)
            except np.linalg.LinAlgError:
                icov_vv = np.linalg.pinv(c_vv)

            # delta computation
            temp = c_ya @ icov_a @ c_ya.T
            delta_n = np.diag(c_vv - temp)
            delta_d = np.diag(icov_vv)

            delta_n = np.maximum(delta_n, 1e-12)
            delta_d = np.maximum(delta_d, 1e-12)

            results[:, d] = delta_n * delta_d

        return results
