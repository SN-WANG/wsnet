# MICO: Mutual Information and Correlation with Multi-fidelity Data
# Paper reference: https://doi.org/10.1007/s00366-023-01858-z
# Paper author: Shuo Wang, Xiaonan Lai, Xiwang He, Kunpeng Li, Liye Lv, Xueguan Song
# Code author: Shengning Wang

import numpy as np
from typing import Optional

from wsnet.models.classical.krg import KRG
from wsnet.data.scaler import MinMaxScalerNP


class MICOInfill:
    """
    Mutual Information and Correlation-based Optimization (MICO) infill criterion.

    Proposes a single new candidate location per call from a discrete low-fidelity
    pool by maximising mutual information between selected and unobserved locations,
    leveraging multi-fidelity covariance structure derived from co-kriging.

    Interface mirrors :class:`~wsnet.sampling.infill.Infill`: construct once per
    iteration, call ``propose()`` to obtain one new point (coordinates), then rebuild
    with updated data for the next iteration.

    Reference:
        Wang et al. (2024). Optimal sensor placement for digital twin based on
        mutual information and correlation with multi-fidelity data.
        Engineering with Computers, 40, 1289–1308.

    Attributes:
        model (KRG): Pre-trained Kriging model on HF data.
        x_hf (np.ndarray): HF training locations, shape (num_hf, input_dim).
        y_hf (np.ndarray): HF training responses, shape (num_hf, output_dim).
        x_lf (np.ndarray): LF candidate pool, shape (num_lf, input_dim).
        y_lf (np.ndarray): LF responses at candidates, shape (num_lf, output_dim).
        target_index (int): Output dimension used to score candidates.
        ratio (float): Weight for the MI term vs. distance-diversity term, in [0, 1].
        theta_v (np.ndarray): LF correlation lengths, shape (input_dim,).
        theta_d (np.ndarray): Discrepancy correlation lengths, shape (input_dim,).
        rho (np.ndarray): HF/LF scaling factors, shape (output_dim,).
        sigma_sq_v (np.ndarray): LF process variance, shape (output_dim,).
        sigma_sq_d (np.ndarray): Discrepancy variance, shape (output_dim,).
        scaler_dist (MinMaxScalerNP): Scaler for distance normalisation.
    """

    def __init__(
        self,
        model: KRG,
        x_hf: np.ndarray,
        y_hf: np.ndarray,
        x_lf: np.ndarray,
        y_lf: np.ndarray,
        target_index: int = 0,
        ratio: float = 0.5,
        theta_v: Optional[np.ndarray] = None,
        theta_d: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialise MICO multi-fidelity infill strategy.

        Args:
            model (KRG): A fitted Kriging model on HF data.
            x_hf (np.ndarray): HF training locations. shape: (num_hf, input_dim).
            y_hf (np.ndarray): HF training responses. shape: (num_hf, output_dim).
            x_lf (np.ndarray): LF candidate pool (simulation nodes).
                shape: (num_lf, input_dim).
            y_lf (np.ndarray): LF responses at candidate locations.
                shape: (num_lf, output_dim).
            target_index (int): Output dimension to optimise. Default 0.
            ratio (float): Weight for MI term vs. diversity term, in [0, 1].
                Default 0.5.
            theta_v (Optional[np.ndarray]): Correlation lengths for the LF process.
                If None, extracted from ``model.theta``. shape: (input_dim,).
            theta_d (Optional[np.ndarray]): Correlation lengths for the discrepancy
                process. If None, copied from ``theta_v``. shape: (input_dim,).

        Raises:
            RuntimeError: If ``model`` is not fitted (``model.theta`` is None).
            ValueError: If spatial dimensions of HF and LF arrays are inconsistent.
        """
        if not hasattr(model, "theta") or model.theta is None:
            raise RuntimeError("Provided KRG model is not fitted.")

        self.model = model
        self.x_hf = np.array(x_hf, dtype=np.float64)
        self.y_hf = np.array(y_hf, dtype=np.float64)
        self.x_lf = np.array(x_lf, dtype=np.float64)
        self.y_lf = np.array(y_lf, dtype=np.float64)

        self.num_lf, self.input_dim = self.x_lf.shape
        self.num_hf = self.x_hf.shape[0]
        self.output_dim = self.y_lf.shape[1] if self.y_lf.ndim > 1 else 1

        if self.x_hf.shape[1] != self.input_dim:
            raise ValueError(
                f"x_hf dimension {self.x_hf.shape[1]} does not match "
                f"x_lf dimension {self.input_dim}."
            )
        if self.y_hf.shape[0] != self.num_hf:
            raise ValueError(
                f"y_hf samples {self.y_hf.shape[0]} does not match "
                f"x_hf samples {self.num_hf}."
            )

        self.target_index = int(target_index)
        self.ratio = float(ratio)

        self.theta_v = self._initialize_theta(theta_v)
        self.theta_d = (
            self._initialize_theta(theta_d) if theta_d is not None
            else self.theta_v.copy()
        )

        self.rho = self._estimate_rho()
        self.sigma_sq_v = self._estimate_sigma_sq_v()
        self.sigma_sq_d = self._estimate_sigma_sq_d()

        self.hf_indices = self._map_hf_to_lf()
        self.scaler_dist = MinMaxScalerNP(norm_range="unit")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_dists(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances between two point sets.

        Uses the algebraic expansion ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
        for vectorised computation without explicit loops.

        Args:
            x (np.ndarray): Query points. shape: (num_queries, num_features).
            c (np.ndarray): Reference points. shape: (num_refs, num_features).

        Returns:
            np.ndarray: Squared distance matrix. shape: (num_queries, num_refs).
                Entry (i, j) is ||x[i] - c[j]||^2.
        """
        x_norm_sq = np.sum(x ** 2, axis=1, keepdims=True)
        c_norm_sq = np.sum(c ** 2, axis=1)
        cross = 2.0 * (x @ c.T)
        dists_sq = x_norm_sq + c_norm_sq - cross
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return dists_sq

    def _initialize_theta(self, theta: Optional[np.ndarray]) -> np.ndarray:
        """
        Resolve correlation length parameters.

        Args:
            theta (Optional[np.ndarray]): User-supplied theta or None.

        Returns:
            np.ndarray: Correlation lengths. shape: (input_dim,).
        """
        if theta is not None:
            return np.array(theta, dtype=np.float64).flatten()

        if hasattr(self.model, "theta") and self.model.theta is not None:
            model_theta = self.model.theta
            if np.isscalar(model_theta) or np.asarray(model_theta).size == 1:
                return np.full(self.input_dim, float(model_theta), dtype=np.float64)
            return np.array(model_theta, dtype=np.float64).flatten()

        return np.ones(self.input_dim, dtype=np.float64)

    def _estimate_rho(self) -> np.ndarray:
        """
        Estimate the HF/LF scaling factor ``rho`` from observed HF data.

        Returns:
            np.ndarray: Per-output scaling factors. shape: (output_dim,).
        """
        y_lf_at_hf, _ = self.model.predict(self.x_hf)
        rho = np.ones(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            yl = y_lf_at_hf[:, d]
            yh = self.y_hf[:, d]
            mask = np.abs(yl) > 1e-10
            if np.any(mask):
                rho[d] = np.mean(yh[mask] / yl[mask])
        return rho

    def _estimate_sigma_sq_v(self) -> np.ndarray:
        """
        Estimate the LF process variance from the fitted KRG model.

        Returns:
            np.ndarray: LF variance per output. shape: (output_dim,).
        """
        if hasattr(self.model, "sigma2") and self.model.sigma2 is not None:
            sigma2 = self.model.sigma2
            if np.isscalar(sigma2):
                return np.full(self.output_dim, float(sigma2), dtype=np.float64)
            return np.array(sigma2, dtype=np.float64).flatten()
        return np.var(self.y_lf, axis=0).astype(np.float64)

    def _estimate_sigma_sq_d(self) -> np.ndarray:
        """
        Estimate the discrepancy variance from initial HF observations.

        Discrepancy is defined as ``y_hf - rho * y_lf_predicted``.

        Returns:
            np.ndarray: Discrepancy variance per output. shape: (output_dim,).
        """
        y_lf_at_hf, _ = self.model.predict(self.x_hf)
        sigma_sq_d = np.zeros(self.output_dim, dtype=np.float64)
        for d in range(self.output_dim):
            discrepancy = self.y_hf[:, d] - self.rho[d] * y_lf_at_hf[:, d]
            sigma_sq_d[d] = np.var(discrepancy)
        return sigma_sq_d

    def _map_hf_to_lf(self) -> np.ndarray:
        """
        Map each HF location to the nearest LF candidate index.

        Returns:
            np.ndarray: LF indices for each HF location. shape: (num_hf,).
        """
        dists = self._compute_dists(self.x_hf, self.x_lf)
        return np.argmin(dists, axis=1).astype(np.int64)

    def _compute_correlation_matrix(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the anisotropic Gaussian correlation matrix.

        ``Psi[i, j] = exp(-sum_k theta_k * (x1[i,k] - x2[j,k])^2)``

        Matches Eq. 47–48 in Wang et al. (2024).

        Args:
            x1 (np.ndarray): First point set. shape: (n1, input_dim).
            x2 (np.ndarray): Second point set. shape: (n2, input_dim).
            theta (np.ndarray): Correlation lengths. shape: (input_dim,).

        Returns:
            np.ndarray: Correlation matrix. shape: (n1, n2).
        """
        x1s = x1 * np.sqrt(theta)
        x2s = x2 * np.sqrt(theta)
        x1_norm_sq = np.sum(x1s ** 2, axis=1, keepdims=True)
        x2_norm_sq = np.sum(x2s ** 2, axis=1)
        cross = 2.0 * (x1s @ x2s.T)
        dists_sq = x1_norm_sq + x2_norm_sq - cross
        np.maximum(dists_sq, 0.0, out=dists_sq)
        return np.exp(-dists_sq)

    def _compute_multi_fidelity_covariance(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        output_dim: int,
    ) -> np.ndarray:
        """
        Compute the multi-fidelity co-kriging covariance matrix.

        ``C_HF = rho^2 * sigma_sq_v * Psi_v + sigma_sq_d * Psi_d``

        Matches Eq. 16 in Wang et al. (2024).

        Args:
            x1 (np.ndarray): First point set. shape: (n1, input_dim).
            x2 (np.ndarray): Second point set. shape: (n2, input_dim).
            output_dim (int): Index of the output dimension.

        Returns:
            np.ndarray: Covariance matrix. shape: (n1, n2).
        """
        rho_d = self.rho[output_dim]
        sigma_v = self.sigma_sq_v[output_dim]
        sigma_d = self.sigma_sq_d[output_dim]
        psi_v = self._compute_correlation_matrix(x1, x2, self.theta_v)
        psi_d = self._compute_correlation_matrix(x1, x2, self.theta_d)
        return (rho_d ** 2 * sigma_v) * psi_v + sigma_d * psi_d

    def _select_new_point(self, output_dim: int) -> int:
        """
        Select one new candidate from the LF pool via the MICO criterion.

        The initial "selected" set is seeded with the nearest-neighbour LF indices
        of the existing HF observations. One greedy MICO step then picks the
        unselected candidate that maximises the combined mutual-information and
        diversity score.

        MICO criterion (Eq. 40 + 57 in Wang et al. 2024):

        .. code-block::

            delta = delta_n * delta_d

            delta_n = diag(C_vv - C_yA @ C_AA^{-1} @ C_Ay)   # residual variance
            delta_d = diag(C_vv^{-1})                         # inverse self-covariance

        Args:
            output_dim (int): Index of the output dimension to score.

        Returns:
            int: Index into ``x_lf`` of the best new candidate.
        """
        selected = list(self.hf_indices.copy())
        x_selected = self.x_lf[selected]

        # Covariance of the current selected set
        c_aa = self._compute_multi_fidelity_covariance(
            x_selected, x_selected, output_dim
        )
        c_aa += np.eye(len(selected), dtype=np.float64) * 1e-6

        try:
            icov_a = np.linalg.inv(c_aa)
        except np.linalg.LinAlgError:
            icov_a = np.linalg.pinv(c_aa)

        # Candidate set: LF locations not already selected
        candidate_mask = np.ones(self.num_lf, dtype=bool)
        candidate_mask[selected] = False
        candidates = np.where(candidate_mask)[0]

        if len(candidates) == 0:
            return int(np.random.randint(self.num_lf))

        x_candidates = self.x_lf[candidates]

        # Distance-diversity term: min distance from each candidate to selected set
        dists_all = self._compute_dists(x_candidates, x_selected)
        dists = np.min(dists_all, axis=1)

        if len(dists) > 1 and np.max(dists) > np.min(dists):
            dists_norm = self.scaler_dist.fit(
                dists.reshape(-1, 1), channel_dim=0
            ).transform(dists.reshape(-1, 1)).flatten()
        else:
            dists_norm = np.zeros_like(dists)

        # Covariance terms for MICO criterion
        c_ya = self._compute_multi_fidelity_covariance(
            x_candidates, x_selected, output_dim
        )  # (num_cand, num_sel)
        c_vv = self._compute_multi_fidelity_covariance(
            x_candidates, x_candidates, output_dim
        )  # (num_cand, num_cand)
        c_vv += np.eye(len(candidates), dtype=np.float64) * 1e-6

        try:
            icov_vv = np.linalg.inv(c_vv)
        except np.linalg.LinAlgError:
            icov_vv = np.linalg.pinv(c_vv)

        # MICO criterion: delta_n * delta_d
        temp = c_ya @ icov_a @ c_ya.T
        delta_n = np.maximum(np.diag(c_vv - temp), 1e-12)
        delta_d = np.maximum(np.diag(icov_vv), 1e-12)
        delta = delta_n * delta_d

        d_min, d_max = np.min(delta), np.max(delta)
        if d_max > d_min:
            delta_norm = (delta - d_min) / (d_max - d_min)
        else:
            delta_norm = np.ones_like(delta)

        criterion = self.ratio * delta_norm + (1.0 - self.ratio) * dists_norm
        best_in_candidates = int(np.argmax(criterion))
        return int(candidates[best_in_candidates])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(self) -> np.ndarray:
        """
        Propose a single new sampling point from the LF candidate pool.

        Selects the LF candidate that maximises the MICO mutual-information
        and diversity criterion for ``target_index``.

        Returns:
            np.ndarray: Coordinates of the proposed point. shape: (1, input_dim).
        """
        new_idx = self._select_new_point(self.target_index)
        return self.x_lf[new_idx].reshape(1, -1)

    def evaluate(self, indices: np.ndarray) -> np.ndarray:
        """
        Evaluate the MICO criterion for a set of candidate indices.

        Computes scores without modifying the internal selection state.
        Useful for analysis and debugging.

        Args:
            indices (np.ndarray): Candidate indices into ``x_lf``.
                shape: (n_candidates,).

        Returns:
            np.ndarray: Criterion values per candidate and output dimension.
                shape: (n_candidates, output_dim). Higher values indicate
                better candidates.
        """
        indices = np.array(indices, dtype=np.int64)
        n_candidates = len(indices)
        results = np.zeros((n_candidates, self.output_dim), dtype=np.float64)

        for d in range(self.output_dim):
            selected = list(self.hf_indices.copy())
            x_selected = self.x_lf[selected]

            c_aa = self._compute_multi_fidelity_covariance(
                x_selected, x_selected, d
            )
            c_aa += np.eye(len(selected), dtype=np.float64) * 1e-6

            try:
                icov_a = np.linalg.inv(c_aa)
            except np.linalg.LinAlgError:
                icov_a = np.linalg.pinv(c_aa)

            x_candidates = self.x_lf[indices]
            c_ya = self._compute_multi_fidelity_covariance(x_candidates, x_selected, d)
            c_vv = self._compute_multi_fidelity_covariance(x_candidates, x_candidates, d)
            c_vv += np.eye(n_candidates, dtype=np.float64) * 1e-6

            try:
                icov_vv = np.linalg.inv(c_vv)
            except np.linalg.LinAlgError:
                icov_vv = np.linalg.pinv(c_vv)

            temp = c_ya @ icov_a @ c_ya.T
            delta_n = np.maximum(np.diag(c_vv - temp), 1e-12)
            delta_d = np.maximum(np.diag(icov_vv), 1e-12)
            results[:, d] = delta_n * delta_d

        return results
