# Many-Objective Sequential Sampling via Importance Sampling EHVI (IS-EHVI)
# Paper: Pang et al. (2023) "An Expensive Many-Objective Optimization Algorithm
#        Based on Efficient Expected Hypervolume Improvement"
# Code author: Shengning Wang

import warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
from typing import List, Optional

from wsnet.models.classical.krg import KRG


class MOInfill:
    """
    Multi-Objective Infill via IS-EHVI with Constraint Handling (cEHVI).

    Implements the algorithm from Pang et al. (2023), proposing new sample
    locations to efficiently improve the Pareto front of expensive black-box
    functions. Constraint handling uses Probability of Feasibility (PoF).

    Key formula (Eq. 12, with corrected N_E denominator from Eq. 8 typo):
        EHVI(x) ≈ (1/N_E) * Σ_{q_i ∈ S_nd} HVI(q_i) · PDF_F(q_i | x)

    where:
        N_E    = total IS samples (denominator — NOT N_nd, paper typo in Eq. 8)
        S_nd   = uniform samples NOT dominated by current Pareto front
        HVI    = MC hypervolume contribution (count of S_nd weakly dominated)
        PDF_F  = product of normalized Gaussian PDFs (Eq. 17)

    Attributes:
        model (KRG): Pre-trained Kriging surrogate model.
        bounds (np.ndarray): Design variable bounds, shape (num_features, 2).
        obj_indices (List[int]): Output indices minimized as objectives.
        constraint_indices (Optional[List[int]]): Output indices for constraints.
        constraint_ubs (Optional[np.ndarray]): Upper bounds for each constraint.
        n_samples (int): Total IS sample count (N_E). Default 5000.
        n_candidates (int): Candidate pool size in propose(). Default 200.
        num_restarts (int): L-BFGS-B restarts for local refinement. Default 5.
        beta (float): Diversity selection hyperparameter. Default 0.3.
        obj_lb (float): IS sample lower bound (normalized space). Default -0.5.
        obj_ub (float): IS sample upper bound (normalized space). Default 1.2.
        y_obj_min (np.ndarray): Per-objective minimum from training data, shape (M,).
        y_obj_range (np.ndarray): Per-objective range from training data, shape (M,).
        pf_norm (np.ndarray): Normalized Pareto front, shape (n_pf, M).
        S_nd (np.ndarray): Non-dominated IS samples in normalized obj space, shape (N_nd, M).
        hvi_nd (np.ndarray): HVI weights for each S_nd point, shape (N_nd,).
        N_E (int): Total IS sample count (stored for EHVI denominator).
    """

    def __init__(
        self,
        model: KRG,
        bounds,
        y_train: np.ndarray,
        obj_indices: List[int],
        constraint_indices: Optional[List[int]] = None,
        constraint_ubs: Optional[np.ndarray] = None,
        n_samples: int = 5000,
        n_candidates: int = 200,
        num_restarts: int = 5,
        beta: float = 0.3,
        obj_lb: float = -0.5,
        obj_ub: float = 1.2,
    ):
        """
        Initialize MOInfill strategy.

        Args:
            model (KRG): A fitted Kriging model instance.
            bounds: Design variable bounds. shape: (num_features, 2).
            y_train (np.ndarray): Training output data. shape: (n_train, num_outputs).
            obj_indices (List[int]): Output indices treated as objectives (minimized).
            constraint_indices (Optional[List[int]]): Output indices for constraints.
            constraint_ubs (Optional[np.ndarray]): Upper bounds for constraint outputs.
            n_samples (int): Number of uniform IS samples (N_E). Default 5000.
            n_candidates (int): Candidate pool size in propose(). Default 200.
            num_restarts (int): L-BFGS-B restarts for local refinement. Default 5.
            beta (float): Diversity selection hyperparameter. Default 0.3.
            obj_lb (float): Lower bound for IS sample generation (normalized). Default -0.5.
            obj_ub (float): Upper bound for IS sample generation (normalized). Default 1.2.

        Raises:
            RuntimeError: If model is not fitted (model.beta is None).
            ValueError: If constraint_indices and constraint_ubs are inconsistent.
        """
        if not hasattr(model, "beta") or model.beta is None:
            raise RuntimeError("provided KRG model is not fitted.")

        self.model = model
        self.bounds = np.array(bounds, dtype=np.float64)
        if self.bounds.ndim == 1:
            self.bounds = self.bounds.reshape(-1, 2)

        self.obj_indices = list(obj_indices)
        self.constraint_indices = list(constraint_indices) if constraint_indices is not None else None
        if constraint_ubs is not None:
            self.constraint_ubs = np.array(constraint_ubs, dtype=np.float64)
            if self.constraint_indices is not None and len(self.constraint_ubs) != len(self.constraint_indices):
                raise ValueError(
                    f"constraint_ubs length {len(self.constraint_ubs)} must match "
                    f"constraint_indices length {len(self.constraint_indices)}."
                )
        else:
            self.constraint_ubs = None

        self.n_samples = int(n_samples)
        self.n_candidates = int(n_candidates)
        self.num_restarts = int(num_restarts)
        self.beta = float(beta)
        self.obj_lb = float(obj_lb)
        self.obj_ub = float(obj_ub)

        # Compute per-objective statistics from training data
        y_obj = np.array(y_train[:, self.obj_indices], dtype=np.float64)  # (n_train, M)
        self.y_obj_min = np.min(y_obj, axis=0)           # (M,)
        self.y_obj_max = np.max(y_obj, axis=0)           # (M,)
        self.y_obj_range = self.y_obj_max - self.y_obj_min  # (M,)
        # Guard against degenerate ranges (all training points identical in one dim)
        self.y_obj_range = np.where(self.y_obj_range < 1e-12, 1.0, self.y_obj_range)

        self._precompute_samples(y_obj)

    # ======================================================================
    # Pareto Utility
    # ======================================================================

    def _compute_pareto_mask(self, y: np.ndarray) -> np.ndarray:
        """
        Compute non-dominance mask for minimization objectives.

        Point i is non-dominated iff no other point j satisfies:
            y[j, k] <= y[i, k]  for all objectives k
            y[j, k] <  y[i, k]  for at least one objective k

        Args:
            y (np.ndarray): Objective values. shape: (n, M).

        Returns:
            np.ndarray: Boolean mask, True = non-dominated. shape: (n,).
        """
        # Vectorized pairwise dominance: diff[i, j, k] = y[j, k] - y[i, k]
        # j dominates i: all diff <= 0 AND some diff < 0
        y_i = y[:, np.newaxis, :]   # (n, 1, M)
        y_j = y[np.newaxis, :, :]   # (1, n, M)
        diff = y_j - y_i            # (n, n, M)
        dominated_by_j = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)  # (n, n)
        np.fill_diagonal(dominated_by_j, False)
        return ~np.any(dominated_by_j, axis=1)  # (n,)

    # ======================================================================
    # Pre-computation
    # ======================================================================

    def _precompute_samples(self, y_obj: np.ndarray) -> None:
        """
        Generate IS samples and compute HVI weights (called once at init).

        Steps:
        1. Draw N_E uniform samples S in [obj_lb, obj_ub]^M (normalized space).
        2. Compute normalized Pareto front from training objectives.
        3. Filter S to S_nd: rows NOT dominated by any PF point.
        4. Compute hvi_nd[i] = count of S_nd weakly dominated by S_nd[i]
           (MC hypervolume contribution, Eq. 7).

        Args:
            y_obj (np.ndarray): Training objective values. shape: (n_train, M).
        """
        M = len(self.obj_indices)
        N_E = self.n_samples

        # Step 1: Uniform IS samples in normalized objective space [obj_lb, obj_ub]^M
        S = np.random.uniform(self.obj_lb, self.obj_ub, size=(N_E, M))

        # Step 2: Normalize training objectives, extract Pareto front
        pf_norm_all = (y_obj - self.y_obj_min) / self.y_obj_range  # (n_train, M)
        pf_mask = self._compute_pareto_mask(pf_norm_all)
        self.pf_norm = pf_norm_all[pf_mask]  # (n_pf, M)

        n_pf = self.pf_norm.shape[0]

        # Step 3: Find S_nd — S rows NOT dominated by any Pareto front point
        # pf[k] dominates S[i]: pf[k,j] <= S[i,j] for all j AND < for some j
        # Process in chunks to limit peak memory to O(chunk * n_pf * M)
        chunk_size = 1000
        nd_mask = np.ones(N_E, dtype=bool)
        for start in range(0, N_E, chunk_size):
            end = min(start + chunk_size, N_E)
            S_chunk = S[start:end]                              # (chunk, M)
            S_exp = S_chunk[:, np.newaxis, :]                   # (chunk, 1, M)
            pf_exp = self.pf_norm[np.newaxis, :, :]            # (1, n_pf, M)
            diff = pf_exp - S_exp                               # (chunk, n_pf, M)
            # pf[k,j] - S[i,j] <= 0 for all j → pf[k] ≤ S[i]
            dominated = np.all(diff <= 0, axis=2) & np.any(diff < 0, axis=2)
            nd_mask[start:end] = ~np.any(dominated, axis=1)

        self.S_nd = S[nd_mask]   # (N_nd, M)
        self.N_E = N_E           # stored for EHVI denominator

        N_nd = self.S_nd.shape[0]

        # Step 4: hvi_nd[i] = count of S_nd[k] weakly dominated by S_nd[i]
        # Weak dominance: S_nd[i, j] <= S_nd[k, j] for ALL j
        # (diff = S_nd[k, j] - S_nd[i, j] >= 0 for all j)
        self.hvi_nd = np.zeros(N_nd, dtype=np.float64)
        chunk_size = 200
        for start in range(0, N_nd, chunk_size):
            end = min(start + chunk_size, N_nd)
            batch = self.S_nd[start:end]                        # (batch, M)
            b_exp = batch[:, np.newaxis, :]                     # (batch, 1, M)
            s_exp = self.S_nd[np.newaxis, :, :]                 # (1, N_nd, M)
            diff = s_exp - b_exp                                # (batch, N_nd, M)
            # S_nd[k, j] - batch[i, j] >= 0 for all j → batch[i] weakly dominates S_nd[k]
            self.hvi_nd[start:end] = np.sum(np.all(diff >= 0, axis=2), axis=1).astype(np.float64)

    # ======================================================================
    # Acquisition Functions
    # ======================================================================

    def _compute_pof_batch(self, mu_c: np.ndarray, sigma_c: np.ndarray) -> np.ndarray:
        """
        Compute Probability of Feasibility (PoF) for constraint satisfaction.

        PoF(x) = Π_j Φ((ub_j - μ_j(x)) / σ_j(x))

        Args:
            mu_c (np.ndarray): Constraint output means. shape: (N_c, num_constraints).
            sigma_c (np.ndarray): Constraint output std devs. shape: (N_c, num_constraints).

        Returns:
            np.ndarray: PoF values. shape: (N_c,).
        """
        z = (self.constraint_ubs[np.newaxis, :] - mu_c) / sigma_c  # (N_c, num_c)
        return np.prod(norm.cdf(z), axis=1)                         # (N_c,)

    def _compute_ehvi_batch(self, x_candidates: np.ndarray) -> np.ndarray:
        """
        Compute cEHVI (or EHVI) for a batch of design candidates.

        Implements Eq. 12 from Pang et al. (2023) with corrected N_E denominator:
            EHVI(x) ≈ (1/N_E) * Σ_{q_i ∈ S_nd} HVI(q_i) · PDF_F(q_i | x)

        Normalization follows Eq. 15-17; log-space accumulation for stability.
        If constraint_indices are set, multiplies by PoF to yield cEHVI.

        Args:
            x_candidates (np.ndarray): Candidate design points. shape: (N_c, num_features).

        Returns:
            np.ndarray: cEHVI values per candidate. shape: (N_c,).
        """
        N_c = x_candidates.shape[0]
        N_nd = self.S_nd.shape[0]

        # Degenerate case: all IS samples dominated by PF
        if N_nd == 0:
            return np.zeros(N_c)

        # Batch surrogate predictions
        mu_raw, var_raw = self.model.predict(x_candidates)  # (N_c, num_out) each

        # Extract objective dimensions
        mu_obj = mu_raw[:, self.obj_indices]    # (N_c, M)
        var_obj = var_raw[:, self.obj_indices]  # (N_c, M)

        # Mark candidates with zero predictive variance (already observed locations)
        zero_var = np.all(var_obj < 1e-12, axis=1)  # (N_c,)

        sigma_obj = np.sqrt(np.maximum(var_obj, 1e-12))  # (N_c, M)

        # Normalize predictions: Eq. 15-16
        mu_norm = (mu_obj - self.y_obj_min) / self.y_obj_range    # (N_c, M)
        sigma_norm = sigma_obj / self.y_obj_range                  # (N_c, M)

        # Compute log-PDF for each (candidate c, IS sample i, objective j)
        # z[c, i, j] = (S_nd[i, j] - mu_norm[c, j]) / sigma_norm[c, j]
        s_exp     = self.S_nd[np.newaxis, :, :]      # (1, N_nd, M)
        mu_exp    = mu_norm[:, np.newaxis, :]         # (N_c, 1, M)
        sigma_exp = sigma_norm[:, np.newaxis, :]      # (N_c, 1, M)

        with np.errstate(divide='ignore', invalid='ignore'):
            z = (s_exp - mu_exp) / sigma_exp         # (N_c, N_nd, M)

        # log PDF_F[c, i] = Σ_j log φ(z[c, i, j])   — Eq. 17
        log_phi   = norm.logpdf(z)                   # (N_c, N_nd, M)
        log_pdf_f = np.sum(log_phi, axis=2)          # (N_c, N_nd)
        pdf_f     = np.exp(log_pdf_f)                # (N_c, N_nd)

        # EHVI[c] = dot(hvi_nd, pdf_f[c]) / N_E      — Eq. 12 (corrected denominator)
        ehvi = (pdf_f @ self.hvi_nd) / self.N_E      # (N_c,)

        # Zero out degenerate (zero-variance) candidates
        ehvi[zero_var] = 0.0

        # Apply PoF multiplier for constraint handling → cEHVI
        if self.constraint_indices is not None and self.constraint_ubs is not None:
            mu_c  = mu_raw[:, self.constraint_indices]    # (N_c, num_c)
            var_c = var_raw[:, self.constraint_indices]   # (N_c, num_c)
            sigma_c = np.sqrt(np.maximum(var_c, 1e-12))
            pof = self._compute_pof_batch(mu_c, sigma_c)  # (N_c,)
            ehvi = ehvi * pof

        return ehvi

    def _compute_diversity_batch(self, mu_obj_norm: np.ndarray) -> np.ndarray:
        """
        Compute diversity score for candidates in normalized objective space (Eq. 19).

        d(x) = min_{y_PF ∈ PF} ||μ_obj(x) - y_PF||_2

        Args:
            mu_obj_norm (np.ndarray): Normalized predicted objectives. shape: (N_c, M).

        Returns:
            np.ndarray: Min-distance-to-PF per candidate. shape: (N_c,).
        """
        # (N_c, 1, M) - (1, n_pf, M) → (N_c, n_pf, M)
        diff = mu_obj_norm[:, np.newaxis, :] - self.pf_norm[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N_c, n_pf)
        return np.min(dists, axis=1)                 # (N_c,)

    # ======================================================================
    # Public API
    # ======================================================================

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate cEHVI at arbitrary design points.

        Args:
            x (np.ndarray): Design points. shape: (N, num_features).

        Returns:
            np.ndarray: cEHVI values. shape: (N,).
        """
        return self._compute_ehvi_batch(x)

    def propose(self) -> np.ndarray:
        """
        Propose a single new sampling point using IS-EHVI and diversity selection.

        Algorithm (Eq. 19-20 from Pang et al. 2023):
        1. Generate n_candidates uniform random candidates in design space.
        2. Compute cEHVI for all candidates.
        3. Select top-w% subset P_w (w ~ U(0, beta), min 5%).
        4. Among P_w, choose x_new = argmax diversity (min distance to PF).
        5. Local L-BFGS-B refinement from best candidate.

        Returns:
            np.ndarray: Proposed design point. shape: (1, num_features).
        """
        num_features = self.bounds.shape[0]

        # Step 1: Uniform random candidate pool
        x_candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(self.n_candidates, num_features),
        )

        # Step 2: Compute cEHVI for all candidates
        ehvi = self._compute_ehvi_batch(x_candidates)  # (N_c,)

        # Step 3: Select top-w% by cEHVI (P_w) — Eq. 20
        w = np.random.uniform(0.0, self.beta)
        frac = max(w, 0.05)  # clamp to at least 5% to avoid empty P_w
        n_top = max(1, int(self.n_candidates * frac))
        top_indices = np.argsort(ehvi)[::-1][:n_top]
        P_w = x_candidates[top_indices]  # (n_top, num_features)

        # Step 4: Diversity selection within P_w — Eq. 19
        mu_raw_pw, _ = self.model.predict(P_w)
        mu_obj_pw = mu_raw_pw[:, self.obj_indices]                    # (n_top, M)
        mu_norm_pw = (mu_obj_pw - self.y_obj_min) / self.y_obj_range  # (n_top, M)
        diversity = self._compute_diversity_batch(mu_norm_pw)         # (n_top,)

        best_candidate = P_w[int(np.argmax(diversity))]

        # Step 5: Local L-BFGS-B refinement from best candidate
        scipy_bounds = Bounds(self.bounds[:, 0], self.bounds[:, 1])

        def neg_ehvi(x_vec: np.ndarray) -> float:
            return -float(self._compute_ehvi_batch(x_vec[np.newaxis, :])[0])

        best_x = best_candidate.copy()
        best_val = -neg_ehvi(best_x)

        for _ in range(self.num_restarts):
            try:
                res = minimize(neg_ehvi, x0=best_candidate, bounds=scipy_bounds, method="L-BFGS-B")
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_x = res.x
            except Exception:
                continue

        if best_x is None:
            warnings.warn("L-BFGS-B failed entirely; returning diversity-selected candidate.", RuntimeWarning)
            best_x = best_candidate

        return best_x[np.newaxis, :]
