# Physics-Informed Criterion for Compressible Flow
# Author: Shengning Wang

import torch
from torch import Tensor
from typing import Optional, List

from wsnet.training.base_criterion import BaseCriterion


# ============================================================
# Finite-Difference helpers
# Support 2D grid [G1,G2] and 3D grid [G1,G2,G3]
# ============================================================

def _scatter_to_grid(field: Tensor, coords: Tensor, grid_size: List[int]) -> Tensor:
    """
    Scatter-mean: map N unstructured nodes to a regular grid via nearest-neighbor binning.
    Supports both 2D ([G1,G2]) and 3D ([G1,G2,G3]) grids.

    Args:
        field:     (B, N, C) — node features to project.
        coords:    (B, N, D) — normalized coordinates in [-1, 1], D = spatial_dim.
        grid_size: [G1, G2] or [G1, G2, G3]

    Returns:
        (B, C, G1, G2) or (B, C, G1, G2, G3)
    """
    B, N, C = field.shape
    D = len(grid_size)
    device = field.device

    total_cells = 1
    for g in grid_size:
        total_cells *= g

    # Map coords [-1,1] to nearest grid indices
    dims = torch.tensor(grid_size, dtype=torch.float32, device=device).view(1, 1, D)
    unnorm = (coords + 1.0) / 2.0 * (dims - 1.0)
    indices = unnorm.round().long()
    for d, g in enumerate(grid_size):
        indices[..., d] = indices[..., d].clamp(0, g - 1)

    # Row-major flat index: stride[d] = product of grid_size[d+1:]
    strides = []
    s = 1
    for g in reversed(grid_size):
        strides.append(s)
        s *= g
    strides = list(reversed(strides))
    strides_t = torch.tensor(strides, dtype=torch.long, device=device).view(1, 1, D)
    flat_idx = (indices * strides_t).sum(dim=-1)                # (B, N)

    batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)
    global_idx = (batch_idx * total_cells + flat_idx).view(-1)  # (B*N,)

    flat_field = field.reshape(-1, C)
    grid_flat = torch.zeros(B * total_cells, C, device=device, dtype=field.dtype)
    counts = torch.zeros(B * total_cells, 1, device=device, dtype=field.dtype)

    grid_flat.index_add_(0, global_idx, flat_field)
    counts.index_add_(0, global_idx,
                      torch.ones(B * N, 1, device=device, dtype=field.dtype))

    grid_flat = grid_flat / counts.clamp(min=1.0)

    # (B * total_cells, C) -> (B, G1, ..., GD, C) -> (B, C, G1, ..., GD)
    grid = grid_flat.view(B, *grid_size, C)
    perm = [0, D + 1] + list(range(1, D + 1))
    return grid.permute(*perm).contiguous()


def _central_diff(tensor: Tensor, dim: int) -> Tensor:
    """
    Central difference along spatial dimension `dim` (dim >= 1, 1-indexed from batch).
    Boundaries use first-order forward/backward difference.
    """
    out = torch.zeros_like(tensor)
    ndim = tensor.dim()
    # Interior: central difference
    slc_fwd = [slice(None)] * ndim; slc_fwd[dim] = slice(2, None)
    slc_bwd = [slice(None)] * ndim; slc_bwd[dim] = slice(None, -2)
    slc_out = [slice(None)] * ndim; slc_out[dim] = slice(1, -1)
    out[tuple(slc_out)] = (tensor[tuple(slc_fwd)] - tensor[tuple(slc_bwd)]) * 0.5
    # Forward boundary (first slice)
    slc_f1 = [slice(None)] * ndim; slc_f1[dim] = slice(1, 2)
    slc_f0 = [slice(None)] * ndim; slc_f0[dim] = slice(0, 1)
    slc_o0 = [slice(None)] * ndim; slc_o0[dim] = slice(0, 1)
    out[tuple(slc_o0)] = tensor[tuple(slc_f1)] - tensor[tuple(slc_f0)]
    # Backward boundary (last slice)
    slc_lm1 = [slice(None)] * ndim; slc_lm1[dim] = slice(-2, -1)
    slc_l   = [slice(None)] * ndim; slc_l[dim]   = slice(-1, None)
    slc_ol  = [slice(None)] * ndim; slc_ol[dim]  = slice(-1, None)
    out[tuple(slc_ol)] = tensor[tuple(slc_l)] - tensor[tuple(slc_lm1)]
    return out


def _grid_divergence(vector_grid: Tensor) -> Tensor:
    """
    Divergence of a vector field on a regular grid using central differences.
    Supports 2D and 3D grids.

    Args:
        vector_grid: (B, D, *spatial) where D = spatial_dim.
            2D: (B, 2, H, W);  3D: (B, 3, X, Y, Z)

    Returns:
        (B, *spatial) — div(V) = sum_d dV_d/dx_d
    """
    D = vector_grid.shape[1]
    div = torch.zeros_like(vector_grid[:, 0])
    for d in range(D):
        div = div + _central_diff(vector_grid[:, d], dim=d + 1)
    return div


def _grid_gradient(scalar_grid: Tensor) -> Tensor:
    """
    Gradient of a scalar field on a regular grid using central differences.
    Supports 2D and 3D grids.

    Args:
        scalar_grid: (B, *spatial)
            2D: (B, H, W);  3D: (B, X, Y, Z)

    Returns:
        (B, D, *spatial) — [df/dx_0, df/dx_1, ...]
    """
    spatial_dims = scalar_grid.dim() - 1
    grads = [_central_diff(scalar_grid, dim=d + 1) for d in range(spatial_dims)]
    return torch.stack(grads, dim=1)


# ============================================================
# Flow Criterion
# ============================================================

class FlowCriterion(BaseCriterion):
    """
    Physics-informed loss for compressible flow (FD-based, no autograd).

    Total loss = per-channel NMSE(pred, target) + lambda_physics * L_physics

    Data loss uses per-channel NMSE: for each channel c, compute
    NMSE_c = sum((t_c - p_c)^2) / sum(t_c^2), then sum across channels.
    This prevents high-magnitude channels from dominating the loss.

    Physics residuals enforce compressible Euler equations on a scatter-projected
    regular grid. Density is inferred via ideal gas proportionality rho ~ p/T
    in normalized space (no physical constants needed).

    Channel layout (inferred from coords.shape[-1] = spatial_dim):
        2D: [Vx, Vy, P_log, T]        v=[:2], p=[2], T=[3]
        3D: [Vx, Vy, Vz, P_log, T]    v=[:3], p=[3], T=[4]

    Compressible Euler residuals:
        - Mass:     d(rho)/dt + div(rho * v) ≈ 0
        - Momentum: dv/dt + (1/rho) * grad(p) ≈ 0
        - Energy:   dT/dt + v · grad(T) ≈ 0

    Args:
        lambda_physics:  Weight of physics loss relative to data (NMSE) loss.
        lambda_mass:     Sub-weight for mass residual.
        lambda_momentum: Sub-weight for momentum residual.
        lambda_energy:   Sub-weight for energy residual.
        eps:             Numerical stability constant.
    """

    def __init__(
        self,
        lambda_physics: float = 0.1,
        lambda_mass: float = 1.0,
        lambda_momentum: float = 1.0,
        lambda_energy: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if any(v < 0 for v in (lambda_physics, lambda_mass, lambda_momentum, lambda_energy)):
            raise ValueError("All lambda weights must be non-negative")

        self.lambda_physics = lambda_physics
        self.lambda_mass = lambda_mass
        self.lambda_momentum = lambda_momentum
        self.lambda_energy = lambda_energy
        self.eps = eps

    def _physics_loss(
        self,
        pred: Tensor,
        prev: Tensor,
        coords: Tensor,
        latent_grid_size: List[int],
        dt: float,
    ) -> Tensor:
        """
        Compute compressible Euler physics residual loss via FD on latent grid.

        Uses ideal gas proportionality rho ~ p/T in normalized space to infer
        density, then enforces compressible continuity, momentum, and energy
        conservation residuals.

        Args:
            pred:             (B, N, C) — predicted state at t+1
            prev:             (B, N, C) — clean state at t (no noise)
            coords:           (B, N, D) — normalized coords in [-1, 1]
            latent_grid_size: [G1, G2] or [G1, G2, G3]
            dt:               Time step size (normalized)
        """
        sd = coords.shape[-1]  # spatial_dim: 2 or 3

        # Project to regular grid: (B, C, *spatial)
        pred_grid = _scatter_to_grid(pred, coords, latent_grid_size)
        prev_grid = _scatter_to_grid(prev, coords, latent_grid_size)

        # Channel indexing: [Vx, Vy, (Vz,) P_log, T]
        v_pred = pred_grid[:, :sd]       # (B, sd, *spatial)
        p_pred = pred_grid[:, sd]        # (B, *spatial)
        T_pred = pred_grid[:, sd + 1]    # (B, *spatial)

        v_prev = prev_grid[:, :sd]
        p_prev = prev_grid[:, sd]
        T_prev = prev_grid[:, sd + 1]

        # Infer density from normalized space (proportional to p/T)
        rho_pred = p_pred / (T_pred + self.eps)
        rho_prev = p_prev / (T_prev + self.eps)

        # Mass conservation: d(rho)/dt + div(rho * v) ~ 0
        drho_dt = (rho_pred - rho_prev) / dt
        rho_v = rho_pred.unsqueeze(1) * v_pred   # (B, sd, *spatial)
        loss_mass = torch.mean((drho_dt + _grid_divergence(rho_v)) ** 2)

        # Momentum conservation: dv/dt + (1/rho) * grad(p) ~ 0
        dv_dt = (v_pred - v_prev) / dt
        grad_p = _grid_gradient(p_pred)
        inv_rho = 1.0 / (rho_pred.unsqueeze(1) + self.eps)
        loss_momentum = torch.mean((dv_dt + inv_rho * grad_p) ** 2)

        # Energy conservation: dT/dt + v . grad(T) ~ 0  (convective transport)
        dT_dt = (T_pred - T_prev) / dt
        grad_T = _grid_gradient(T_pred)
        v_dot_gradT = (v_pred * grad_T).sum(dim=1)
        loss_energy = torch.mean((dT_dt + v_dot_gradT) ** 2)

        return (self.lambda_mass * loss_mass +
                self.lambda_momentum * loss_momentum +
                self.lambda_energy * loss_energy)

    def forward(
        self,
        pred: Tensor,
        target: Optional[Tensor] = None,
        prev: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
        latent_grid_size: Optional[List[int]] = None,
        dt: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """
        Compute total loss = NMSE + lambda_physics * physics_loss.

        Args:
            pred:             Predicted state. Shape (B, N, C).
            target:           Ground truth. Shape (B, N, C). Required for data loss.
            prev:             Clean previous state at t. Shape (B, N, C). Required for physics.
            coords:           Normalized node coordinates. Shape (B, N, D). Required for physics.
            latent_grid_size: [L1, L2] or [L1, L2, L3] for FD grid. Required for physics.
            dt:               Time step (normalized). Default 1.0.
            **kwargs:         Ignored (for API compatibility).

        Returns:
            Scalar loss tensor.
        """
        # Data loss (per-channel NMSE)
        if target is not None:
            C = pred.shape[-1]
            sq_err = (target - pred) ** 2
            mse_c = sq_err.reshape(-1, C).sum(0)        # (C,)
            norm_c = (target ** 2).reshape(-1, C).sum(0) + self.eps  # (C,)
            data_loss = (mse_c / norm_c).sum()
        else:
            data_loss = torch.tensor(0.0, device=pred.device)

        # Physics loss (only when all required inputs are present)
        if prev is not None and coords is not None and latent_grid_size is not None:
            phy_loss = self._physics_loss(pred, prev, coords, latent_grid_size, dt)
            return data_loss + self.lambda_physics * phy_loss

        return data_loss
