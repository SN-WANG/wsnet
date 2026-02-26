# Physics-Informed Criterion for Compressible Flow
# Author: Shengning Wang

import torch
from torch import Tensor
from typing import Optional, List

from wsnet.training.base_criterion import BaseCriterion


# ============================================================
# Finite-Difference helpers (module-level, no autograd)
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
# Base class (kept minimal; autograd helpers removed)
# ============================================================

class PhysicsCriterion(BaseCriterion):
    """Base class for physics-constrained loss functions."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps


# ============================================================
# Compressible Flow Criterion
# ============================================================

class CompressibleFlowCriterion(PhysicsCriterion):
    """
    Physics-informed loss for compressible flow (FD-based, no autograd).

    Total loss = NMSE(pred, target) + lambda_phy * L_physics

    Physics residuals are computed in normalized feature space via finite differences
    on a scatter-projected regular grid. Zero-residual is scale-invariant, so working
    in normalized space is valid without denormalization.

    Channel layout (inferred from coords.shape[-1] = spatial_dim):
        2D: [Vx, Vy, P_log, T]        v=[:2], p=[2], T=[3]
        3D: [Vx, Vy, Vz, P_log, T]    v=[:3], p=[3], T=[4]

    Kinematic residuals enforced (density-free, no ideal-gas law needed):
        - Mass proxy:     ∇·v ≈ 0
        - Momentum proxy: ∂v/∂t + ∇p ≈ 0
        - Energy proxy:   ∂T/∂t ≈ 0

    Args:
        lambda_phy:      Weight of physics loss relative to data (NMSE) loss.
        lambda_mass:     Sub-weight for mass residual.
        lambda_momentum: Sub-weight for momentum residual.
        lambda_energy:   Sub-weight for energy residual.
        eps:             Numerical stability constant.
    """

    def __init__(
        self,
        lambda_phy: float = 0.1,
        lambda_mass: float = 1.0,
        lambda_momentum: float = 1.0,
        lambda_energy: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__(eps=eps)

        if any(v < 0 for v in (lambda_phy, lambda_mass, lambda_momentum, lambda_energy)):
            raise ValueError("All lambda weights must be non-negative")

        self.lambda_phy = lambda_phy
        self.lambda_mass = lambda_mass
        self.lambda_momentum = lambda_momentum
        self.lambda_energy = lambda_energy

    def _physics_loss(
        self,
        pred: Tensor,
        prev: Tensor,
        coords: Tensor,
        latent_grid_size: List[int],
        dt: float,
    ) -> Tensor:
        """
        Compute physics residual loss via FD on latent grid.

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

        # Temporal FD
        d_dt = (pred_grid - prev_grid) / dt

        # Channel indexing: [Vx, Vy, (Vz,) P_log, T]
        v_grid = pred_grid[:, :sd]   # (B, sd, *spatial)
        p_grid = pred_grid[:, sd]    # (B, *spatial)

        # Mass proxy: ∇·v ≈ 0
        loss_mass = torch.mean(_grid_divergence(v_grid) ** 2)

        # Momentum proxy: ∂v/∂t + ∇p ≈ 0
        dv_dt = d_dt[:, :sd]
        grad_p = _grid_gradient(p_grid)
        loss_momentum = torch.mean((dv_dt + grad_p) ** 2)

        # Energy proxy: ∂T/∂t ≈ 0
        dT_dt = d_dt[:, sd + 1]
        loss_energy = torch.mean(dT_dt ** 2)

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
        Compute total loss = NMSE + lambda_phy * physics_loss.

        Args:
            pred:             Predicted state. Shape (B, N, C).
            target:           Ground truth. Shape (B, N, C). Required for data loss.
            prev:             Clean previous state at t. Shape (B, N, C). Required for physics.
            coords:           Normalized node coordinates. Shape (B, N, D). Required for physics.
            latent_grid_size: [G1, G2] or [G1, G2, G3] for FD grid. Required for physics.
            dt:               Time step (normalized). Default 1.0.
            **kwargs:         Ignored (for API compatibility).

        Returns:
            Scalar loss tensor.
        """
        # Data loss (NMSE)
        if target is not None:
            mse = torch.sum((target - pred) ** 2)
            norm = torch.sum(target ** 2) + self.eps
            data_loss = mse / norm
        else:
            data_loss = torch.tensor(0.0, device=pred.device)

        # Physics loss (only when all required inputs are present)
        if prev is not None and coords is not None and latent_grid_size is not None:
            phy_loss = self._physics_loss(pred, prev, coords, latent_grid_size, dt)
            return data_loss + self.lambda_phy * phy_loss

        return data_loss
