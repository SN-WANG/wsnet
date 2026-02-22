# Physics-Informed Criterion for Compressible Flow
# Author: Shengning Wang

import torch
from torch import Tensor
from typing import Optional

from wsnet.training.base_criterion import BaseCriterion


class PhysicsCriterion(BaseCriterion):
    """Base class for physics-constrained loss functions.

    Provides interface for losses that enforce physical constraints
    such as conservation laws (mass, momentum, energy).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _compute_time_derivative(
        self,
        field: Tensor,
        time: Tensor
    ) -> Tensor:
        """Compute time derivative using automatic differentiation.

        Args:
            field: Scalar or vector field. Shape (B, N, ...) or (N, ...).
            time: Time scalar or tensor with requires_grad=True.

        Returns:
            Time derivative of the field. Same shape as field.

        Raises:
            RuntimeError: If time does not have requires_grad=True.
        """
        if not time.requires_grad:
            raise RuntimeError("Time tensor must have requires_grad=True for gradient computation")

        return torch.autograd.grad(
            outputs=field,
            inputs=time,
            grad_outputs=torch.ones_like(field),
            create_graph=True,
            retain_graph=True
        )[0]

    def _compute_divergence_2d(
        self,
        vector_field: Tensor,
        coords: Tensor
    ) -> Tensor:
        """Compute 2D divergence of a vector field.

        Computes: div(V) = dVx/dx + dVy/dy

        Args:
            vector_field: Vector field with 2 components. Shape (B, N, 2) or (N, 2).
                vector_field[..., 0] is x-component, [..., 1] is y-component.
            coords: Spatial coordinates [x, y]. Shape (B, N, 2) or (N, 2).

        Returns:
            Divergence of the vector field. Shape (B, N) or (N,).

        Raises:
            ValueError: If tensor dimensions are incorrect.
        """
        vx = vector_field[..., 0]  # x-component
        vy = vector_field[..., 1]  # y-component

        x = coords[..., 0]
        y = coords[..., 1]

        dvx_dx = torch.autograd.grad(
            outputs=vx,
            inputs=x,
            grad_outputs=torch.ones_like(vx),
            create_graph=True,
            retain_graph=True
        )[0]

        dvy_dy = torch.autograd.grad(
            outputs=vy,
            inputs=y,
            grad_outputs=torch.ones_like(vy),
            create_graph=True,
            retain_graph=True
        )[0]

        return dvx_dx + dvy_dy

    def _compute_gradient_2d(
        self,
        scalar_field: Tensor,
        coords: Tensor
    ) -> Tensor:
        """Compute 2D gradient of a scalar field.

        Computes: grad(f) = [df/dx, df/dy]

        Args:
            scalar_field: Scalar field. Shape (B, N) or (N,).
            coords: Spatial coordinates [x, y]. Shape (B, N, 2) or (N, 2).

        Returns:
            Gradient of the scalar field. Shape (B, N, 2) or (N, 2).

        Raises:
            ValueError: If tensor dimensions are incorrect.
        """
        x = coords[..., 0]
        y = coords[..., 1]

        df_dx = torch.autograd.grad(
            outputs=scalar_field,
            inputs=x,
            grad_outputs=torch.ones_like(scalar_field),
            create_graph=True,
            retain_graph=True
        )[0]

        df_dy = torch.autograd.grad(
            outputs=scalar_field,
            inputs=y,
            grad_outputs=torch.ones_like(scalar_field),
            create_graph=True,
            retain_graph=True
        )[0]

        return torch.stack([df_dx, df_dy], dim=-1)


class CompressibleFlowCriterion(PhysicsCriterion):
    """Physics-informed loss for compressible flow conservation laws.

    Enforces mass, momentum, and energy conservation for compressible
    Navier-Stokes equations. Based on the paper's methodology for
    high-pressure-ratio transient flows.

    Governing Equations (Euler form, inviscid):
        1. Continuity (Mass): ∂ρ/∂t + ∇·(ρv) = 0
        2. Momentum: ∂(ρv)/∂t + ∇·(ρv⊗v) + ∇p = 0
        3. Energy: ∂(ρE)/∂t + ∇·[(ρE + p)v] = 0
        4. State Equation: p = ρRT

    Attributes:
        R: Gas constant for hydrogen (J/(kg·K)). Default 4124.0.
        cv: Specific heat at constant volume (J/(kg·K)). Default 10120.0.
        lambda_mass: Weight for mass conservation residual.
        lambda_momentum: Weight for momentum conservation residual.
        lambda_energy: Weight for energy conservation residual.
        eps: Small constant for numerical stability.
    """

    HYDROGEN_R: float = 4124.0  # J/(kg·K)
    HYDROGEN_CV: float = 10120.0  # J/(kg·K)

    def __init__(
        self,
        R: float = None,
        cv: float = None,
        lambda_mass: float = 1.0,
        lambda_momentum: float = 1.0,
        lambda_energy: float = 1.0,
        eps: float = 1e-8
    ):
        super().__init__(eps=eps)

        self.R = R if R is not None else self.HYDROGEN_R
        self.cv = cv if cv is not None else self.HYDROGEN_CV

        if lambda_mass < 0 or lambda_momentum < 0 or lambda_energy < 0:
            raise ValueError("Lambda weights must be non-negative")

        self.lambda_mass = lambda_mass
        self.lambda_momentum = lambda_momentum
        self.lambda_energy = lambda_energy

    def forward(
        self,
        pred: Tensor,
        target: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Compute physics residuals for compressible flow.

        Args:
            pred: Predicted flow field [vx, vy, p, T].
                Shape (B, N, 4) where:
                - B: batch_size, N: num_nodes
                - pred[..., 0]: velocity x-component (vx)
                - pred[..., 1]: velocity y-component (vy)
                - pred[..., 2]: pressure (p)
                - pred[..., 3]: temperature (T)
            target: Ground truth (optional, unused in physics loss).
            coords: Spatial coordinates [x, y]. Shape (B, N, 2) or (N, 2).
                Required for spatial gradients.
            time: Time scalar with requires_grad=True for temporal derivatives.
            **kwargs: Additional arguments (ignored).

        Returns:
            Tensor: Scalar total loss.

        Raises:
            ValueError: If input shapes are incorrect or coords/time are missing.
        """
        if pred.dim() != 3 or pred.shape[-1] != 4:
            raise ValueError(
                f"pred must have shape (B, N, 4), got {pred.shape}"
            )

        if coords is None:
            raise ValueError("coords must be provided for spatial gradient computation")
        if time is None:
            raise ValueError("time must be provided for temporal derivative computation")

        # Unpack prediction variables
        vx = pred[..., 0]        # (B, N)
        vy = pred[..., 1]        # (B, N)
        p = pred[..., 2]         # (B, N)
        T = pred[..., 3]         # (B, N)

        # Compute density from ideal gas law: ρ = p / (R * T)
        rho = p / (self.R * T + self.eps)  # (B, N)

        # Compute velocity vector and momentum
        v = torch.stack([vx, vy], dim=-1)           # (B, N, 2)
        rho_v = rho.unsqueeze(-1) * v               # (B, N, 2)

        # Compute total energy: E = cv * T + 0.5 * |v|^2
        kinetic_energy = 0.5 * (vx ** 2 + vy ** 2)  # (B, N)
        internal_energy = self.cv * T                  # (B, N)
        E = internal_energy + kinetic_energy           # (B, N)
        rho_E = rho * E                              # (B, N)

        # Compute total enthalpy: ρE + p
        rho_E_p = rho_E + p                         # (B, N)

        # ==============================
        # Compute Conservation Residuals
        # ==============================

        # Time derivatives
        drho_dt = self._compute_time_derivative(rho, time)                  # (B, N)
        drho_v_dt = self._compute_time_derivative(rho_v, time)              # (B, N, 2)
        drho_E_dt = self._compute_time_derivative(rho_E, time)              # (B, N)

        # Spatial derivatives
        div_rho_v = self._compute_divergence_2d(rho_v, coords)             # (B, N)

        # Momentum convection: ∇·(ρv⊗v)
        # Outer product: (ρv_i * v_j)
        v_expanded = v.unsqueeze(-1)                    # (B, N, 2, 1)
        rho_v_expanded = rho_v.unsqueeze(-1)            # (B, N, 2, 1)
        rho_v_outer_v = rho_v_expanded * v_expanded.transpose(-2, -1)  # (B, N, 2, 2)

        # Tensor divergence
        def _tensor_divergence_2d(tensor: Tensor, coords: Tensor) -> Tensor:
            """Compute divergence of a 2D tensor field."""
            x = coords[..., 0]
            y = coords[..., 1]

            # First row
            t11 = tensor[..., 0, 0]
            t12 = tensor[..., 0, 1]
            dt11_dx = torch.autograd.grad(t11, x, torch.ones_like(t11), create_graph=True, retain_graph=True)[0]
            dt12_dy = torch.autograd.grad(t12, y, torch.ones_like(t12), create_graph=True, retain_graph=True)[0]
            div_x = dt11_dx + dt12_dy

            # Second row
            t21 = tensor[..., 1, 0]
            t22 = tensor[..., 1, 1]
            dt21_dx = torch.autograd.grad(t21, x, torch.ones_like(t21), create_graph=True, retain_graph=True)[0]
            dt22_dy = torch.autograd.grad(t22, y, torch.ones_like(t22), create_graph=True, retain_graph=True)[0]
            div_y = dt21_dx + dt22_dy

            return torch.stack([div_x, div_y], dim=-1)

        div_rho_v_v = _tensor_divergence_2d(rho_v_outer_v, coords)  # (B, N, 2)

        # Pressure gradient
        grad_p = self._compute_gradient_2d(p, coords)                # (B, N, 2)

        # Energy flux divergence: ∇·[(ρE + p)v]
        energy_flux = rho_E_p.unsqueeze(-1) * v                    # (B, N, 2)
        div_energy_flux = self._compute_divergence_2d(energy_flux, coords)  # (B, N)

        # ==============================
        # Assemble Residuals
        # ==============================

        # Continuity (mass) residual: ∂ρ/∂t + ∇·(ρv)
        residual_mass = drho_dt + div_rho_v                           # (B, N)

        # Momentum residual: ∂(ρv)/∂t + ∇·(ρv⊗v) + ∇p
        residual_momentum = drho_v_dt + div_rho_v_v + grad_p            # (B, N, 2)

        # Energy residual: ∂(ρE)/∂t + ∇·[(ρE + p)v]
        residual_energy = drho_E_dt + div_energy_flux                   # (B, N)

        # ==============================
        # Compute Loss Components
        # ==============================

        # L2 norm squared of each residual
        loss_mass = torch.mean(residual_mass ** 2)                     # ()
        loss_momentum = torch.mean(torch.sum(residual_momentum ** 2, dim=-1))  # ()
        loss_energy = torch.mean(residual_energy ** 2)                   # ()

        # Weighted sum
        total = (
            self.lambda_mass * loss_mass +
            self.lambda_momentum * loss_momentum +
            self.lambda_energy * loss_energy
        )

        return total
