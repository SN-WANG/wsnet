# Geometry-aware Wavelet Neural Operator (Geo-WNO)
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Optional

import math


# ============================================================
# Haar DWT/IDWT — Pure PyTorch, no 3rd-party wavelet libs
# ============================================================

class _HaarDWT1d(nn.Module):
    """1D Haar DWT: (B, C, L) -> (approx, detail, orig_L)."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))   # low-pass
        self.register_buffer('h_H', torch.tensor([s, -s]))  # high-pass

    def forward(self, x: Tensor):
        B, C, L = x.shape
        if L % 2 == 1:
            x = F.pad(x, (0, 1))
        x_bc = x.view(B * C, 1, x.shape[-1])
        approx = F.conv1d(x_bc, self.h_L.view(1, 1, 2), stride=2).view(B, C, -1)
        detail = F.conv1d(x_bc, self.h_H.view(1, 1, 2), stride=2).view(B, C, -1)
        return approx, detail, L


class _HaarIDWT1d(nn.Module):
    """1D Haar IDWT: (approx, detail, orig_L) -> (B, C, orig_L)."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))
        self.register_buffer('h_H', torch.tensor([s, -s]))

    def forward(self, approx: Tensor, detail: Tensor, orig_L: int) -> Tensor:
        B, C, _ = approx.shape
        approx_bc = approx.view(B * C, 1, -1)
        detail_bc = detail.view(B * C, 1, -1)
        rec_bc = (F.conv_transpose1d(approx_bc, self.h_L.view(1, 1, 2), stride=2) +
                  F.conv_transpose1d(detail_bc, self.h_H.view(1, 1, 2), stride=2))
        return rec_bc.view(B, C, -1)[..., :orig_L]


class _HaarDWT2d(nn.Module):
    """
    2D separable Haar DWT: (B, C, H, W) -> (LL, LH, HL, HH, orig_H, orig_W).

    Step 1: DWT along W (last dim) -> L_w, H_w
    Step 2: DWT along H (2nd-to-last) for each -> LL, LH (from L_w), HL, HH (from H_w)
    Each subband: (B, C, ceil(H/2), ceil(W/2))
    """

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))
        self.register_buffer('h_H', torch.tensor([s, -s]))

    def _dwt_along_w(self, x: Tensor):
        B, C, H, W = x.shape
        x_flat = x.reshape(B * C * H, 1, W)
        if W % 2 == 1:
            x_flat = F.pad(x_flat, (0, 1))
        L = F.conv1d(x_flat, self.h_L.view(1, 1, 2), stride=2).reshape(B, C, H, -1)
        H_ = F.conv1d(x_flat, self.h_H.view(1, 1, 2), stride=2).reshape(B, C, H, -1)
        return L, H_

    def _dwt_along_h(self, x: Tensor):
        B, C, H, W_out = x.shape
        x_perm = x.permute(0, 1, 3, 2).reshape(B * C * W_out, 1, H)
        if H % 2 == 1:
            x_perm = F.pad(x_perm, (0, 1))
        Lo = F.conv1d(x_perm, self.h_L.view(1, 1, 2), stride=2).reshape(B, C, W_out, -1).permute(0, 1, 3, 2)
        Hi = F.conv1d(x_perm, self.h_H.view(1, 1, 2), stride=2).reshape(B, C, W_out, -1).permute(0, 1, 3, 2)
        return Lo, Hi

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        L_w, H_w = self._dwt_along_w(x)
        LL, LH = self._dwt_along_h(L_w)
        HL, HH = self._dwt_along_h(H_w)
        return LL, LH, HL, HH, H, W


class _HaarIDWT2d(nn.Module):
    """2D separable Haar IDWT: (LL, LH, HL, HH, orig_H, orig_W) -> (B, C, orig_H, orig_W)."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))
        self.register_buffer('h_H', torch.tensor([s, -s]))

    def _idwt_along_h(self, Lo: Tensor, Hi: Tensor, orig_H: int) -> Tensor:
        B, C, H_out, W_out = Lo.shape
        Lo_perm = Lo.permute(0, 1, 3, 2).reshape(B * C * W_out, 1, H_out)
        Hi_perm = Hi.permute(0, 1, 3, 2).reshape(B * C * W_out, 1, H_out)
        rec = (F.conv_transpose1d(Lo_perm, self.h_L.view(1, 1, 2), stride=2) +
               F.conv_transpose1d(Hi_perm, self.h_H.view(1, 1, 2), stride=2))
        return rec.reshape(B, C, W_out, -1).permute(0, 1, 3, 2)[..., :orig_H, :]

    def forward(self, LL: Tensor, LH: Tensor, HL: Tensor, HH: Tensor,
                orig_H: int, orig_W: int) -> Tensor:
        B, C, H_out, W_out = LL.shape
        L_w = self._idwt_along_h(LL, LH, orig_H)
        H_w = self._idwt_along_h(HL, HH, orig_H)
        # IDWT along W
        L_flat = L_w.reshape(B * C * orig_H, 1, W_out)
        H_flat = H_w.reshape(B * C * orig_H, 1, W_out)
        rec = (F.conv_transpose1d(L_flat, self.h_L.view(1, 1, 2), stride=2) +
               F.conv_transpose1d(H_flat, self.h_H.view(1, 1, 2), stride=2))
        return rec.reshape(B, C, orig_H, -1)[..., :orig_W]


class _HaarDWT3d(nn.Module):
    """
    3D separable Haar DWT: (B, C, X, Y, Z) -> 8 subbands + (X, Y, Z).

    Subbands returned as a tuple: (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH)
    Each: (B, C, ceil(X/2), ceil(Y/2), ceil(Z/2))
    """

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))
        self.register_buffer('h_H', torch.tensor([s, -s]))

    def _dwt_along_dim(self, x: Tensor, dim: int):
        # Move target dim to last, apply conv, restore
        x = x.transpose(dim, -1)
        shape = x.shape
        L_orig = shape[-1]
        x_flat = x.reshape(-1, 1, L_orig)
        if L_orig % 2 == 1:
            x_flat = F.pad(x_flat, (0, 1))
        Lo = F.conv1d(x_flat, self.h_L.view(1, 1, 2), stride=2)
        Hi = F.conv1d(x_flat, self.h_H.view(1, 1, 2), stride=2)
        new_shape = list(shape[:-1]) + [-1]
        Lo = Lo.reshape(*new_shape).transpose(dim, -1)
        Hi = Hi.reshape(*new_shape).transpose(dim, -1)
        return Lo, Hi, L_orig

    def forward(self, x: Tensor):
        B, C, X, Y, Z = x.shape
        # DWT along Z (dim 4)
        Lz, Hz, _ = self._dwt_along_dim(x, -1)
        # DWT along Y (dim 3)
        LLy, LHy, _ = self._dwt_along_dim(Lz, -2)
        HLy, HHy, _ = self._dwt_along_dim(Hz, -2)
        # DWT along X (dim 2)
        LLL, LLH, _ = self._dwt_along_dim(LLy, -3)
        LHL, LHH, _ = self._dwt_along_dim(LHy, -3)
        HLL, HLH, _ = self._dwt_along_dim(HLy, -3)
        HHL, HHH, _ = self._dwt_along_dim(HHy, -3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, X, Y, Z


class _HaarIDWT3d(nn.Module):
    """3D separable Haar IDWT."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([s, s]))
        self.register_buffer('h_H', torch.tensor([s, -s]))

    def _idwt_along_dim(self, Lo: Tensor, Hi: Tensor, orig_L: int, dim: int) -> Tensor:
        Lo = Lo.transpose(dim, -1)
        Hi = Hi.transpose(dim, -1)
        shape = Lo.shape
        Lo_flat = Lo.reshape(-1, 1, shape[-1])
        Hi_flat = Hi.reshape(-1, 1, shape[-1])
        rec = (F.conv_transpose1d(Lo_flat, self.h_L.view(1, 1, 2), stride=2) +
               F.conv_transpose1d(Hi_flat, self.h_H.view(1, 1, 2), stride=2))
        new_shape = list(shape[:-1]) + [-1]
        rec = rec.reshape(*new_shape)[..., :orig_L]
        return rec.transpose(dim, -1)

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                orig_X: int, orig_Y: int, orig_Z: int) -> Tensor:
        # IDWT along X (dim 2)
        LLy = self._idwt_along_dim(LLL, LLH, orig_X, -3)
        LHy = self._idwt_along_dim(LHL, LHH, orig_X, -3)
        HLy = self._idwt_along_dim(HLL, HLH, orig_X, -3)
        HHy = self._idwt_along_dim(HHL, HHH, orig_X, -3)
        # IDWT along Y (dim 3)
        Lz = self._idwt_along_dim(LLy, LHy, orig_Y, -2)
        Hz = self._idwt_along_dim(HLy, HHy, orig_Y, -2)
        # IDWT along Z (dim 4)
        return self._idwt_along_dim(Lz, Hz, orig_Z, -1)


# ============================================================
# Deformation Network (shared with GeoFNO)
# ============================================================

class DeformationNet(nn.Module):
    """
    Learnable coordinate deformation network f_theta.
    Maps physical coordinates x to latent coordinates x_hat = x + f_theta(x).
    """

    def __init__(self, spatial_dim: int, num_layers: int = 3, hidden_dim: int = 32):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = spatial_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, spatial_dim)]
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


# ============================================================
# Spatio-Temporal Encoder
# ============================================================

class SpatioTemporalEncoder(nn.Module):
    """
    Encodes spatial coordinates with Random Fourier Features (RFF)
    and time steps with sinusoidal positional encoding.

    Spatial (RFF): gamma(x) = [sin(2pi * B * x); cos(2pi * B * x)]  in R^{2*rff_features}
    Temporal: psi(t) = [sin(omega_i * t); cos(omega_i * t)]          in R^{2*time_features}
    """

    def __init__(self, spatial_dim: int, rff_features: int = 8, time_features: int = 4,
                 max_steps: int = 1000, sigma: float = 1.0):
        super().__init__()
        self.rff_features = rff_features
        self.time_features = time_features

        # RFF projection matrix: fixed, non-trainable
        B_matrix = torch.randn(spatial_dim, rff_features) * sigma
        self.register_buffer('B_matrix', B_matrix)

        # Temporal frequencies: omega_i = max_steps^(-i/time_features)
        i = torch.arange(time_features, dtype=torch.float32)
        omega = max_steps ** (-i / max(time_features, 1))
        self.register_buffer('omega', omega)

    def encode_space(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: (B, N, spatial_dim) in [-1, 1]
        Returns:
            (B, N, 2*rff_features)
        """
        proj = 2.0 * math.pi * (coords @ self.B_matrix)  # (B, N, rff_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def encode_time(self, step: int, B: int, N: int, device: torch.device) -> Tensor:
        """
        Args:
            step: integer time step
        Returns:
            (B, N, 2*time_features)
        """
        t = torch.tensor(float(step), device=device)
        angles = self.omega * t                                      # (time_features,)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (2*time_features,)
        return emb.view(1, 1, -1).expand(B, N, -1)


# ============================================================
# WaveletConv (replaces SpectralConv)
# ============================================================

class WaveletConv(nn.Module):
    """
    Wavelet Convolution: DWT -> per-subband learnable channel mixer -> IDWT.

    For each subband, a Conv{1,2,3}d(width, width, kernel_size=1) is applied.
    This replaces the spectral weight multiplication in FNO.
    """

    def __init__(self, channels: int, modes: List[int]):
        """
        Args:
            channels: Number of feature channels (width).
            modes: Spatial dimension indicator (length of list = spatial_dim).
                   Values are unused for Haar level-1 (no truncation), kept for API compat.
        """
        super().__init__()
        self.spatial_dim = len(modes)

        _scale = 1.0 / channels

        if self.spatial_dim == 1:
            self.dwt = _HaarDWT1d()
            self.idwt = _HaarIDWT1d()
            self.weight_approx = nn.Conv1d(channels, channels, 1)
            self.weight_detail = nn.Conv1d(channels, channels, 1)
            for _w in [self.weight_approx, self.weight_detail]:
                nn.init.uniform_(_w.weight, -_scale, _scale)
                nn.init.zeros_(_w.bias)

        elif self.spatial_dim == 2:
            self.dwt = _HaarDWT2d()
            self.idwt = _HaarIDWT2d()
            self.weight_LL = nn.Conv2d(channels, channels, 1)
            self.weight_LH = nn.Conv2d(channels, channels, 1)
            self.weight_HL = nn.Conv2d(channels, channels, 1)
            self.weight_HH = nn.Conv2d(channels, channels, 1)
            for _w in [self.weight_LL, self.weight_LH, self.weight_HL, self.weight_HH]:
                nn.init.uniform_(_w.weight, -_scale, _scale)
                nn.init.zeros_(_w.bias)

        elif self.spatial_dim == 3:
            self.dwt = _HaarDWT3d()
            self.idwt = _HaarIDWT3d()
            self.weight_LLL = nn.Conv3d(channels, channels, 1)
            self.weight_LLH = nn.Conv3d(channels, channels, 1)
            self.weight_LHL = nn.Conv3d(channels, channels, 1)
            self.weight_LHH = nn.Conv3d(channels, channels, 1)
            self.weight_HLL = nn.Conv3d(channels, channels, 1)
            self.weight_HLH = nn.Conv3d(channels, channels, 1)
            self.weight_HHL = nn.Conv3d(channels, channels, 1)
            self.weight_HHH = nn.Conv3d(channels, channels, 1)
            for _w in [self.weight_LLL, self.weight_LLH, self.weight_LHL, self.weight_LHH,
                       self.weight_HLL, self.weight_HLH, self.weight_HHL, self.weight_HHH]:
                nn.init.uniform_(_w.weight, -_scale, _scale)
                nn.init.zeros_(_w.bias)

        else:
            raise ValueError(f'Only 1D, 2D, 3D supported. Got spatial_dim={self.spatial_dim}')

    def forward(self, x: Tensor) -> Tensor:
        if self.spatial_dim == 1:
            approx, detail, L = self.dwt(x)
            approx = self.weight_approx(approx)
            detail = self.weight_detail(detail)
            return self.idwt(approx, detail, L)

        elif self.spatial_dim == 2:
            LL, LH, HL, HH, H, W = self.dwt(x)
            LL = self.weight_LL(LL)
            LH = self.weight_LH(LH)
            HL = self.weight_HL(HL)
            HH = self.weight_HH(HH)
            return self.idwt(LL, LH, HL, HH, H, W)

        elif self.spatial_dim == 3:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, X, Y, Z = self.dwt(x)
            LLL = self.weight_LLL(LLL)
            LLH = self.weight_LLH(LLH)
            LHL = self.weight_LHL(LHL)
            LHH = self.weight_LHH(LHH)
            HLL = self.weight_HLL(HLL)
            HLH = self.weight_HLH(HLH)
            HHL = self.weight_HHL(HHL)
            HHH = self.weight_HHH(HHH)
            return self.idwt(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH, X, Y, Z)


# ============================================================
# WNO Block (replaces FNOBlock)
# ============================================================

class WNOBlock(nn.Module):
    """
    WNO residual block: y = Act(Norm(WaveletConv(x) + PointwiseConv(x)))

    Identical structure to FNOBlock but uses WaveletConv instead of SpectralConv.
    """

    def __init__(self, width: int, modes: List[int]):
        super().__init__()
        self.spatial_dim = len(modes)
        self.wavelet_conv = WaveletConv(width, modes)

        if self.spatial_dim == 1:
            self.pointwise_conv = nn.Conv1d(width, width, 1)
            self.norm = nn.InstanceNorm1d(width)
        elif self.spatial_dim == 2:
            self.pointwise_conv = nn.Conv2d(width, width, 1)
            self.norm = nn.InstanceNorm2d(width)
        elif self.spatial_dim == 3:
            self.pointwise_conv = nn.Conv3d(width, width, 1)
            self.norm = nn.InstanceNorm3d(width)

    def forward(self, x: Tensor) -> Tensor:
        wavelet_out = self.wavelet_conv(x)
        pointwise_out = self.pointwise_conv(x)
        return F.gelu(self.norm(wavelet_out + pointwise_out))


# ============================================================
# Geo-WNO Network
# ============================================================

class GeoWNO(nn.Module):
    """
    Geometry-Aware Wavelet Neural Operator (Geo-WNO).

    Upgrades GeoFNO with three targeted substitutions:
    1. KNN-IDW P2G (replaces scatter-mean → reduces numerical dissipation)
    2. Haar WNO (replaces FNO spectral layers → handles shocks)
    3. Spatio-temporal encoding (RFF + sinusoidal PE)

    Pipeline:
    1. Encoding: RFF spatial embedding + sinusoidal temporal embedding
    2. Lifting: fc_lift([features; space_emb; time_emb]) → hidden
    3. Deformation: physical_coords → latent_coords
    4. KNN-P2G: unstructured nodes → regular latent grid
    5. WNO blocks: wavelet convolutions on latent grid
    6. G2P: grid → node features (grid_sample interpolation, same as GeoFNO)
    7. Projection: hidden → output
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: List[int],
        latent_grid_size: List[int],
        depth: int,
        width: int,
        deformation_kwargs: Optional[Dict] = None,
        knn_k: int = 6,
        knn_chunk: int = 256,
        rff_features: int = 8,
        time_features: int = 4,
        max_steps: int = 1000,
        rff_sigma: float = 1.0,
    ):
        """
        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            modes: Wavelet block dimension indicator (len = spatial_dim). Values unused for Haar.
            latent_grid_size: Shape of the latent grid, e.g. [64, 64].
            depth: Number of WNO blocks.
            width: Hidden channel dimension.
            deformation_kwargs: kwargs for DeformationNet.
            knn_k: Number of nearest neighbors for KNN-IDW P2G.
            knn_chunk: Chunk size for batched KNN to bound memory.
            rff_features: RFF embedding half-dimension (output: 2*rff_features).
            time_features: Temporal PE half-dimension (output: 2*time_features).
            max_steps: Maximum time step (for sinusoidal frequency scaling).
        """
        super().__init__()

        assert len(modes) == len(latent_grid_size), 'modes and latent_grid_size must match in length'
        self.spatial_dim = len(modes)
        self.modes = modes
        self.latent_grid_size = latent_grid_size
        self.width = width
        self.knn_k = knn_k
        self.knn_chunk = knn_chunk
        self.eps = 1e-8

        if deformation_kwargs is None:
            deformation_kwargs = {'num_layers': 3, 'hidden_dim': 32}

        # Coordinate deformation (same as GeoFNO)
        if self.spatial_dim > 1:
            self.deformation_net = DeformationNet(spatial_dim=self.spatial_dim, **deformation_kwargs)
        else:
            self.deformation_net = None

        # Spatio-temporal encoder (detected by rollout_trainer via hasattr)
        self.time_encoder = SpatioTemporalEncoder(
            spatial_dim=self.spatial_dim,
            rff_features=rff_features,
            time_features=time_features,
            max_steps=max_steps,
            sigma=rff_sigma,
        )

        # Lifting layer: concatenate features + spatial emb + temporal emb
        lift_in = in_channels + 2 * rff_features + 2 * time_features
        self.fc_lift = nn.Linear(lift_in, width)

        # WNO blocks (replaces FNO blocks)
        self.wno_blocks = nn.ModuleList([WNOBlock(width, modes) for _ in range(depth)])

        # Projection (same as GeoFNO)
        self.fc_proj1 = nn.Linear(width, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_proj2 = nn.Linear(128, out_channels)

        # Precomputed grid centers for KNN-P2G (registered as buffer → moves with .to(device))
        self._init_grid_centers()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_grid_centers(self) -> None:
        """Precompute regular grid cell centers in [-1, 1]^d as a buffer."""
        axes = [torch.linspace(-1.0, 1.0, g) for g in self.latent_grid_size]
        grids = torch.meshgrid(*axes, indexing='ij')  # each: latent_grid_size
        centers = torch.stack([g.flatten() for g in grids], dim=-1)  # (G, spatial_dim)
        self.register_buffer('grid_centers', centers)

    # ------------------------------------------------------------------
    # KNN-IDW Point-to-Grid
    # ------------------------------------------------------------------

    def _p2g_knn(self, features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        KNN-IDW Point-to-Grid aggregation (replaces scatter-mean in GeoFNO).

        Uses chunked torch.cdist to bound per-chunk memory.

        Args:
            features: Lifted node features. Shape (B, N, W).
            latent_coords: Deformed coordinates in [-1,1]. Shape (B, N, spatial_dim).

        Returns:
            Grid features. Shape (B, W, G1, (G2), (G3)).
        """
        B, N, W = features.shape
        device = features.device
        G = self.grid_centers.shape[0]

        grid_out = torch.zeros(B, G, W, device=device, dtype=features.dtype)

        for chunk_start in range(0, G, self.knn_chunk):
            chunk_end = min(chunk_start + self.knn_chunk, G)
            centers_chunk = self.grid_centers[chunk_start:chunk_end]  # (G', spatial_dim)
            Gc = centers_chunk.shape[0]

            # Pairwise distances: (B, G', N)
            centers_exp = centers_chunk.unsqueeze(0).expand(B, -1, -1)
            dists = torch.cdist(centers_exp, latent_coords)  # (B, G', N)

            # k nearest neighbors
            topk_dists, topk_idx = dists.topk(self.knn_k, dim=-1, largest=False)  # (B, G', k)

            # IDW weights: 1/d, normalized
            weights = 1.0 / (topk_dists + self.eps)             # (B, G', k)
            weights = weights / weights.sum(dim=-1, keepdim=True)

            # Gather features via advanced indexing — no N-dim expansion
            b_idx = (torch.arange(B, device=device)
                     .view(B, 1, 1).expand(B, Gc, self.knn_k))  # (B, G', k)
            gathered = features[b_idx, topk_idx]                # (B, G', k, W)

            # Weighted aggregation
            grid_chunk = (gathered * weights.unsqueeze(-1)).sum(dim=-2)  # (B, G', W)
            grid_out[:, chunk_start:chunk_end] = grid_chunk

        # Reshape (B, G, W) -> (B, W, G1, G2, ...)
        target_shape = [B] + self.latent_grid_size + [W]
        grid_features = grid_out.view(*target_shape)
        perm = [0, self.spatial_dim + 1] + list(range(1, self.spatial_dim + 1))
        return grid_features.permute(*perm).contiguous()

    # ------------------------------------------------------------------
    # Grid-to-Point (identical to GeoFNO)
    # ------------------------------------------------------------------

    def _g2p_sample(self, grid_features: Tensor, latent_coords: Tensor) -> Tensor:
        """
        Grid-to-Point decoding using bilinear/trilinear interpolation (F.grid_sample).

        Args:
            grid_features: (B, W, G1, (G2), (G3))
            latent_coords: (B, N, spatial_dim) in [-1, 1]

        Returns:
            (B, N, W)
        """
        B, N, _ = latent_coords.shape

        if self.spatial_dim == 1:
            grid_input = grid_features.unsqueeze(2)
            zeros = torch.zeros_like(latent_coords)
            coords_input = torch.cat([latent_coords, zeros], dim=-1).view(B, N, 1, 2)
            sampled = F.grid_sample(grid_input, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 2:
            coords_input = latent_coords.view(B, N, 1, 2)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).permute(0, 2, 1)

        elif self.spatial_dim == 3:
            coords_input = latent_coords.view(B, N, 1, 1, 3)
            sampled = F.grid_sample(grid_features, coords_input, align_corners=True, padding_mode='border')
            return sampled.squeeze(-1).squeeze(-1).permute(0, 2, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_features: Tensor, physical_coords: Tensor, step: int = 0) -> Tensor:
        """
        Geo-WNO forward pass.

        Args:
            input_features: Node features at t. Shape (B, N, in_channels).
            physical_coords: Normalized coordinates in [-1, 1]. Shape (B, N, spatial_dim).
            step: Integer time step for temporal encoding. Default 0.

        Returns:
            Predicted node features. Shape (B, N, out_channels).
        """
        B, N, _ = physical_coords.shape
        device = physical_coords.device

        # 1. Spatio-temporal encoding
        space_emb = self.time_encoder.encode_space(physical_coords)   # (B, N, 2*rff)
        time_emb = self.time_encoder.encode_time(step, B, N, device)  # (B, N, 2*time)

        # 2. Lifting
        cat_input = torch.cat([input_features, space_emb, time_emb], dim=-1)
        lifted = self.fc_lift(cat_input)  # (B, N, width)

        # 3. Coordinate deformation
        if self.spatial_dim > 1 and self.deformation_net is not None:
            latent_coords = torch.clamp(self.deformation_net(physical_coords), -1.0, 1.0)
        else:
            latent_coords = physical_coords

        # 4. KNN-P2G: nodes -> regular latent grid
        grid_features = self._p2g_knn(lifted, latent_coords)  # (B, W, G1, ...)

        # 5. WNO blocks
        for block in self.wno_blocks:
            grid_features = block(grid_features)

        # 6. G2P: grid -> nodes
        recovered = self._g2p_sample(grid_features, latent_coords)  # (B, N, width)

        # 7. Projection
        output = F.gelu(self.fc_proj1(recovered))
        output = self.fc_proj2(self.dropout(output))

        return output

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive inference (same interface as GeoFNO.predict).

        Args:
            initial_state: State at t=0. Shape (B, N, in_channels).
            coords: Node coordinates in [-1, 1]. Shape (B, N, spatial_dim).
            steps: Number of future steps to generate.

        Returns:
            Predicted sequence. Shape (B, steps+1, N, out_channels).
        """
        self.eval()
        device = next(self.parameters()).device
        current_state = initial_state.to(device)
        coords = coords.to(device)

        seq: List[Tensor] = [current_state.cpu()]

        with torch.no_grad():
            for t in tqdm(range(steps), desc='Predicting', leave=False, dynamic_ncols=True):
                next_state = self.forward(current_state, coords, step=t)
                seq.append(next_state.cpu())
                current_state = next_state

        return torch.stack(seq, dim=1)
