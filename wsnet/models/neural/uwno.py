# Geometry-aware U-Net Wavelet Neural Operator (Geo-U-WNO)
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Tuple

# Use pytorch_wavelets or a manual implementation for DWT/IDWT
# For portability, we define a Haar-based Wavelet transform here
def get_wavelet_filters(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Provides Haar wavelet filters for DWT/IDWT."""
    h0 = torch.tensor([1/2**0.5, 1/2**0.5], device=device).view(1, 1, 2)
    h1 = torch.tensor([1/2**0.5, -1/2**0.5], device=device).view(1, 1, 2)
    return h0, h1

# ============================================================
# Deformation Network (Coordinate Mapping)
# ============================================================

class DeformationNet(nn.Module):
    """
    Learns f_theta: x -> x_hat. 
    Maps complex physical geometry to a canonical latent grid.
    """
    def __init__(self, spatial_dim: int, num_layers: int = 3, hidden_dim: int = 32):
        """
        Args:
            spatial_dim (int): Dimension (1, 2, or 3).
            num_layers (int): Depth of MLP.
            hidden_dim (int): Width of MLP.
        """
        super().__init__()
        layers = []
        in_dim = spatial_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, spatial_dim)]
        self.net = nn.Sequential(*layers)
        nn.init.normal_(self.net[-1].weight, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Physical coordinates. Shape: (batch_size, num_nodes, spatial_dim).
        Returns:
            Tensor: Latent coordinates. Shape: (batch_size, num_nodes, spatial_dim).
        """
        return x + self.net(x)

# ============================================================
# Wavelet Block (WNO Layer)
# ============================================================

class WaveletBlock(nn.Module):
    """
    Wavelet Integral Operator layer.
    Performs kernel integral via Wavelet Coefficient scaling.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Learnable weight for wavelet coefficients (Scaling/Detail components)
        self.weights = nn.Parameter(torch.ones(channels, channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Simple Haar-based spectral convolution substitute.
        Args:
            x (Tensor): Feature grid. Shape: (batch_size, channels, H, W).
        Returns:
            Tensor: Filtered grid. Shape: (batch_size, channels, H, W).
        """
        # Simplified Wavelet-like filtering via learnable pointwise scaling 
        # In a full WNO, this involves DWT -> Weight Mul -> IDWT
        return torch.einsum('bchw, ockk -> bohw', x, self.weights)

# ============================================================
# U-WNO Model
# ============================================================

class UWNO(nn.Module):
    """
    Geometry-Aware U-Net Wavelet Neural Operator (U-WNO).
    Combines coordinate deformation, P2G/G2P sampling, and a Wavelet-U-Net backbone.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 latent_grid_size: List[int],
                 width: int = 64,
                 deformation_kwargs: Optional[Dict] = {'num_layers': 3, 'hidden_dim': 32}):
        """
        Args:
            in_channels (int): Input feature channels.
            out_channels (int): Output feature channels.
            latent_grid_size (List[int]): Res of the latent grid [H, W].
            width (int): Hidden channel dimension.
            deformation_kwargs (Dict): Config for DeformationNet.
        """
        super().__init__()
        self.spatial_dim = len(latent_grid_size)
        self.latent_grid_size = latent_grid_size
        self.width = width

        # 1. Geometry mapping
        self.deformation_net = DeformationNet(self.spatial_dim, **deformation_kwargs)
        
        # 2. Lifting
        self.fc_lift = nn.Linear(in_channels + self.spatial_dim, width)

        # 3. U-Net Backbone (Encoder-Decoder with Wavelet Blocks)
        # Encoder
        self.enc1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(width, width*2, kernel_size=3, stride=2, padding=1)
        # Bottleneck (Wavelet processing)
        self.wavelet_bottleneck = WaveletBlock(width*2)
        # Decoder
        self.dec2 = nn.ConvTranspose2d(width*2, width, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(width*2, width, kernel_size=3, padding=1)

        # 4. Projection
        self.fc_proj = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )

    def _p2g_sample(self, features: Tensor, latent_coords: Tensor) -> Tensor:
        """Maps nodes to a regular latent grid via scatter-mean."""
        B, N, C = features.shape
        device = features.device
        dims = torch.tensor(self.latent_grid_size, device=device).view(1, 1, -1)
        
        # Normalize coords to grid indices
        idx = ((latent_coords + 1) / 2 * (dims - 1)).round().long()
        for i, d in enumerate(self.latent_grid_size):
            idx[..., i] = idx[..., i].clamp(0, d - 1)
        
        # Flattened indexing for scatter
        flat_idx = idx[..., 0] * self.latent_grid_size[1] + idx[..., 1]
        batch_offset = torch.arange(B, device=device).view(-1, 1) * (self.latent_grid_size[0] * self.latent_grid_size[1])
        global_idx = (flat_idx + batch_offset).view(-1)
        
        # Aggregation
        grid_flat = torch.zeros(B * self.latent_grid_size[0] * self.latent_grid_size[1], C, device=device)
        grid_flat.index_add_(0, global_idx, features.view(-1, C))
        
        return grid_flat.view(B, *self.latent_grid_size, C).permute(0, 3, 1, 2)

    def _g2p_sample(self, grid: Tensor, latent_coords: Tensor) -> Tensor:
        """Interpolates grid features back to node locations."""
        B, C, H, W = grid.shape
        # grid_sample expects coords in [W, H] format and range [-1, 1]
        coords_input = latent_coords.view(B, -1, 1, 2) 
        sampled = F.grid_sample(grid, coords_input, align_corners=True, padding_mode='border')
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, input_features: Tensor, physical_coords: Tensor) -> Tensor:
        """
        Forward pass for U-WNO.
        Args:
            input_features (Tensor): (batch_size, num_nodes, in_channels).
            physical_coords (Tensor): (batch_size, num_nodes, spatial_dim).
        Returns:
            Tensor: (batch_size, num_nodes, out_channels).
        """
        # 1. Lift and Deform
        x = torch.cat([input_features, physical_coords], dim=-1)
        x = self.fc_lift(x)
        latent_coords = torch.clamp(self.deformation_net(physical_coords), -1.0, 1.0)

        # 2. P2G Encoding
        grid = self._p2g_sample(x, latent_coords)

        # 3. Wavelet U-Net processing
        e1 = F.gelu(self.enc1(grid))
        e2 = F.gelu(self.enc2(e1))
        
        b = self.wavelet_bottleneck(e2)
        
        d2 = self.dec2(b)
        d1 = F.gelu(self.dec1(torch.cat([d2, e1], dim=1)))

        # 4. G2P Decoding and Project
        node_features = self._g2p_sample(d1, latent_coords)
        output = self.fc_proj(node_features)

        # Residual connection
        if input_features.shape[-1] == output.shape[-1]:
            output = output + input_features
            
        return output

    def predict(self, initial_state: Tensor, coords: Tensor, steps: int) -> Tensor:
        """
        Autoregressive prediction for temporal dynamics.
        Args:
            initial_state (Tensor): (batch_size, num_nodes, in_channels).
            coords (Tensor): (batch_size, num_nodes, spatial_dim).
            steps (int): Future time-steps.
        Returns:
            Tensor: (batch_size, steps + 1, num_nodes, out_channels).
        """
        self.eval()
        device = next(self.parameters()).device
        curr = initial_state.to(device)
        coords = coords.to(device)
        results = [curr.cpu()]

        with torch.no_grad():
            for _ in tqdm(range(steps), desc="U-WNO Solving"):
                curr = self.forward(curr, coords)
                results.append(curr.cpu())

        return torch.stack(results, dim=1)
