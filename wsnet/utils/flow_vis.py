# Flow Sequence Visualization Module
# Author: Shengning Wang

from pathlib import Path
from typing import List, Union, Tuple

import torch
import numpy as np
import pyvista as pv
from torch import Tensor
from tqdm.auto import tqdm

from wsnet.utils import sl, logger

class FlowVis:
    """
    High-performance CFD visualization engine using PyVista (VTK backend).

    This class provides GPU-accelerated rendering for nodal field data, supporting
    both single-sequence animations and side-by-side ground truth comparisons.
    """

    def __init__(self, output_dir: Union[str, Path], spatial_dim: int = 2, fps: int = 30, theme: str = 'document'
                ) -> None:
        """
        Initialize the FlowVis renderer.

        Args:
            output_dir (Union[str, Path]): Directory to save rendered animations.
            spatial_dim (int): Spatial dimensionality of data (2 or 3).
            fps (int): Frames per second for output video.
            theme (str): PyVista plotting theme ('document', 'paraview', 'dark').
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_dim = spatial_dim
        self.fps = fps

        # set visualization theme (affects background color, fonts, etc.)
        pv.set_plot_theme(theme)

        # define standard field labels
        if spatial_dim == 2:
            self.labels: List[str] = ['Vx', 'Vy', 'P', 'T']
        elif spatial_dim == 3:
            self.labels: List[str] = ['Vx', 'Vy', 'Vz', 'P', 'T']
        else:
            self.labels = [f'Field_{i}' for i in range(spatial_dim + 2)]

    def _prepare_mesh(self, coords: Tensor) -> pv.PolyData:
        """
        Converts coordinate tensor into a PyVista PolyData object.

        Args:
            coords (Tensor): Spatial coordinates. Shape: (num_nodes, spatial_dim)

        Returns:
            pv.PolyData: The VTK point cloud object ready for rendering.
        """
        coords_np = coords.detach().cpu().numpy()
        
        # VTK requires 3D points. If 2D, pad Z-axis with zeros.
        if self.spatial_dim == 2:
            num_nodes = coords_np.shape[0]
            zeros = np.zeros((num_nodes, 1), dtype=coords_np.dtype)
            points = np.hstack([coords_np, zeros])
        else:
            points = coords_np

        return pv.PolyData(points)

    def _get_clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Computes the global min and max for consistent color scaling.
        
        Args:
            data (np.ndarray): The full temporal sequence for a specific channel.
                               Shape: (seq_len, num_nodes)
        
        Returns:
            Tuple[float, float]: (min_val, max_val)
        """
        return float(np.min(data)), float(np.max(data))

    def animate_sequence(
        self, 
        sequence: Tensor, 
        coords: Tensor, 
        case_name: str,
        point_size: int = 5
    ) -> None:
        """
        Renders a single CFD sequence animation using PyVista.

        Args:
            sequence (Tensor): Temporal field data. 
                               Shape: (seq_len, num_nodes, num_channels)
            coords (Tensor): Spatial coordinates of nodes. 
                             Shape: (num_nodes, spatial_dim)
            case_name (str): Identifier for the output filename.
            point_size (int): visual size of the render points.
        """
        seq_len, num_nodes, num_channels = sequence.shape
        seq_np = sequence.detach().cpu().numpy()
        
        # Initialize mesh
        mesh = self._prepare_mesh(coords)
        
        # Setup Plotter: One row per channel
        plotter = pv.Plotter(
            shape=(num_channels, 1), 
            off_screen=True, 
            window_size=(800, 300 * num_channels)
        )

        # Initialize actors and color ranges
        actors = []
        clims = []
        
        for i in range(num_channels):
            plotter.subplot(i, 0)
            data_channel = seq_np[:, :, i]
            clim = self._get_clim(data_channel)
            clims.append(clim)
            
            # Assign initial frame data
            mesh_copy = mesh.copy()
            mesh_copy.point_data['values'] = seq_np[0, :, i]
            
            actor = plotter.add_mesh(
                mesh_copy,
                scalars='values',
                cmap='viridis',
                clim=clim,
                point_size=point_size,
                render_points_as_spheres=True,
                scalar_bar_args={'title': self.labels[i]}
            )
            actors.append(mesh_copy)
            
            if self.spatial_dim == 2:
                plotter.view_xy()
            plotter.add_text(f"Field: {self.labels[i]}", font_size=10)

        # Open Movie File
        filename = self.output_dir / f"{case_name}_seq.mp4"
        plotter.open_movie(str(filename), framerate=self.fps)
        
        logger.info(f"Rendering sequence: {case_name} -> {filename}")
        
        # Animation Loop
        for t in tqdm(range(seq_len), desc=f"Rendering {case_name}", leave=False):
            for i in range(num_channels):
                # Efficient in-place update of point data
                actors[i].point_data['values'] = seq_np[t, :, i]
            
            plotter.write_frame()

        plotter.close()

    def animate_comparison(
        self, 
        gt: Tensor, 
        pred: Tensor, 
        coords: Tensor, 
        case_name: str,
        point_size: int = 5
    ) -> None:
        """
        Renders a side-by-side comparison (GT, Pred, Error) animation.

        Args:
            gt (Tensor): Ground truth data. 
                         Shape: (seq_len, num_nodes, num_channels)
            pred (Tensor): Predicted data. 
                           Shape: (seq_len, num_nodes, num_channels)
            coords (Tensor): Spatial coordinates. 
                             Shape: (num_nodes, spatial_dim)
            case_name (str): Identifier for the output filename.
            point_size (int): visual size of the render points.
        """
        seq_len, num_nodes, num_channels = gt.shape
        error = torch.abs(gt - pred)
        
        # Convert to numpy once
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        err_np = error.detach().cpu().numpy()
        
        mesh = self._prepare_mesh(coords)
        
        # Setup Plotter: Grid of (num_channels x 3)
        # Columns: Ground Truth | Prediction | Absolute Error
        plotter = pv.Plotter(
            shape=(num_channels, 3), 
            off_screen=True, 
            window_size=(1200, 300 * num_channels)
        )
        
        column_titles = ["Ground Truth", "Prediction", "Abs Error"]
        meshes_grid = [] # Stores (channel, column) mesh references
        
        for c in range(num_channels):
            channel_meshes = []
            
            # Determine color limits
            # GT and Pred share the same clim for visual consistency
            # Error gets its own clim (usually 0 to max_error)
            val_clim = (
                min(self._get_clim(gt_np[:, :, c])[0], self._get_clim(pred_np[:, :, c])[0]),
                max(self._get_clim(gt_np[:, :, c])[1], self._get_clim(pred_np[:, :, c])[1])
            )
            err_clim = self._get_clim(err_np[:, :, c])
            
            data_sources = [gt_np, pred_np, err_np]
            
            for col in range(3):
                plotter.subplot(c, col)
                
                mesh_copy = mesh.copy()
                mesh_copy.point_data['values'] = data_sources[col][0, :, c]
                
                is_error_col = (col == 2)
                cmap = 'RdBu_r' if is_error_col else 'viridis'
                clim = err_clim if is_error_col else val_clim
                
                plotter.add_mesh(
                    mesh_copy,
                    scalars='values',
                    cmap=cmap,
                    clim=clim,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    scalar_bar_args={'title': f"{self.labels[c]} ({column_titles[col]})"}
                )
                
                if self.spatial_dim == 2:
                    plotter.view_xy()
                
                if c == 0:
                    plotter.add_text(column_titles[col], font_size=12, position='upper_left')
                
                channel_meshes.append(mesh_copy)
            
            meshes_grid.append(channel_meshes)

        # Open Movie File
        filename = self.output_dir / f"{case_name}_comp.mp4"
        plotter.open_movie(str(filename), framerate=self.fps)
        
        logger.info(f"Rendering comparison: {case_name} -> {filename}")
        
        # Animation Loop
        for t in tqdm(range(seq_len), desc=f"Rendering {case_name}", leave=False):
            for c in range(num_channels):
                # Update GT
                meshes_grid[c][0].point_data['values'] = gt_np[t, :, c]
                # Update Pred
                meshes_grid[c][1].point_data['values'] = pred_np[t, :, c]
                # Update Error
                meshes_grid[c][2].point_data['values'] = err_np[t, :, c]
            
            plotter.write_frame()
            
        plotter.close()

if __name__ == "__main__":
    # Validation / Example Usage

    # 1. Mock Data Generation
    spatial_dim = 2
    T, N, C = 40, 2000, spatial_dim + 2 # (seq_len, num_nodes, num_channels)

    logger.info("Generating mock data...")
    mock_coords = torch.rand(N, spatial_dim)
    # Create a moving wave pattern
    time_steps = torch.linspace(0, 4*np.pi, T).view(T, 1, 1)
    mock_seq = torch.sin(time_steps + mock_coords[:, 0] * 5) * torch.cos(mock_coords[:, 1] * 5)
    # Expand to C channels with slight variations
    mock_seq = mock_seq.repeat(1, 1, C) * torch.arange(1, C+1).view(1, 1, C)

    mock_pred = mock_seq * 0.9 + torch.randn_like(mock_seq) * 0.1

    # 2. Instantiate FlowVis
    vis = FlowVis(output_dir='vis_outputs', spatial_dim=spatial_dim, fps=20)

    # 3. Render Sequence
    vis.animate_sequence(mock_seq, mock_coords, case_name='demo_flow')

    # 4. Render Comparison
    vis.animate_comparison(mock_seq, mock_pred, mock_coords, case_name='demo_flow')
