# CFD Data Animation and Quality Inspection Script
# Author: Shengning Wang

from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from torch import Tensor
from tqdm.auto import tqdm

from wsnet.utils.engine import sl, logger


class ProgressWriterMixin:
    """Mixin to add a progress bar callback to matplotlib animation writers."""

    def __init__(self, *args, progress_callback: Optional[callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_callback = progress_callback

    def grab_frame(self, **savefig_kwargs):
        super().grab_frame(**savefig_kwargs)
        if self._progress_callback:
            self._progress_callback()


class ProgressPillowWriter(ProgressWriterMixin, PillowWriter):
    """Custom PillowWriter with progress bar support."""
    pass


class ProgressFFMpegWriter(ProgressWriterMixin, FFMpegWriter):
    """Custom FFMpegWriter with progress bar support."""
    pass


class CFDAnimation:
    """
    High-performance CFD sequence visualizer.
    Supports single sequence animation and Ground Truth vs Prediction comparison.
    """

    def __init__(self, output_dir: Union[str, Path], spatial_dim: int = 2, fps: int = 30) -> None:
        """
        Initializes the renderer with spatial configuration and output settings.

        Args:
        - output_dir (Union[str, Path]): Directory where animations will be saved.
        - spatial_dim (int): Dimensionality of the spatial coordinates (2 or 3).
        - fps (int): Frames per second for the output video.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_dim = spatial_dim
        self.fps = fps

        # Define labels based on dimensionality as per CFDVisualizer standards
        if spatial_dim == 2:
            self.labels: List[str] = ['Vx', 'Vy', 'P', 'T']
        elif spatial_dim == 3:
            self.labels: List[str] = ['Vx', 'Vy', 'Vz', 'P', 'T']
        else:
            self.labels: List[str] = [f'Field_{i}' for i in range(spatial_dim + 2)]

    def _get_writer(self, file_format: str, total_frames: int, case_name: str
                    ) -> Union[ProgressPillowWriter, ProgressFFMpegWriter]:
        """Configures the animation writer with a tqdm progress bar."""
        pbar = tqdm(total=total_frames, desc=f"Rendering {case_name}", leave=False)

        if file_format.lower() == 'mp4':
            return ProgressFFMpegWriter(fps=self.fps, progress_callback=lambda: pbar.update(1),
                codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        elif file_format.lower() == 'gif':
            return ProgressPillowWriter(fps=self.fps, progress_callback=lambda: pbar.update(1))
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'mp4' or 'gif'.")

    def animate_sequence(self, sequence: Tensor, coords: Tensor, case_name: str, file_format: str = 'mp4') -> None:
        """
        Generates animation of a single CFD sequence.

        Args:
        - sequence (Tensor): Temporal data. Shape: (seq_len, num_nodes, num_channels)
        - coords (Tensor): Spatial coordinates. Shape: (num_nodes, spatial_dim)
        - case_name (str): Identifier for the output filename.
        - file_format (str): Extension for the output file ('mp4' or 'gif').
        """
        seq_len, _, num_channels = sequence.shape
        seq_np = sequence.detach().cpu().numpy()
        x, y = coords[:, 0].numpy(), coords[:, 1].numpy()

        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 4 * num_channels), constrained_layout=True, dpi=150)
        if num_channels == 1: axes = [axes]

        scatters = []
        for c in range(num_channels):
            ax = axes[c]
            sc = ax.scatter(x, y, c=seq_np[0, :, c], cmap="viridis", s=5, edgecolors='none')
            cbar =fig.colorbar(sc, ax=ax)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            ax.set_aspect('equal')
            ax.set_title(f'Case: {case_name} | Field: {self.labels[c]} | Frame 1/{seq_len}')
            scatters.append(sc)

        def update(frame: int) -> List[plt.Artist]:
            artists = []
            for c in range(num_channels):
                scatters[c].set_array(seq_np[frame, :, c])
                axes[c].set_title(f'Case: {case_name} | Frame {frame + 1}/{seq_len} | Field: {self.labels[c]}')
                artists.append(scatters[c])
            return artists

        anim = FuncAnimation(fig, update, frames=seq_len, blit=False)
        save_path = self.output_dir / f'{case_name}_seq.{file_format}'

        anim.save(save_path, writer=self._get_writer(file_format, seq_len, case_name))

        plt.close(fig)
        logger.info(f'sequence animation saved to {save_path}')

    def animate_comparison(self, gt: Tensor, pred: Tensor, coords: Tensor, case_name: str, file_format: str = 'mp4'
                        ) -> None:
            """
            Generates a side-by-side comparison animation of Ground Truth, Prediction, and Absolute Error.

            Args:
            - gt (Tensor): Ground truth temporal data. Shape: (seq_len, num_nodes, num_channels)
            - pred (Tensor): Predicted temporal data. Shape: (seq_len, num_nodes, num_channels)
            - coords (Tensor): Spatial coordinates. Shape: (num_nodes, spatial_dim)
            - case_name (str): Identifier for the output filename.
            - file_format (str): Extension for the output file ('mp4' or 'gif').
            """
            # Calculate Absolute Error: (seq_len, num_nodes, num_channels)
            error = torch.abs(gt - pred)

            seq_len, _, num_channels = gt.shape

            gt_np = gt.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            err_np = error.detach().cpu().numpy()
            coords_np = coords.cpu().numpy()

            x, y = coords_np[:, 0], coords_np[:, 1]

            fig, axes = plt.subplots(num_channels, 3, figsize=(15, 4 * num_channels), constrained_layout=True, dpi=150)
            if num_channels == 1: axes = np.expand_dims(axes, axis=0)

            scatters = [[] for _ in range(num_channels)]

            column_titles = ['Ground Truth', 'Prediction (FNO)', 'Abs Error']

            for c in range(num_channels):
                data_triplet = [gt_np[0, :, c], pred_np[0, :, c], err_np[0, :, c]]

                for col_idx in range(3):
                    ax = axes[c, col_idx]
                    cmap = 'viridis' if col_idx < 2 else 'RdBu_r'
                    sc = ax.scatter(x, y, c=data_triplet[col_idx], cmap=cmap, s=5, edgecolors='none')
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                    ax.set_aspect('equal')
                    # if c == 0:
                    #     ax.set_title(f"{column_titles[col_idx]}\n\nField: {self.labels[c]}", fontweight='bold')
                    # else:
                    #     ax.set_title(f"Field: {self.labels[c]}")
                    ax.set_title(f"{column_titles[col_idx]}\n\nField: {self.labels[c]}", fontweight='bold')

                    scatters[c].append(sc)

            def update(frame: int) -> List[plt.Artist]:
                updated_artists = []
                for c in range(num_channels):
                    frame_data = [gt_np[frame, :, c], pred_np[frame, :, c], err_np[frame, :, c]]

                    for col_idx in range(3):
                        scatters[c][col_idx].set_array(frame_data[col_idx])
                        updated_artists.append(scatters[c][col_idx])

                # Global title update to show progress
                fig.suptitle(f"Case: {case_name} | Frame {frame + 1}/{seq_len}", fontsize=16)
                return updated_artists

            anim = FuncAnimation(fig, update, frames=seq_len, blit=False)
            save_path = self.output_dir / f'{case_name}_comp.{file_format}'

            anim.save(save_path, writer=self._get_writer(file_format, seq_len, case_name))

            plt.close(fig)
            logger.info(f'comparison animation saved to {save_path}')


if __name__ == "__main__":
    # Example Usage for Validation
    import torch

    # Mock Data
    spatial_dim = 2
    T, N, C = 20, 1000, spatial_dim + 2
    mock_coords = torch.rand(N, spatial_dim)
    mock_seq = torch.sin(torch.linspace(0, 2*np.pi, T)).view(T, 1, 1) * torch.rand(T, N, C)
    mock_pred = mock_seq * 0.9 + 0.05

    renderer = CFDAnimation(output_dir='.', spatial_dim=spatial_dim, fps=10)

    # 1. Sequence
    renderer.animate_sequence(mock_seq, mock_coords, case_name='test_101')

    # 2. Comparison
    renderer.animate_comparison(mock_seq, mock_pred, mock_coords, case_name='test_101')
