# Flow Sequence Visualization — PyVista / GPU-accelerated (replaces CFDRender)
# Author: Shengning Wang
#
# GPU setup (NVIDIA headless server):
#   export PYVISTA_FORCE_EGL=1    # select EGL backend before starting Python
#   export PYVISTA_OFF_SCREEN=true
# macOS / workstation with display: no extra env vars needed.

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
import pyvista as pv

from wsnet.utils.hue_logger import hue, logger


# ---------------------------------------------------------------------------
# Per-channel visualization metadata
# ---------------------------------------------------------------------------

def _channel_role(ch_idx: int, spatial_dim: int) -> str:
    """Return role tag: 'velocity', 'pressure', or 'temperature'."""
    if ch_idx < spatial_dim:
        return 'velocity'
    if ch_idx == spatial_dim:
        return 'pressure'
    return 'temperature'


_CMAP: dict = {
    'velocity':    'RdBu_r',   # diverging, zero = white — correct for ±Vx/Vy
    'pressure':    'plasma',   # perceptually uniform sequential (log scale)
    'temperature': 'plasma',   # same
    'error':       'Reds',     # sequential, always non-negative
}


class FlowVis:
    """
    GPU-accelerated CFD visualization engine using PyVista (VTK/OpenGL backend).

    Features:
    - Off-screen GPU rendering via VTK + EGL (NVIDIA headless servers)
    - log₁₀(P) colormap for high-pressure-ratio (10000:1) flows
    - Percentile-clipped (2 %–98 %) color limits: reveals small variations
      in near-constant fields such as temperature
    - Diverging colormap (RdBu_r) for velocity channels (zero = white)
    - Error column uses Reds (strictly non-negative)
    - No imageio-ffmpeg dependency: encodes via system ffmpeg subprocess

    Notes:
        - For MP4 output, system ffmpeg must be on PATH (or set FFMPEG_EXE env var).
        - For GIF output, PIL/Pillow is used (already a PyVista dependency).
    """

    FFMPEG_EXE: str = os.environ.get('FFMPEG_EXE', 'ffmpeg')

    def __init__(
        self,
        output_dir: Union[str, Path],
        spatial_dim: int = 2,
        fps: int = 30,
        theme: str = 'document',
        window_width: int = 2400,
        subplot_height: int = 250,
    ) -> None:
        """
        Args:
            output_dir:     Directory to save rendered animations.
            spatial_dim:    Spatial dimensionality (2 or 3).
            fps:            Frames per second for output video.
            theme:          PyVista theme ('document', 'paraview', 'dark').
            window_width:   Total window width in pixels (for comparison layout).
            subplot_height: Pixel height per channel row.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_dim = spatial_dim
        self.fps = fps
        self.p_idx = spatial_dim        # pressure channel index
        self.window_width = window_width
        self.subplot_height = subplot_height

        pv.set_plot_theme(theme)

        if spatial_dim == 2:
            self.ch_names: List[str] = ['Vx', 'Vy', 'P', 'T']
        elif spatial_dim == 3:
            self.ch_names: List[str] = ['Vx', 'Vy', 'Vz', 'P', 'T']
        else:
            self.ch_names = [f'Field_{i}' for i in range(spatial_dim + 2)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_points(self, coords: Tensor) -> np.ndarray:
        """
        Convert coordinate tensor to (N, 3) float32 array for VTK.
        Pads Z = 0 for 2-D data.

        Args:
            coords: (N, spatial_dim)

        Returns:
            (N, 3) float32 numpy array.
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        if self.spatial_dim == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        return pts

    def _preprocess(self, data: np.ndarray, ch_idx: int) -> np.ndarray:
        """
        Apply channel-specific visualization transform.

        Pressure: log₁₀(clip(P, 1.0)) — compresses 10000:1 dynamic range.
        Others:   identity.

        Args:
            data:   Array of any shape for a single channel.
            ch_idx: Channel index.

        Returns:
            Transformed array (same shape as input).
        """
        if ch_idx == self.p_idx:
            return np.log10(np.clip(data, 1.0, None))
        return data

    def _get_clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Percentile-clipped color limits from a full temporal stack.

        Uses 2nd–98th percentile so near-constant fields (e.g. temperature
        with ±0.3 K variation) still produce a meaningful color gradient.

        Args:
            data: Array of any shape (will be raveled).

        Returns:
            (vmin, vmax) float tuple.
        """
        flat = data.ravel()
        lo = float(np.percentile(flat, 2))
        hi = float(np.percentile(flat, 98))
        if abs(hi - lo) < 1e-9:          # truly constant field: add tiny epsilon
            center = (lo + hi) * 0.5
            lo, hi = center - 1e-6, center + 1e-6
        return lo, hi

    def _channel_cmap(self, ch_idx: int) -> str:
        """Colormap string for a given channel (not error column)."""
        return _CMAP[_channel_role(ch_idx, self.spatial_dim)]

    def _scalar_bar_title(self, ch_idx: int, col: int) -> str:
        """
        Scalar bar label for (channel_index, column).
        col: 0 = GT / single, 1 = Pred, 2 = Abs Error.
        """
        name = self.ch_names[ch_idx] if ch_idx < len(self.ch_names) else f'Ch{ch_idx}'
        if col == 2:
            # Error is always in physical units (Pa for pressure, not log₁₀)
            suffix = ' (Pa)' if ch_idx == self.p_idx else ''
            return f'|\u0394{name}|{suffix}'
        if ch_idx == self.p_idx:
            return 'P (log\u2081\u2080 Pa)'   # P (log₁₀ Pa)
        return name

    def _setup_camera(self, plotter: pv.Plotter) -> None:
        """Fix camera view for the active subplot."""
        if self.spatial_dim == 2:
            plotter.view_xy()
        else:
            plotter.view_isometric()
        plotter.reset_camera()

    def _encode_mp4(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        desc: str,
    ) -> None:
        """
        Capture frames via plotter.screenshot() and encode to MP4 using
        system ffmpeg (stdin raw-video pipe). No imageio-ffmpeg dependency.

        Args:
            plotter:   Configured PyVista Plotter (off_screen=True).
            update_fn: Callable(t: int) — updates all mesh point_data for frame t.
            seq_len:   Total number of frames.
            out_path:  Output .mp4 path.
            desc:      tqdm progress bar description.
        """
        # First frame is already in the mesh (set during plotter setup).
        # Capture it to detect the actual pixel dimensions.
        first_frame = plotter.screenshot(return_img=True)  # (H, W, 3) uint8
        H, W = first_frame.shape[:2]

        # Ensure even dimensions (libx264 requirement)
        W_enc = W + (W % 2)
        H_enc = H + (H % 2)

        ffmpeg_cmd = [
            self.FFMPEG_EXE, '-y',
            '-framerate', str(self.fps),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-i', 'pipe:0',
            '-vf', f'pad={W_enc}:{H_enc}:0:0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
            str(out_path),
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        proc.stdin.write(first_frame.tobytes())
        for t in tqdm(range(1, seq_len), desc=desc, leave=False):
            update_fn(t)
            plotter.render()
            proc.stdin.write(plotter.screenshot(return_img=True).tobytes())

        proc.stdin.close()
        proc.wait()
        plotter.close()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}. "
                f"Ensure ffmpeg is on PATH or set FFMPEG_EXE env var."
            )

    def _encode_gif(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        desc: str,
    ) -> None:
        """
        Capture frames and write to GIF via imageio (uses PIL, no ffmpeg needed).

        Args:
            plotter:   Configured PyVista Plotter (off_screen=True).
            update_fn: Callable(t: int) — updates all mesh point_data for frame t.
            seq_len:   Total number of frames.
            out_path:  Output .gif path.
            desc:      tqdm progress bar description.
        """
        import imageio
        frames: List[np.ndarray] = []
        for t in tqdm(range(seq_len), desc=desc, leave=False):
            update_fn(t)
            plotter.render()
            frames.append(plotter.screenshot(return_img=True))
        plotter.close()
        imageio.mimwrite(str(out_path), frames, fps=self.fps, loop=0)

    def _animate(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        file_format: str,
        desc: str,
    ) -> None:
        """Dispatch to the right encoder based on file_format."""
        if file_format == 'mp4':
            self._encode_mp4(plotter, update_fn, seq_len, out_path, desc)
        elif file_format == 'gif':
            self._encode_gif(plotter, update_fn, seq_len, out_path, desc)
        else:
            raise ValueError(f"Unsupported format '{file_format}'. Use 'mp4' or 'gif'.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def animate_sequence(
        self,
        sequence: Tensor,
        coords: Tensor,
        case_name: str,
        file_format: str = 'mp4',
        point_size: int = 5,
    ) -> None:
        """
        Render a single-sequence animation.

        Args:
            sequence:    (seq_len, N, C) field data in physical units.
            coords:      (N, spatial_dim) node coordinates.
            case_name:   Output filename prefix.
            file_format: 'mp4' or 'gif'.
            point_size:  Rendered point size in pixels.
        """
        seq_len, _, num_channels = sequence.shape
        seq_np = sequence.detach().cpu().numpy()
        points = self._prepare_points(coords)

        # Apply per-channel transforms; compute color limits from full stack.
        seq_vis = np.stack(
            [self._preprocess(seq_np[:, :, c], c) for c in range(num_channels)],
            axis=-1,
        )  # (seq_len, N, C)
        clims = [self._get_clim(seq_vis[:, :, c]) for c in range(num_channels)]
        for c in range(num_channels):
            if _channel_role(c, self.spatial_dim) == 'velocity':
                lo, hi = clims[c]
                vmax = max(abs(lo), abs(hi))
                clims[c] = (-vmax, vmax)

        plotter = pv.Plotter(
            shape=(num_channels, 1),
            off_screen=True,
            window_size=(self.window_width // 3, self.subplot_height * num_channels),
        )

        meshes: List[pv.PolyData] = []
        for c in range(num_channels):
            plotter.subplot(c, 0)
            mesh = pv.PolyData(points)
            mesh.point_data['scalar'] = seq_vis[0, :, c].astype(np.float32)
            plotter.add_mesh(
                mesh, scalars='scalar',
                cmap=self._channel_cmap(c),
                clim=clims[c],
                point_size=point_size,
                render_points_as_spheres=False,
                scalar_bar_args={'title': self._scalar_bar_title(c, 0)},
            )
            plotter.add_text(
                f'Field: {self.ch_names[c]}', font_size=10, position='upper_edge',
            )
            self._setup_camera(plotter)
            meshes.append(mesh)

        def _update(t: int) -> None:
            for c, mesh in enumerate(meshes):
                mesh.point_data['scalar'] = seq_vis[t, :, c].astype(np.float32)

        out_path = self.output_dir / f'{case_name}_seq.{file_format}'
        self._animate(plotter, _update, seq_len, out_path, file_format,
                      desc=f'Rendering {case_name}')
        logger.info(f'sequence animation saved to {hue.g}{out_path}{hue.q}')

    def animate_comparison(
        self,
        gt: Tensor,
        pred: Tensor,
        coords: Tensor,
        case_name: str,
        file_format: str = 'mp4',
        point_size: int = 5,
    ) -> None:
        """
        Render Ground Truth / Prediction / Abs Error comparison animation.

        Layout: num_channels rows × 3 columns.
        GT and Pred share the same color limits; Error has its own.
        Pressure is shown in log₁₀(Pa) in GT/Pred columns and in raw Pa in
        the Error column (absolute residual preserves physical interpretation).

        Args:
            gt:          (seq_len, N, C) ground truth in physical units.
            pred:        (seq_len, N, C) prediction in physical units.
            coords:      (N, spatial_dim) node coordinates.
            case_name:   Output filename prefix.
            file_format: 'mp4' or 'gif'.
            point_size:  Rendered point size in pixels.
        """
        seq_len, _, num_channels = gt.shape
        err = torch.abs(gt - pred)

        gt_np   = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        err_np  = err.detach().cpu().numpy()
        points  = self._prepare_points(coords)

        # Log₁₀(P) applied to GT/Pred only; error stays in physical units (Pa).
        gt_vis   = np.stack(
            [self._preprocess(gt_np[:, :, c],   c) for c in range(num_channels)], axis=-1,
        )
        pred_vis = np.stack(
            [self._preprocess(pred_np[:, :, c], c) for c in range(num_channels)], axis=-1,
        )

        # GT + Pred share one range; error gets its own.
        val_clims = [
            self._get_clim(
                np.concatenate([gt_vis[:, :, c].ravel(), pred_vis[:, :, c].ravel()])
            )
            for c in range(num_channels)
        ]
        for c in range(num_channels):
            if _channel_role(c, self.spatial_dim) == 'velocity':
                lo, hi = val_clims[c]
                vmax = max(abs(lo), abs(hi))
                val_clims[c] = (-vmax, vmax)
        err_clims = [self._get_clim(err_np[:, :, c]) for c in range(num_channels)]

        plotter = pv.Plotter(
            shape=(num_channels, 3),
            off_screen=True,
            window_size=(self.window_width, self.subplot_height * num_channels),
        )

        col_titles = ['Ground Truth', 'Prediction', 'Abs Error']
        meshes: List[List[pv.PolyData]] = []  # meshes[channel][col]

        for c in range(num_channels):
            srcs  = [gt_vis,       pred_vis,       err_np]
            clims = [val_clims[c], val_clims[c],   err_clims[c]]
            cmaps = [self._channel_cmap(c), self._channel_cmap(c), _CMAP['error']]

            row: List[pv.PolyData] = []
            for col in range(3):
                plotter.subplot(c, col)
                mesh = pv.PolyData(points)
                mesh.point_data['scalar'] = srcs[col][0, :, c].astype(np.float32)
                plotter.add_mesh(
                    mesh, scalars='scalar',
                    cmap=cmaps[col],
                    clim=clims[col],
                    point_size=point_size,
                    render_points_as_spheres=False,
                    scalar_bar_args={'title': self._scalar_bar_title(c, col)},
                )
                if c == 0:
                    plotter.add_text(col_titles[col], font_size=11, position='upper_edge')
                self._setup_camera(plotter)
                row.append(mesh)

            meshes.append(row)

        def _update(t: int) -> None:
            for c in range(num_channels):
                meshes[c][0].point_data['scalar'] = gt_vis[t, :, c].astype(np.float32)
                meshes[c][1].point_data['scalar'] = pred_vis[t, :, c].astype(np.float32)
                meshes[c][2].point_data['scalar'] = err_np[t, :, c].astype(np.float32)

        out_path = self.output_dir / f'{case_name}_comp.{file_format}'
        self._animate(plotter, _update, seq_len, out_path, file_format,
                      desc=f'Rendering {case_name}')
        logger.info(f'comparison animation saved to {hue.g}{out_path}{hue.q}')


if __name__ == '__main__':
    # Smoke test with synthetic data
    spatial_dim = 2
    T, N, C = 20, 2000, spatial_dim + 2

    logger.info('Generating mock data...')
    mock_coords = torch.rand(N, spatial_dim)

    time_steps = torch.linspace(0, 4 * np.pi, T).view(T, 1, 1)
    wave = torch.sin(time_steps + mock_coords[:, 0].view(1, N, 1) * 5)
    mock_seq = wave.expand(T, N, C).clone()
    mock_seq[:, :, spatial_dim] = torch.rand(T, N) * 4e6 + 5e5   # P: 500 kPa – 4.5 MPa
    mock_seq[:, :, spatial_dim + 1] = 300.0 + torch.randn(T, N) * 0.5  # T: near-constant

    mock_pred = mock_seq * 0.95 + torch.randn_like(mock_seq) * 0.05

    vis = FlowVis(output_dir='vis_outputs', spatial_dim=spatial_dim, fps=10)
    vis.animate_sequence(mock_seq, mock_coords, case_name='demo_seq')
    vis.animate_comparison(mock_seq, mock_pred, mock_coords, case_name='demo_comp')
