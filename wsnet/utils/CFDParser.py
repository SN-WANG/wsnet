# ANSYS FLuent Simulation Sequence Dataset Module
# Author: Shengning Wang

import re
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm

import wsnet.utils.Engine as E


def find_case_dirs(data_dir: Union[str, Path], prefix: str = 'case') -> List[str]:
    """
    Identifies and sorts all case folders within the data directory.

    Args:
    - data_dir (Union[str, Path]): Root dierctory containing case-specific folders
    - prefix (str): Subdirectory naming prefix to filter

    Returns:
    - List[str]: A sorted list of directory names
    """
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f'Directory not found: {path}')

    cases = [d.name for d in path.iterdir() if d.is_dir() and d.name.startswith(prefix)]

    return sorted(cases)


class CFDataset(Dataset):
    """
    A generic dataset for 2D/3D CFD simulation data exported from ANSYS Fluent.
    Designed for autoregressive (rollout) training

    Data Structure:
    - Stores full sequence for each case
    - Supports caching parsed tensors to .pt files for efficiency

    Attributes:
    - sequences (List[Tensor]): List of feature tensors. Each tensor has shape (seq_len, num_nodes, in_channels)
    - coords (List[Tensor]): List of coordinate tensors. Each tensor has shape (num_nodes, spatial_dim)
    - spatial_dim (int): Spatial dimension of the data (2 or 3)

    Features (in_channels / out_channels):
    - 2D: [V_x, V_y, P, T] (4 channels)
    - 3D: [V_x, V_y, V_z, P, T] (5 channels)

    Coordinates (spatial_dim):
    - 2D: [x, y]
    - 3D: [x, y, z]
    """

    def __init__(self, data_dir: Union[str, Path], case_list: List[str], spatial_dim: int = 2,
                 limits: Optional[Tuple[int, int]] = None, force_reprocess: bool = False) -> None:
        """
        Initialize the dataset with caching logic

        Args:
        - data_dir (Union[str, Path]): Root directory containing raw case folders and processed cache
        - case_list (List[str]): List of case names to load
        - spatial_dim (int): 2 for 2D flows, 3 for 3D flows
        - limits (Optional[Tuple[int, int]]): Maximum number of seq_len and nodes to keep (subsampling)
        - force_reprocess (bool): If True, re-parses raw .txt files
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed_cache'
        self.case_list = case_list
        self.spatial_dim = spatial_dim
        self.limits = limits

        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Containers for loaded tensors
        self.sequences: List[Tensor] = []  # List of (seq_len, num_nodes, in_channels)
        self.coords: List[Tensor] = []  # List of (num_nodes, spatial_dim)

        E.logger.info(f'Initializing dataset with {len(case_list)} cases...')

        for case_name in tqdm(self.case_list, desc='Loading Cases', leave=False):
            # Define cache path
            cache_path = self.processed_dir / f'{case_name}.pt'

            if cache_path.exists() and not force_reprocess:
                # Load from cache
                case_data = torch.load(cache_path)
            else:
                # Parse from raw .txt and save to cache
                case_data = self._parse_case_sequence(case_name, cache_path)

            if case_data:
                seq_tensor = case_data['seq']  # (seq_len, num_nodes, num_channels)
                coords_tensor = case_data['coords']  # (num_nodes, spatial_dim)

                # Apply node subsampling if requested
                if self.limits:
                    max_seq_len, max_nodes = self.limits
                    curr_seq_len, curr_nodes, _ = seq_tensor.shape

                    # 1. Temporal equidistant downsampling (seq_len)
                    if curr_seq_len > max_seq_len:
                        E.logger.info(f'Subsampling sequence length to {max_seq_len}...')
                        t_indices = torch.linspace(0, curr_seq_len - 1, max_seq_len).long()
                        seq_tensor = torch.index_select(seq_tensor, 0, t_indices)

                    # 2. Spatial equidistant downsampling (num_nodes)
                    if curr_nodes > max_nodes:
                        E.logger.info(f'Subsampling nodes number to {max_nodes}...')
                        s_indices = torch.linspace(0, curr_nodes - 1, max_nodes).long()
                        seq_tensor = torch.index_select(seq_tensor, 1, s_indices)
                        coords_tensor = torch.index_select(coords_tensor, 0, s_indices)

                self.sequences.append(seq_tensor)
                self.coords.append(coords_tensor)

        E.logger.info(f'Dataset initialized. Cases: {len(self.sequences)}, '
                      f'Frames: {self.sequences[0].shape[0]}, '
                      f'Nodes: {self.sequences[0].shape[1]}, '
                      f'Channels: {self.sequences[0].shape[2]}')

    def _parse_case_sequence(self, case_name: str, save_path: Path) -> Optional[Dict[str, Tensor]]:
        """
        Reads raw .txt files for a single case, creates sequence tensor, and saves to .pt

        Args:
        - case_name (str): Name of the case folder
        - save_path (Path): Path to save the processed .pt file

        Returns:
        - Optional[Dict[str, Tensor]]: Dictionary containing processed tensors, or None if failed
        """
        case_path = self.data_dir / case_name

        def natural_key(s):
            name = Path(s).name
            priority = 0 if '-' not in name else 1
            parts = [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]
            return [priority] + parts

        txt_files = sorted(glob.glob(str(case_path / '*.txt')), key=natural_key)

        if not txt_files:
            return None

        seq: List[np.ndarray] = []
        coords_ref: Optional[np.ndarray] = None

        for file_path in txt_files:
            # File format: index, x, y, (z), P, V_x, V_y, (V_z), T
            data = np.loadtxt(file_path, skiprows=1, dtype=np.float32)

            # Determine column indices based on spatial dimension
            if self.spatial_dim == 2:
                # [x:1, y:2], [P:3, V_x:4, V_y:5, T:6]
                c = data[:, 1:3]
                s = data[:, [4, 5, 3, 6]]
            else:
                # [x:1, y:2, z:3], [P:4, V_x:5, V_y:6, V_z:7, T:8]
                c = data[:, 1:4]
                s = data[:, [5, 6, 7, 4, 8]]

            seq.append(s)
            if coords_ref is None:
                coords_ref = c

        if not seq:
            return None

        # Convert to Tensors
        # seq: (seq_len, num_nodes, num_channels)
        seq_tensor = torch.from_numpy(np.stack(seq, axis=0))
        # coords: (num_nodes, spatial_dim)
        coords_tensor = torch.from_numpy(coords_ref)

        data_dict = {'seq': seq_tensor, 'coords': coords_tensor}

        torch.save(data_dict, save_path)

        return data_dict

    @staticmethod
    def build_datasets(split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                       data_dir: Union[str, Path] = './dataset',
                       spatial_dim: int = 2, limits: Optional[Tuple[int, int]] = None
                       ) -> Tuple['CFDataset', 'CFDataset', 'CFDataset']:
        """
        Factory method to automatically discover cases, split them, and return dataset instances.

        Args:
        - split_ratio (Tuple[float, float, float]): Ratios for (train, val, test).
        - data_dir (Union[str, Path]): Root directory of the dataset.
        - spatial_dim (int): Spatial dimension of the data.
        - limits (Optional[Tuple[int, int]]): Maximum number of seq_len and nodes to keep (subsampling).

        Returns:
        - Tuple[CFDataset, CFDataset, CFDataset]: (train set, validation set, test set).
        """
        all_cases = find_case_dirs(data_dir)

        # Shuffle cases for random splitting
        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_cases)

        num_total = len(all_cases)
        num_train = int(num_total * split_ratio[0])
        num_val = int(num_total * split_ratio[1])

        train_cases = all_cases[:num_train]
        val_cases = all_cases[num_train : num_train + num_val]
        test_cases = all_cases[num_train + num_val:]

        train_data = CFDataset(data_dir, train_cases, spatial_dim, limits)
        val_data = CFDataset(data_dir, val_cases, spatial_dim, limits)
        test_data = CFDataset(data_dir, test_cases, spatial_dim, limits)

        return train_data, val_data, test_data

    def __len__(self) -> int:
        """
        Returns the number of cases (sequences) in the dataset.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves a single sequence and its coordinates

        Args:
        - idx (int): Index of the case

        Returns:
        - Tuple[Tensor, Tensor]:
            - sequence: Shape (seq_len, num_nodes, num_channels)
            - coords: Shape (num_nodes, spatial_dim)
        """
        return self.sequences[idx], self.coords[idx]

    def get_normalization_stats(self) -> Tuple[Tensor, Tensor]:
        """
        Computes the global mean and standard deviation of features across the entire dataset.

        Returns:
        - Tuple[Tensor, Tensor]:
            - mean: Shape (1, 1, num_channels)
            - std: Shape (1, 1, num_channels)
        """
        if not self.sequences:
            raise RuntimeError('Dataset is empty. Cannot compute statistics')

        # Concatenate all trajectories along the time/batch dimension to compute global stats
        # Shape: (total_seq_len, num_nodes, num_channels)
        all_data = torch.cat(self.sequences, dim=0)

        # Flatten time and node dimensions
        # Shape: (total_pixels, num_channels)
        flat_data = all_data.view(-1, all_data.shape[-1])

        mean = flat_data.mean(dim=0).view(1, 1, -1)
        std = flat_data.std(dim=0).view(1, 1, -1)

        # Prevent division by zero
        std[std < 1e-7] = 1.0

        return mean, std

    def get_bounds(self) -> Tensor:
        """"
        Computes the bounding box of the geometry across all loaded cases for normalization.

        Returns:
        - Tensor: Bounding box tensor of shape (2, spatial_dim) containing [[mins], [maxs]].
        """
        if not self.coords:
            raise RuntimeError('Dataset is empty. Cannot compute bounds')

        # Initialize mins and maxs with infinity to ensure any coordinate will update them
        # Shape: (spatial_dim,)
        global_min = torch.full((self.spatial_dim,), float('inf'))
        global_max = torch.full((self.spatial_dim,), float('-inf'))

        for c in self.coords:
            # c shape: (num_nodes, spatial_dim)
            case_min = c.min(dim=0)[0]
            case_max = c.max(dim=0)[0]

            global_min = torch.minimum(global_min, case_min)
            global_max = torch.maximum(global_max, case_max)

        # Stack into (2, spatial_dim) -> [[mins...], [maxs...]]
        bounds = torch.stack([global_min, global_max], dim=0)

        return bounds
