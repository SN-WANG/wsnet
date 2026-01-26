# Main Script for Autoregressive Hyper Flow Net Training and Inference
# Author: Shengning Wang

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple

import args

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets import GeoFNO
from wsnet.utils import (
    sl, logger, seed_everything, compute_ar_metrics, TensorScaler, AutoregressiveTrainer, CFDataset, CFDAnimation
)


# ======================================================================
# 1. Dataset Wrapper with Normalization
# ======================================================================

class CoordinateScaler:
    """
    Manages coordinate normalization (MinMax Scaling to [-1, 1]).
    Follows the state_dict interface for unified checkpointing.
    """
    def __init__(self, spatial_dim: int):
        self.spatial_dim = spatial_dim
        self.mins = torch.full((spatial_dim,), float('inf'))
        self.maxs = torch.full((spatial_dim,), float('-inf'))
        self._is_fitted = False

    def fit(self, coords_list: List[Tensor]) -> 'CoordinateScaler':
        """
        Computes global min/max from a list of coordinate tensors.

        Args:
            coords_list: List of tensors, each shape (num_nodes, spatial_dim).
        """
        for c in coords_list:
            self.mins = torch.minimum(self.mins, c.min(dim=0)[0])
            self.maxs = torch.maximum(self.maxs, c.max(dim=0)[0])

        diff = self.maxs - self.mins
        diff[diff < 1e-7] = 1.0
        self.maxs = self.mins + diff

        self._is_fitted = True
        return self

    def transform(self, coords: Tensor) -> Tensor:
        """
        Normalizes coordinates to [-1, 1].
        """
        if not self._is_fitted:
            raise RuntimeError('CoordinateScaler must be fitted before transform.')

        self.mins = self.mins.to(coords.device)
        self.maxs = self.maxs.to(coords.device)

        return 2.0 * (coords - self.mins) / (self.maxs - self.mins) - 1.0

    def state_dict(self) -> Dict[str, Tensor]:
        return {"mins": self.mins, "maxs": self.maxs}

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        self.mins = state_dict["mins"]
        self.maxs = state_dict["maxs"]
        self._is_fitted = True


class ScaledCFDataset(Dataset):
    """
    Wraps CFDataset to apply feature standardization and coordinate normalization.

    attributes:
        dataset (CFDataset): The underlying raw dataset.
        scaler (TensorScaler): fitted scaler for feature standardization.
        mins (Tensor): Minimum coordinate values. Shape (spatial_dim,).
        maxs (Tensor): Maximum coordinate values. Shape (spatial_dim,).
    """
    def __init__(self, dataset: CFDataset, feature_scaler: TensorScaler, coord_scaler: CoordinateScaler):
        """
        Args:
            dataset: Instance of CFDataset.
            feature_scaler: Fitted TensorScaler instance.
            coord_scaler: Fitted CoordinateScaler instance.
        """
        self.dataset = dataset
        self.feature_scaler = feature_scaler
        self.coord_scaler = coord_scaler

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves and normalizes a sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Tensor, Tensor]:
                seq_scaled: Standardized sequence. Shape (seq_len, num_nodes, num_channels).
                coords_norm: Normalized coordinates in [-1, 1]. Shape (num_nodes, spatial_dim).
        """
        seq, coords = self.dataset[idx]

        # 1. Feature standardization (Mean/Std)
        seq_scaled = self.feature_scaler.transform(seq)

        # 2. Coordinate normalization (Min/Max -> [-1, 1])
        coords_norm = self.coord_scaler.transform(coords)

        return seq_scaled, coords_norm


# ======================================================================
# 2. Training Pipeline
# ======================================================================

def train_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the training workflow: Data loading -> Normalization -> Model Init -> Training.

    Args:
        args (argsparse.Namespace): Parsed command line arguments.
    """
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Preparation ---
    logger.info("initializing datasets...")
    train_raw, val_raw, _ = CFDataset.build_datasets(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim, win_len=args.win_len, win_stride=args.win_stride)

    train_seq = torch.cat(train_raw.sequences, dim=0)
    feature_scaler = TensorScaler().fit(train_seq, feature_dim=-1)
    coord_scaler = CoordinateScaler(args.spatial_dim).fit(train_raw.coords)

    train_dataset = ScaledCFDataset(train_raw, feature_scaler, coord_scaler)
    val_dataset = ScaledCFDataset(val_raw, feature_scaler, coord_scaler)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- 2. Model Initialization ---
    logger.info("instantiating model...")
    model = GeoFNO(in_channels=args.spatial_dim+2, out_channels=args.spatial_dim+2, modes=args.modes,
                   latent_grid_size=args.latent_grid_size, depth=args.depth, width=args.width,
                   deformation_kwargs={'num_layers': args.deform_layers, 'hidden_dim': args.deform_hidden})

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {sl.m}{num_params}{sl.q} parameters")

    # --- 3. Training Execution ---
    scalers={"feature_scaler": feature_scaler, "coord_scaler": coord_scaler}

    trainer = AutoregressiveTrainer(model=model, max_epochs=args.max_epochs, patience=args.patience,
                                    scalers=scalers, output_dir=output_dir, device=args.device,
                                    lr=args.lr, weight_decay=args.weight_decay, scheduler_t0=args.scheduler_t0,
                                    scheduler_t_mult=args.scheduler_t_mult, eta_min=args.eta_min,
                                    max_rollout_steps=args.max_rollout_steps,
                                    curr_patience=args.curr_patience, curr_sensitivity=args.curr_sensitivity,
                                    noise_std_init=args.noise_std_init, noise_decay=args.noise_decay)

    trainer.fit(train_loader, val_loader)


# ======================================================================
# 3. Inference Pipeline
# ======================================================================

def inference_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the testing workflow using artifacts from the training phase.
    Independent of training logic.

    Args:
        args (argsparse.Namespace): Parsed command line arguments.
    """
    device = torch.device(args.device)

    run_dir = Path(args.output_dir)
    model_path = run_dir / "ckpt.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"ckpt.pt not found at {model_path}.")

    # --- 1. Restore State ---
    logger.info("loading training artifacts...")
    checkpoint = torch.load(model_path, map_location=device)

    feature_scaler = TensorScaler()
    feature_scaler.load_state_dict(checkpoint["scaler_state_dict"]["feature_scaler"])
    coord_scaler = CoordinateScaler(args.spatial_dim)
    coord_scaler.load_state_dict(checkpoint["scaler_state_dict"]["coord_scaler"])

    # --- 2. Data Preparation ---
    _, _, test_raw = CFDataset.build_datasets(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim, win_len=args.win_len, win_stride=args.win_stride)

    test_dataset = ScaledCFDataset(test_raw, feature_scaler, coord_scaler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 3. Model Restoration ---
    model = GeoFNO(in_channels=args.spatial_dim+2, out_channels=args.spatial_dim+2, modes=args.modes,
                   latent_grid_size=args.latent_grid_size, depth=args.depth, width=args.width,
                   deformation_kwargs={'num_layers': args.deform_layers, 'hidden_dim': args.deform_hidden})

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {sl.m}{num_params}{sl.q} parameters")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- 4. Inference and Analysis ---
    logger.info(f'{sl.g}running inference on test set...{sl.q}')
    visualizer = CFDAnimation(output_dir=run_dir, spatial_dim=args.spatial_dim)

    case_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    with torch.no_grad():
        for i, (seq_scaled, coords_norm) in enumerate(test_loader):
            seq_scaled = seq_scaled.to(device)
            coords_norm = coords_norm.to(device)

            case_name = test_raw.case_list[i]
            steps = seq_scaled.shape[1] - 1
            initial_state = seq_scaled[:, 0]

            pred_seq_scaled = model.predict(initial_state, coords_norm, steps)

            pred_seq = feature_scaler.inverse_transform(pred_seq_scaled).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_scaled).cpu().squeeze(0)
            coords_raw = test_raw.coords[i].cpu()

            # compute metrics
            metrics = compute_ar_metrics(gt_seq, pred_seq, channel_names=args.channel_names)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch in args.channel_names:
                nmse = metrics[ch]["NMSE"]
                r2 = metrics[ch]["R2"]
                log_metrics.append(f"{sl.c}{ch}:{sl.q} NMSE={sl.m}{nmse:.2e}{sl.q}, R2={sl.m}{r2:.4f}{sl.q}")

            logger.info(f"case {sl.b}{case_name}{sl.q} | " + " | ".join(log_metrics))

            # save predictions
            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            # visualize
            if i == 0:
                logger.info(f'rendering animation for case: {sl.b}{case_name}{sl.q}')
                visualizer.animate_comparison(
                    gt=gt_seq, pred=pred_seq, coords=coords_raw, case_name=case_name)

    # Save all metrics
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    logger.info(f"{sl.g}inference completed.{sl.q}")


# ======================================================================
# 4. Main Execution
# ======================================================================

if __name__ == "__main__":
    args = args.get_args()

    if "train" in args.mode: train_pipeline(args)

    if "infer" in args.mode: inference_pipeline(args)
