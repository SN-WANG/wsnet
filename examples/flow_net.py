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
from typing import Dict, Tuple

import flow_args

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)


from wsnet.models.neural.geofno import GeoFNO

from wsnet.data.flow_data import FlowData
from wsnet.data.CFDRender import CFDAnimation
from wsnet.data.scaler import StandardScalerTensor, MinMaxScalerTensor

from wsnet.training.rollout_trainer import RolloutTrainer
from wsnet.training.base_criterion import Metrics

from wsnet.utils.seeder import seed_everything
from wsnet.utils.hue_logger import hue, logger


# ======================================================================
# 1. Dataset Wrapper with Scalers
# ======================================================================

class ScaledCFDataset(Dataset):
    """
    Wraps FlowData to apply feature standardization and coordinate normalization.

    attributes:
        dataset (FlowData): The underlying raw dataset.
        scaler (StandardScalerTensor): fitted scaler for feature standardization.
        mins (Tensor): Minimum coordinate values. Shape (spatial_dim,).
        maxs (Tensor): Maximum coordinate values. Shape (spatial_dim,).
    """
    def __init__(
            self, dataset: FlowData,
            feature_scaler: StandardScalerTensor,
            coord_scaler: MinMaxScalerTensor
        ):
        """
        Args:
            dataset: Instance of FlowData.
            feature_scaler: Fitted StandardScalerTensor instance.
            coord_scaler: Fitted MinMaxScalerTensor instance.
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
    train_raw, val_raw, _ = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim, win_len=args.win_len, win_stride=args.win_stride
    )

    train_seq = torch.cat(train_raw.seqs, dim=0)
    train_coord = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seq, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coord, channel_dim=-1)

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
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    # --- 3. Training Execution ---
    scalers={"feature_scaler": feature_scaler, "coord_scaler": coord_scaler}

    trainer = RolloutTrainer(model=model, lr=args.lr, max_epochs=args.max_epochs,
                             scalers=scalers, output_dir=output_dir, device=args.device,
                             weight_decay=args.weight_decay, eta_min=args.eta_min,
                             max_rollout_steps=args.max_rollout_steps, rollout_patience=args.rollout_patience,
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

    feature_scaler = StandardScalerTensor()
    feature_scaler.load_state_dict(checkpoint["scaler_state_dict"]["feature_scaler"])
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar")
    coord_scaler.load_state_dict(checkpoint["scaler_state_dict"]["coord_scaler"])

    # --- 2. Data Preparation ---
    _, _, test_raw = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim, win_len=args.win_len, win_stride=args.win_stride
    )

    test_dataset = ScaledCFDataset(test_raw, feature_scaler, coord_scaler)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 3. Model Restoration ---
    model = GeoFNO(in_channels=args.spatial_dim+2, out_channels=args.spatial_dim+2, modes=args.modes,
                   latent_grid_size=args.latent_grid_size, depth=args.depth, width=args.width,
                   deformation_kwargs={'num_layers': args.deform_layers, 'hidden_dim': args.deform_hidden})

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {hue.m}{num_params}{hue.q} parameters")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- 4. Inference and Analysis ---
    logger.info(f'{hue.g}running inference on test set...{hue.q}')
    visualizer = CFDAnimation(output_dir=run_dir, spatial_dim=args.spatial_dim)

    case_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    with torch.no_grad():
        for i, (seq_scaled, coords_norm) in enumerate(test_loader):
            seq_scaled = seq_scaled.to(device)
            coords_norm = coords_norm.to(device)

            case_name = test_raw.case_names[i]
            steps = seq_scaled.shape[1] - 1
            initial_state = seq_scaled[:, 0]

            pred_seq_scaled = model.predict(initial_state, coords_norm, steps)

            pred_seq = feature_scaler.inverse_transform(pred_seq_scaled).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_scaled).cpu().squeeze(0)
            coords_raw = test_raw.coords[i].cpu()

            # compute metrics
            metrics_evaluator = Metrics(channel_names=args.channel_names)
            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch in args.channel_names:
                nmse = metrics[ch]["global"]["nmse"]
                r2 = metrics[ch]["global"]["r2"]
                log_metrics.append(f"{hue.c}{ch}:{hue.q} NMSE={hue.m}{nmse:.2e}{hue.q}, R2={hue.m}{r2:.4f}{hue.q}")

            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_metrics))

            # save predictions
            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            # visualize
            if i == 0:
                logger.info(f"rendering animation for case: {hue.b}{case_name}{hue.q}")
                visualizer.animate_comparison(
                    gt=gt_seq, pred=pred_seq, coords=coords_raw, case_name=case_name)

    # Save all metrics
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    logger.info(f"{hue.g}inference completed.{hue.q}")


# ======================================================================
# 4. Main Execution
# ======================================================================

if __name__ == "__main__":
    args = flow_args.get_args()

    if "train" in args.mode: train_pipeline(args)

    if "infer" in args.mode: inference_pipeline(args)
