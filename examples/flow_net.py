# Main Script for Autoregressive Hyper Flow Net Training and Inference
# Author: Shengning Wang

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple

import flow_args

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path: sys.path.insert(0, project_root)


from wsnet.models.neural.geofno import GeoFNO
from wsnet.models.neural.geowno import GeoWNO

from wsnet.data.flow_data import FlowData
from wsnet.data.flow_vis import FlowVis
from wsnet.data.scaler import StandardScalerTensor, MinMaxScalerTensor

from wsnet.training.rollout_trainer import RolloutTrainer
from wsnet.training.base_criterion import Metrics, NMSECriterion

from wsnet.utils.seeder import seed_everything
from wsnet.utils.hue_logger import hue, logger


# Log-pressure reference (Pa): p_log = log1p(p / LOG_P_REF)
LOG_P_REF: float = 1e5


# ======================================================================
# 1. Dataset Wrapper with Scalers
# ======================================================================

class ScaledCFDataset(Dataset):
    """
    Wraps FlowData to apply log-pressure transform, feature standardization,
    and coordinate normalization.

    The log-pressure transform is applied BEFORE fitting the scaler so that
    the scaler operates on the compressed representation.

    Attributes:
        dataset:              The underlying raw dataset.
        feature_scaler:       Fitted StandardScalerTensor.
        coord_scaler:         Fitted MinMaxScalerTensor.
        log_pressure:         Whether to apply log1p(p / LOG_P_REF) to the pressure channel.
        pressure_channel_idx: Index of the pressure channel in the feature vector.
    """
    def __init__(
            self, dataset: FlowData,
            feature_scaler: StandardScalerTensor,
            coord_scaler: MinMaxScalerTensor,
            log_pressure: bool = False,
            pressure_channel_idx: int = 2,
        ):
        self.dataset = dataset
        self.feature_scaler = feature_scaler
        self.coord_scaler = coord_scaler
        self.log_pressure = log_pressure
        self.pressure_channel_idx = pressure_channel_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq, coords = self.dataset[idx]

        # 1. Log-pressure transform (before standardization)
        if self.log_pressure:
            seq = seq.clone()
            seq[..., self.pressure_channel_idx] = torch.log1p(
                seq[..., self.pressure_channel_idx] / LOG_P_REF
            )

        # 2. Feature standardization (Mean/Std)
        seq_scaled = self.feature_scaler.transform(seq)

        # 3. Coordinate normalization (Min/Max -> [-1, 1])
        coords_norm = self.coord_scaler.transform(coords)

        return seq_scaled, coords_norm


# ======================================================================
# Model factory
# ======================================================================

def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Instantiate the model selected by --model_type."""
    in_ch = out_ch = args.spatial_dim + 2  # [Vx, Vy, (Vz,) P, T]
    deform_kw = {'num_layers': args.deform_layers, 'hidden_dim': args.deform_hidden}

    if args.model_type == 'geofno':
        return GeoFNO(
            in_channels=in_ch, out_channels=out_ch,
            modes=args.modes, latent_grid_size=args.latent_grid_size,
            depth=args.depth, width=args.width,
            deformation_kwargs=deform_kw,
        )
    elif args.model_type == 'geowno':
        return GeoWNO(
            in_channels=in_ch, out_channels=out_ch,
            modes=args.modes, latent_grid_size=args.latent_grid_size,
            depth=args.depth, width=args.width,
            deformation_kwargs=deform_kw,
            knn_k=args.knn_k,
            rff_features=args.rff_features,
            time_features=args.time_features,
            max_steps=args.win_len,
            rff_sigma=args.rff_sigma,
        )
    else:
        raise ValueError(f"Unknown model_type '{args.model_type}'. Choices: geofno, geowno")


def _inverse_transform_pressure(seq: Tensor, p_idx: int) -> Tensor:
    """Invert log1p(p / LOG_P_REF) -> physical pressure on a sequence tensor."""
    seq = seq.clone()
    seq[..., p_idx] = torch.expm1(seq[..., p_idx]) * LOG_P_REF
    return seq


# ======================================================================
# 2. Training Pipeline
# ======================================================================

def train_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the training workflow: Data loading -> Normalization -> Model Init -> Training.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Data Preparation ---
    logger.info("initializing datasets...")
    train_raw, val_raw, _ = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    # Pressure channel index: [Vx, Vy, P, T] in 2D → idx 2; [Vx, Vy, Vz, P, T] in 3D → idx 3
    pressure_channel_idx: int = args.spatial_dim

    # Fit scalers on (optionally log-transformed) training data
    train_seqs = torch.cat(train_raw.seqs, dim=0)
    if args.use_log_pressure:
        train_seqs = train_seqs.clone()
        train_seqs[..., pressure_channel_idx] = torch.log1p(
            train_seqs[..., pressure_channel_idx] / LOG_P_REF
        )

    train_coord = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coord, channel_dim=-1)

    dataset_kw = dict(
        feature_scaler=feature_scaler,
        coord_scaler=coord_scaler,
        log_pressure=args.use_log_pressure,
        pressure_channel_idx=pressure_channel_idx,
    )
    train_dataset = ScaledCFDataset(train_raw, **dataset_kw)
    val_dataset = ScaledCFDataset(val_raw, **dataset_kw)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- 2. Model Initialization ---
    logger.info(f"instantiating model: {hue.b}{args.model_type}{hue.q}...")
    model = _build_model(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model has {hue.m}{num_params}{hue.q} parameters")

    # Physics grid size for FD loss (use latent_grid_size if physics loss enabled)
    physics_grid_size = args.latent_grid_size if args.use_physics_loss else None

    # --- 3. Training Execution ---
    scalers = {"feature_scaler": feature_scaler, "coord_scaler": coord_scaler}

    trainer = RolloutTrainer(
        model=model, lr=args.lr, max_epochs=args.max_epochs,
        scalers=scalers, output_dir=output_dir, device=args.device,
        weight_decay=args.weight_decay, eta_min=args.eta_min,
        max_rollout_steps=args.max_rollout_steps, rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init, noise_decay=args.noise_decay,
        use_physics_loss=args.use_physics_loss,
        lambda_phy=args.lambda_phy,
        lambda_mass=args.lambda_mass,
        lambda_momentum=args.lambda_momentum,
        lambda_energy=args.lambda_energy,
        physics_grid_size=physics_grid_size,
    )

    trainer.fit(train_loader, val_loader)


# ======================================================================
# 3. Inference Pipeline
# ======================================================================

def inference_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the testing workflow using artifacts from the training phase.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
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

    pressure_channel_idx: int = args.spatial_dim

    # --- 2. Data Preparation ---
    _, _, test_raw = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    test_dataset = ScaledCFDataset(
        test_raw, feature_scaler=feature_scaler, coord_scaler=coord_scaler,
        log_pressure=args.use_log_pressure, pressure_channel_idx=pressure_channel_idx,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # --- 3. Model Restoration ---
    model = _build_model(args)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {hue.m}{num_params}{hue.q} parameters")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- 4. Inference and Analysis ---
    logger.info(f'{hue.g}running inference on test set...{hue.q}')
    visualizer = FlowVis(output_dir=run_dir, spatial_dim=args.spatial_dim)

    case_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    with torch.no_grad():
        for i, (seq_scaled, coords_norm) in enumerate(test_loader):
            seq_scaled = seq_scaled.to(device)
            coords_norm = coords_norm.to(device)

            case_name = test_raw.case_names[i]
            steps = seq_scaled.shape[1] - 1
            initial_state = seq_scaled[:, 0]

            pred_seq_scaled = model.predict(initial_state, coords_norm, steps)

            # Inverse-transform: unstandardize first
            pred_seq = feature_scaler.inverse_transform(pred_seq_scaled).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_scaled).cpu().squeeze(0)

            # Invert log-pressure if applied
            if args.use_log_pressure:
                pred_seq = _inverse_transform_pressure(pred_seq, pressure_channel_idx)
                gt_seq = _inverse_transform_pressure(gt_seq, pressure_channel_idx)

            coords_raw = test_raw.coords[i].cpu()

            # Compute metrics
            metrics_evaluator = Metrics(channel_names=args.channel_names)
            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch in args.channel_names:
                nmse = metrics[ch]["global"]["nmse"]
                r2 = metrics[ch]["global"]["r2"]
                log_metrics.append(f"{hue.c}{ch}:{hue.q} NMSE={hue.m}{nmse:.2e}{hue.q}, R2={hue.m}{r2:.4f}{hue.q}")

            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_metrics))

            # Save predictions
            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            # Visualize first case
            if i == 0:
                logger.info(f"rendering animation for case: {hue.b}{case_name}{hue.q}")
                visualizer.animate_comparison(
                    gt=gt_seq, pred=pred_seq, coords=coords_raw, case_name=case_name)

    # Save all metrics
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    logger.info(f"{hue.g}inference completed.{hue.q}")


# ======================================================================
# 4. Probe Pipeline (OOM pre-flight check)
# ======================================================================

def probe_pipeline(args: argparse.Namespace) -> None:
    """
    Run one full-rollout training step on synthetic data to check peak VRAM usage.
    No dataset required. Reports SAFE / WARNING / CRITICAL verdict.
    """
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        logger.warning("No CUDA device — probe skipped (CPU has no OOM risk).")
        return

    model = _build_model(args)
    model.to(device).train()

    N, C = args.probe_n, args.spatial_dim + 2
    B, win_len = args.batch_size, args.win_len

    logger.info(f"probe config: batch={B}, nodes={N}, win_len={win_len}, "
                f"max_rollout={args.max_rollout_steps}, model={args.model_type}")

    seq    = torch.randn(B, win_len, N, C, device=device)
    coords = torch.rand(B, N, args.spatial_dim, device=device) * 2 - 1

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = NMSECriterion()

    torch.cuda.reset_peak_memory_stats(device)

    input_state = seq[:, 0]
    loss = torch.tensor(0.0, device=device)
    for t in range(args.max_rollout_steps):
        if hasattr(model, 'time_encoder'):
            pred = model(input_state, coords, step=t)
        else:
            pred = model(input_state, coords)
        loss = loss + criterion(pred, seq[:, t + 1])
        input_state = pred.detach()
    (loss / args.max_rollout_steps).backward()
    optimizer.step()

    peak  = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct   = 100 * peak / total
    if pct < 75:
        status = "SAFE"
    elif pct < 92:
        status = "WARNING — close to limit"
    else:
        status = "CRITICAL — likely OOM in real training"

    logger.info(f"Device : {torch.cuda.get_device_name(device)}  ({total / 1e9:.1f} GB)")
    logger.info(f"Peak   : {peak / 1e9:.2f} GB  ({pct:.1f}%)  →  {status}")


# ======================================================================
# 5. Main Execution
# ======================================================================

if __name__ == "__main__":
    args = flow_args.get_args()

    if args.mode == "probe":  probe_pipeline(args)
    if "train" in args.mode:  train_pipeline(args)
    if "infer" in args.mode:  inference_pipeline(args)
