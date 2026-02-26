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

import args_flow

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
        use_log_pressure:     Whether to apply log1p(p / LOG_P_REF) to the pressure channel.
        pressure_channel_idx: Index of the pressure channel in the feature vector.
    """
    def __init__(
            self, dataset: FlowData,
            feature_scaler: StandardScalerTensor,
            coord_scaler: MinMaxScalerTensor,
            use_log_pressure: bool = False,
            pressure_channel_idx: int = 2,
        ):
        self.dataset = dataset
        self.feature_scaler = feature_scaler
        self.coord_scaler = coord_scaler
        self.use_log_pressure = use_log_pressure
        self.pressure_channel_idx = pressure_channel_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        seq, coords = self.dataset[idx]

        # 1. Log-pressure transform (before standardization)
        if self.use_log_pressure:
            seq = seq.clone()
            seq[..., self.pressure_channel_idx] = torch.log1p(
                seq[..., self.pressure_channel_idx] / LOG_P_REF
            )

        # 2. Feature standardization (Mean/Std)
        seq_std = self.feature_scaler.transform(seq)

        # 3. Coordinate normalization (Min/Max -> [-1, 1])
        coords_norm = self.coord_scaler.transform(coords)

        return seq_std, coords_norm


def _inverse_transform_pressure(seq: Tensor, p_idx: int) -> Tensor:
    """Invert log1p(p / LOG_P_REF) -> physical pressure on a sequence tensor."""
    seq = seq.clone()
    seq[..., p_idx] = torch.expm1(seq[..., p_idx]) * LOG_P_REF
    return seq


# ======================================================================
# Model factory
# ======================================================================

def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Instantiate the model selected by --model_type."""
    in_ch = out_ch = args.spatial_dim + 2  # [Vx, Vy, (Vz,) P, T]
    deform_kw = {"num_layers": args.deform_layers, "hidden_dim": args.deform_hidden}

    if args.model_type == "geofno":
        return GeoFNO(
            in_channels=in_ch, out_channels=out_ch,
            modes=args.modes, latent_grid_size=args.latent_grid_size,
            depth=args.depth, width=args.width,
            deformation_kwargs=deform_kw,
        )
    elif args.model_type == "geowno":
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


# ======================================================================
# 2. Training Pipeline
# ======================================================================

def train_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the training workflow: Data loading -> Data Scaling -> Model Init ->  Model Training.

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

    train_seqs = torch.cat(train_raw.seqs, dim=0)
    train_coords = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)

    dataset_kw = dict(
        feature_scaler=feature_scaler,
        coord_scaler=coord_scaler,
        use_log_pressure=args.use_log_pressure,
        pressure_channel_idx=args.spatial_dim,
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

    # --- 3. Training Execution ---
    scalers = {"feature_scaler": feature_scaler, "coord_scaler": coord_scaler}

    trainer = RolloutTrainer(
        # base params
        model=model, lr=args.lr, max_epochs=args.max_epochs,
        scalers=scalers, output_dir=output_dir, device=args.device,
        # optimization params
        weight_decay=args.weight_decay, eta_min=args.eta_min,
        # curriculum params
        max_rollout_steps=args.max_rollout_steps, rollout_patience=args.rollout_patience,
        noise_std_init=args.noise_std_init, noise_decay=args.noise_decay,
        # physics params
        use_physics_loss=args.use_physics_loss,
        lambda_physics=args.lambda_physics,
        lambda_mass=args.lambda_mass,
        lambda_momentum=args.lambda_momentum,
        lambda_energy=args.lambda_energy,
        latent_grid_size=args.latent_grid_size,
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

    # --- 2. Data Preparation ---
    _, _, test_raw = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    test_dataset = ScaledCFDataset(
        test_raw, feature_scaler=feature_scaler, coord_scaler=coord_scaler,
        use_log_pressure=args.use_log_pressure, pressure_channel_idx=args.spatial_dim,
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

    case_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    with torch.no_grad():
        for i, (seq_std, coords_norm) in enumerate(test_loader):
            seq_std = seq_std.to(device)
            coords_norm = coords_norm.to(device)

            # ------------------------------------------------------------------

            case_name = test_raw.case_names[i]
            steps = seq_std.shape[1] - 1
            initial_state = seq_std[:, 0]
            coords_raw = test_raw.coords[i].cpu()

            # ------------------------------------------------------------------

            pred_seq_std = model.predict(initial_state, coords_norm, steps)

            # ------------------------------------------------------------------

            pred_seq = feature_scaler.inverse_transform(pred_seq_std).cpu().squeeze(0)
            gt_seq = feature_scaler.inverse_transform(seq_std).cpu().squeeze(0)

            if args.use_log_pressure:
                pred_seq = _inverse_transform_pressure(pred_seq, args.spatial_dim)
                gt_seq = _inverse_transform_pressure(gt_seq, args.spatial_dim)

            # ------------------------------------------------------------------

            metrics_evaluator = Metrics(channel_names=args.channel_names)
            metrics = metrics_evaluator.compute(pred_seq, gt_seq)
            case_metrics[case_name] = metrics

            log_metrics = []
            for ch in args.channel_names:
                nmse = metrics[ch]["global"]["nmse"]
                r2 = metrics[ch]["global"]["r2"]
                log_metrics.append(f"{hue.c}{ch}:{hue.q} NMSE={hue.m}{nmse:.2e}{hue.q}, R2={hue.m}{r2:.4f}{hue.q}")

            logger.info(f"case {hue.b}{case_name}{hue.q} | " + " | ".join(log_metrics))

            # ------------------------------------------------------------------

            torch.save(pred_seq, run_dir / f"{case_name}_pred.pt")

            # ------------------------------------------------------------------

            if i == 0:
                logger.info(f"rendering animation for case: {hue.b}{case_name}{hue.q}")
                visualizer.animate_comparison(
                    gt=gt_seq, pred=pred_seq, coords=coords_raw, case_name=case_name)


    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(case_metrics, f, indent=4)

    logger.info(f"{hue.g}inference completed.{hue.q}")


# ======================================================================
# 4. Probe Pipeline (OOM pre-flight check)
# ======================================================================

def probe_pipeline(args: argparse.Namespace) -> None:
    """
    Run one full-rollout training step on real training data to check peak VRAM usage.
    Loads and augments the training split (identical to train_pipeline) so that node
    count, window length, and batch size all reflect actual training conditions.
    Reports SAFE / WARNING / CRITICAL verdict.
    """
    device = torch.device(args.device)
    if not torch.cuda.is_available():
        logger.warning("No CUDA device — probe skipped (CPU has no OOM risk).")
        return

    # --- 1. Data Preparation (mirrors train_pipeline) ---
    logger.info("loading training data for probe...")
    train_raw, _, _ = FlowData.spawn(
        data_dir=args.data_dir, spatial_dim=args.spatial_dim,
        win_len=args.win_len, win_stride=args.win_stride,
    )

    train_seqs = torch.cat(train_raw.seqs, dim=0)
    train_coords = torch.cat(train_raw.coords, dim=0)
    feature_scaler = StandardScalerTensor().fit(train_seqs, channel_dim=-1)
    coord_scaler = MinMaxScalerTensor(norm_range="bipolar").fit(train_coords, channel_dim=-1)

    probe_dataset = ScaledCFDataset(
        train_raw, feature_scaler=feature_scaler, coord_scaler=coord_scaler,
        use_log_pressure=args.use_log_pressure, pressure_channel_idx=args.spatial_dim,
    )
    probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    seq_std, coords_norm = next(iter(probe_loader))
    seq_std     = seq_std.to(device)
    coords_norm = coords_norm.to(device)

    B, T, N, C = seq_std.shape

    # --- 2. Model Initialization ---
    model = _build_model(args)
    model.to(device).train()

    logger.info(f"{hue.y}probe config:{hue.q} "
                f"batch={hue.m}{B}{hue.q}, "
                f"frames={hue.m}{T}{hue.q}, "
                f"nodes={hue.m}{N}{hue.q}, "
                f"channels={hue.m}{C}{hue.q}, "
                f"max_rollout={hue.m}{args.max_rollout_steps}{hue.q}, "
                f"model={hue.b}{args.model_type}{hue.q}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = NMSECriterion()

    # --- 3. Forward / Backward Pass ---
    torch.cuda.reset_peak_memory_stats(device)

    input_state = seq_std[:, 0]
    loss = torch.tensor(0.0, device=device)
    for t in range(args.max_rollout_steps):
        if hasattr(model, "time_encoder"):
            pred = model(input_state, coords_norm, step=t)
        else:
            pred = model(input_state, coords_norm)
        loss = loss + criterion(pred, seq_std[:, t + 1])
        input_state = pred.detach()
    (loss / args.max_rollout_steps).backward()
    optimizer.step()

    # --- 4. VRAM Report ---
    peak  = torch.cuda.max_memory_allocated(device)
    total = torch.cuda.get_device_properties(device).total_memory
    pct   = 100 * peak / total
    if pct < 75:
        status = f"{hue.g}SAFE{hue.q}"
    elif pct < 92:
        status = f"{hue.y}WARNING — close to limit{hue.q}"
    else:
        status = f"{hue.r}CRITICAL — likely OOM in real training{hue.q}"

    logger.info(f"{hue.y}device: {hue.b}{torch.cuda.get_device_name(device)}{hue.q}  "
                f"({hue.m}{total / 1e9:.1f}{hue.q} GB)")
    logger.info(f"peak usage: {hue.m}{peak / 1e9:.2f}{hue.q} GB  "
                f"({hue.m}{pct:.1f}{hue.q} %)  →  {status}")

    logger.info(f"{hue.g}probe completed.{hue.q}\n")


# ======================================================================
# 5. Main Execution
# ======================================================================

if __name__ == "__main__":
    args = args_flow.get_args()

    if "probe" in args.mode:  probe_pipeline(args)
    if "train" in args.mode:  train_pipeline(args)
    if "infer" in args.mode:  inference_pipeline(args)
