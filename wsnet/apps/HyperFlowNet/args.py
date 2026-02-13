# Args Config for Autoregressive Hyper Flow Net Training and Inference
# Author: Shengning Wang

import argparse
import torch

def get_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the HyperFlow-Net training and inference pipeline.

    Returns:
    - argparse.Namespace: The parsed arguments object containing all hyperparameters.
    """
    parser = argparse.ArgumentParser(description="HyperFlowNet: High-Pressure Hydrogen Pipeline Flow Simulation")

    # ----------------------------------------------------------------------
    # 1. General Settings
    # ----------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility across numpy and torch.")
    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Root directory to save checkpoints, logs, and animations.")
    parser.add_argument("--mode", type=str, default="train_infer", choices=["train", "infer", "train_infer"],
                        help="Execution mode: train model, run inference, or both.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device ('cuda', 'cpu', or 'cuda:0').")

    # ----------------------------------------------------------------------
    # 2. Data Configuration
    # ----------------------------------------------------------------------
    parser.add_argument("--data_dir", type=str, default="./dataset", 
                        help="Path to the directory containing simulation case folders.")
    parser.add_argument("--channel_names", type=list, default=["Vx", "Vy", "P", "T"],
                        help="Name list of the channels.")
    parser.add_argument("--spatial_dim", type=int, default=2, choices=[2, 3],
                        help="Spatial dimension of the fluid flow (2D or 3D).")
    parser.add_argument("--win_len", type=int, default=8, 
                        help="Total window length for data slicing (input + label sequence).")
    parser.add_argument("--win_stride", type=int, default=1, 
                        help="Stride for sliding window augmentation.")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of subprocesses for data loading.")

    # ----------------------------------------------------------------------
    # 3. GeoFNO Model Architecture
    # ----------------------------------------------------------------------
    parser.add_argument("--modes", type=int, nargs='+', default=[12, 12], 
                        help="Number of Fourier modes to keep per dimension.")
    parser.add_argument("--latent_grid_size", type=int, nargs='+', default=[64, 64], 
                        help="Resolution of the latent grid for spectral convolutions.")
    parser.add_argument("--depth", type=int, default=4, 
                        help="Number of stacked FNO blocks.")
    parser.add_argument("--width", type=int, default=128, 
                        help="Number of hidden channels in the FNO blocks.")

    # Deformation Net Params
    parser.add_argument("--deform_layers", type=int, default=2, 
                        help="Number of layers in the coordinate deformation network.")
    parser.add_argument("--deform_hidden", type=int, default=32, 
                        help="Hidden dimension size for the deformation network.")

    # ----------------------------------------------------------------------
    # 4. Training Strategy
    # ----------------------------------------------------------------------

    # optimizer (AdamW)
    parser.add_argument("--lr", type=float, default=5e-4, 
                        help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay (L2 regularization) factor.")

    # scheduler (Cosine Annealing LR)
    parser.add_argument("--max_epochs", type=int, default=350, 
                        help="Total number of training epochs.")
    parser.add_argument("--eta_min", type=float, default=1e-6, 
                        help="Minimum learning rate.")

    # curriculum
    parser.add_argument("--max_rollout_steps", type=int, default=7, 
                        help="Maximum autoregressive rollout steps allowed.")
    parser.add_argument("--rollout_patience", type=int, default=40, 
                        help="Epochs of stable loss required to increase rollout difficulty.")
    parser.add_argument("--noise_std_init", type=float, default=0.01, 
                        help="Initial Std dev of Gaussian noise injected into input state.")
    parser.add_argument("--noise_decay", type=float, default=0.7, 
                        help="Decay factor for noise when rollout steps increase.")

    args = parser.parse_args()
    return args
