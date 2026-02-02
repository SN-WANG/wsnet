# Deep Learning Engine
# Author: Shengning Wang

import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from tqdm.auto import tqdm


# ======================================================================
# 1. Logger & Seed Setter
# ======================================================================

class SmartLogger:
    """
    A thread-safe logging utility synchronized with tqdm progress bars.

    This class manages a specialized logging handler that prevents log messages 
    from breaking tqdm progress bar rendering by using tqdm.write(). It ensures 
    idempotency by clearing existing handlers on the logger instance, which is 
    critical when re-initializing experiments in interactive environments.

    Attributes:
        logger (logging.Logger): The configured logger instance.
    """

    # ANSI Color Codes
    b = "\033[1;34m"    # major key/parameter:      bold blue
    c = "\033[1;36m"    # minor key/parameter:      bold cyan
    m = "\033[1;35m"    # value/reading:            bold magenta
    y = "\033[1;33m"    # warning/highlighting:     bold yellow
    g = "\033[1;32m"    # success/save:             bold green
    r = "\033[1;31m"    # error/critical:           bold red

    q = "\033[0m"      # quit/reset

    def __init__(self, name: str = __name__, level: int = logging.INFO) -> None:
        """
        Initializes the SmartLogger with a tqdm-compatible stream handler.

        Args:
            name (str): The name of the logger, typically __name__.
            level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # disable propagation to prevent duplicate logs from the root logger
        self.logger.propagate = False

        # check for existing handlers to avoid redundant log outputs in singleton logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        tqdm_handler: logging.StreamHandler = self._get_tqdm_handler()
        self.logger.addHandler(tqdm_handler)

    def _get_tqdm_handler(self) -> logging.StreamHandler:
        """
        Constructs a StreamHandler that routes messages through tqdm.write.

        Returns:
            logging.StreamHandler: Configured handler with ANSI color support.
        """
        # define color-coded format for enhanced visibility in terminal, Green for timestamp, Blue for levelname
        log_format: str = f"\033[90m%(asctime)s{self.q} - {self.b}%(levelname)s{self.q} - %(message)s"
        formatter: logging.Formatter = logging.Formatter(log_format, "%H:%M:%S")

        # direct stream to stdout
        handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)

        # override emit to use tqdm.write, ensuring logs appear above progress bars
        handler.emit = lambda record: tqdm.write(formatter.format(record))

        handler.setFormatter(formatter)
        return handler

sl = SmartLogger()
logger: logging.Logger = sl.logger


def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'global seed set to {sl.m}{seed}{sl.q}')


# ======================================================================
# 2. Metrics & Loss Functions
# ======================================================================

class NMSELoss(nn.Module):
    """
    Normalized Mean Squared Error Loss.
    L = ||y - pred||^2 / (||y||^2 + epsilon)
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred (Tensor): Prediction. Shape (batch_size, seq_len, num_nodes, num_channels)
            target (Tensor): Ground truth. Shape (batch_size, seq_len, num_nodes, num_channels)

        Returns:
            Tensor: Scalar loss.
        """
        mse = torch.sum((target - pred) ** 2)
        norm = torch.sum(target ** 2) + self.eps
        return mse / norm


def compute_ar_metrics(gt: Tensor, pred: Tensor, channel_names: List[str]) -> Dict[str, Any]:
    """
    Computes comprehensive autoregressive evaluation metrics.

    Args:
        gt (Tensor): Ground truth sequence. Shape (seq_len, num_nodes, num_channels)
        pred (Tensor): Predicted sequence. Shape (seq_len, num_nodes, num_channels)
        channel_names (List[str]): List of channel names.

    Returns:
        Dict[str, Any]: Nested dictionary of metrics per channel and aggregate.
    """
    assert gt.shape == pred.shape, "Shape mismatch between GT and Pred"
    assert gt.shape[-1] == len(channel_names), \
        f"Channel mismatch: Tensor has {gt.shape[-1]} channels but provided {len(channel_names)} names."

    metrics: Dict[str, Any] = {}

    # 1. Per-Channel Metrics
    for c, name in enumerate(channel_names):
        gt_c = gt[..., c]  # (seq_len, num_nodes)
        pred_c = pred[..., c]  # (seq_len, num_nodes)

        abs_diff = torch.abs(gt_c - pred_c)
        sq_diff = (gt_c - pred_c) ** 2

        # 1. Global MAE and MaxErr
        mae_val = torch.mean(abs_diff).item()
        max_val = torch.max(abs_diff).item()

        # 2. Global MSE and RMSE
        mse_val = torch.mean(sq_diff).item()
        rmse_val = np.sqrt(mse_val)

        # 3. Global R2 and NMSE
        ss_res = torch.sum(sq_diff).item()
        ss_tot = torch.sum((gt_c - torch.mean(gt_c)) ** 2).item()
        r2_val = 1.0 - (ss_res / (ss_tot + 1e-8))
        nmse_val = mse_val / (torch.mean(gt_c ** 2).item() + 1e-8)

        # 5. Step-wise MAE and MaxErr
        step_mae = torch.mean(abs_diff, dim=1)
        step_max = torch.max(abs_diff, dim=1).values

        # 6. Step-wise MSE and RMSE
        step_mse = torch.mean(sq_diff, dim=1)
        step_rmse = torch.sqrt(step_mse)

        # 7. Step-wise R2 and NMSE
        step_ss_res = torch.sum(sq_diff, dim=1)
        step_ss_tot = torch.sum((gt_c - torch.mean(gt_c, dim=1, keepdim=True)) ** 2, dim=1)
        step_r2 = (1.0 - (step_ss_res / (step_ss_tot + 1e-8)))
        step_nmse = step_mse / (torch.mean(gt_c ** 2, dim=1) + 1e-8)

        metrics[name] = {
            "NMSE": nmse_val,
            "R2": r2_val,
            "MSE": mse_val,
            "RMSE": rmse_val,
            "MAE": mae_val,
            "MaxErr": max_val,
            "Step-NMSE": step_nmse.tolist(),
            "Step-R2": step_r2.tolist(),
        }

    return metrics


# ======================================================================
# 3. Data Processing & Standardization
# ======================================================================

class BaseDataset(Dataset):
    """
    A generic dataset wrapper for input-target tensor pairs.
    """

    def __init__(self, inputs: Tensor, targets: Tensor):
        """
        Args:
        - inputs (Tensor): Input data. Shape: (num_samples, *input_dims)
        - targets (Tensor): Target data. Shape: (num_samples, *output_dims)
        """
        assert len(inputs) == len(targets), 'Inputs and targets must have the same length.'
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.inputs[idx], self.targets[idx]


class TensorScaler:
    """
    Standardizes tensors by removing the mean and scaling to unit variance.
    Supports operation over specific dimensions (e.g., channel-wise normalization).
    """

    def __init__(self):
        self.mean: Optional[Tensor] = None
        self.std: Optional[Tensor] = None
        self.device: Optional[torch.device] = None

    def fit(self, x: Tensor, feature_dim: int = 1) -> 'TensorScaler':
        """
        Computes the mean and std to be used for later scaling.

        Args:
            x (Tensor): Data tensor. Shape: (N, ...)
            feature_dim (int): The dimension representing features/channels.
                               Statistics are computed over all OTHER dimensions.

        Returns:
            TensorScaler: Self instance for method chaining.
        """
        # ensure positive index
        feature_dim = feature_dim % x.ndim

        # aggregate over all dimensions except the feature dimension
        dims = [d for d in range(x.ndim) if d != feature_dim]

        self.mean = x.mean(dim=dims, keepdim=True)
        self.std = x.std(dim=dims, keepdim=True)

        # handle constant values (std=0) to avoid NaN
        self.std[self.std < 1e-7] = 1.0
        self.device = x.device

        return self

    def transform(self, x: Tensor) -> Tensor:
        """
        Standardizes input tensor.

        Args:
            x (Tensor): Data to transform. Shape matches input to fit().

        Returns:
            Tensor: Scaled data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Scaler has not been fitted.')

        # ensure scaler stats are on the same device as input
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Scales back the data to the original representation.

        Args:
            x (Tensor): Scaled data.

        Returns:
            Tensor: Original scale data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Scaler has not been fitted.')

        # ensure scaler stats are on the same device as input
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, Tensor]:
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# ======================================================================
# 4. Core Engine: The Base Trainer
# ======================================================================

class BaseTrainer:
    """
    Base class encapsulating the training loop, checkpointing, and evaluation logic.
    Subclasses must implement 'compute_loss' to define specific task logic.
    """

    def __init__(self, model: nn.Module, max_epochs: int = 100, lr: float = 1e-3, patience: int = None,
                 scalers: Optional[Dict[str, TensorScaler]] = None, output_dir: Optional[Union[str, Path]] = "./runs",
                 optimizer: Optional[Optimizer] = None, scheduler: Optional[_LRScheduler] = None,
                 criterion: Optional[nn.Module] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The neural network.
            max_epochs (int): Maximum training epochs, defaults to 100.
            lr (float): Initial learning rate for the optimizer, defaults to 1e-3.
            patience (int): Epochs to wait before early stopping if no improvement, defaults to max_epochs.
            scalers (Optional[Dict[str, TensorScaler]]): Dictionary of scalers to save.
            output_dir (Union[str, Path]): Directory to save artifacts, defaults to "./runs".
            optimizer (Optional[Optimizer]): Optimizer instance, defaults to Adam.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler, defaults to None.
            criterion (Optional[nn.Module]): Loss function, defaults to MSELoss.
            device (str): Computation device, defaults to "cuda" or "cpu".
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scalers =  scalers
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler
        self.criterion = criterion if criterion else nn.MSELoss()

        self.max_epochs = max_epochs
        self.patience = patience if patience else max_epochs
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history: List[Dict[str, Any]] = []

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Abstract method: Calculate the loss for a single batch.
        Must be implemented by subclasses.

        Args:
            batch (Any): Data batch from DataLoader.

        Returns:
            Tensor: Scalar loss tensor (Attached to graph). Shape: (1,)
        """
        raise NotImplementedError('Subclasses must implement _compute_loss.')

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        """
        Runs a single epoch of training or validation.

        Args:
            loader (DataLoader): The data loader.
            is_training (bool): Whether gradients should be computed.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train(is_training)
        losses = []

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            pbar = tqdm(loader, desc="Training" if is_training else "Validating", leave=False, dynamic_ncols=True)
            for batch in pbar:
                # move batch to device (handling lists/tuples generic logic)
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

                loss = self._compute_loss(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

                loss_val = loss.item()
                losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4e}"})

        return float(np.mean(losses))

    def _on_epoch_end(self, **kwargs) -> None:
        """
        Optional hook called at the end of each epoch.
        Default implementation is a no-op.
        """
        pass

    def _save_checkpoint(self, val_loss: float, is_best: bool = False, extra_state: Dict = {}) -> None:
        """
        Save the training state.
        """
        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            **extra_state
        }
        if self.scalers:
            state["scaler_state_dict"] = {k: v.state_dict() for k, v in self.scalers.items()}
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(state, self.output_dir / "ckpt.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Main training loop.
        """
        logger.info(f"start training on {sl.m}{self.device}{sl.q} with {sl.m}{self.max_epochs}{sl.q} epochs")
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch + 1
            ep_start = time.time()

            # train & validate
            train_loss = self._run_epoch(train_loader, is_training=True)
            val_loss = self._run_epoch(val_loader, is_training=False) if val_loader else None

            # call hook function
            self._on_epoch_end(val_loss)

            # scheduler step
            if self.scheduler:
                self.scheduler.step()

            # check best model
            is_best = val_loss and val_loss < self.best_loss
            if is_best:
                val_str = f" | val loss: {sl.m}{val_loss:.4e} {sl.y}(best){sl.q}"
                self.best_loss = val_loss
                patience_counter = 0
            else:
                val_str = f" | val loss: {sl.m}{val_loss:.4e}{sl.q}" if val_loss else ""
                patience_counter += 1

            # save checkpoint
            self._save_checkpoint(val_loss, is_best)

            # log info
            duration = time.time() - ep_start
            logger.info(f'epoch {sl.b}{self.current_epoch:03d}{sl.q} | time: {sl.c}{duration:.1f}s{sl.q} '
                        f'| train loss: {sl.m}{train_loss:.4e}{sl.q}{val_str}')
            self.history.append({'epoch': self.current_epoch, 'train_loss': train_loss,
                                 'val_loss': val_loss, 'lr': self.optimizer.param_groups[0]['lr']})

            # early stop
            if patience_counter >= self.patience:
                logger.info(f'early stopping triggered at epoch {sl.m}{self.current_epoch}{sl.q}')
                break

        # save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"{sl.g}training finished in {time.time() - start_time:.1f}s{sl.q}")


# ======================================================================
# 5. Concrete Implementations
# ======================================================================

class SupervisedTrainer(BaseTrainer):
    """
    Standard Trainer for mapping X -> Y (One-to-One).
    """
    def _compute_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, targets = batch
        preds = self.model(inputs)
        return self.criterion(preds, targets)

class AutoregressiveTrainer(BaseTrainer):
    """
    Trainer for autoregressive sequence generation with pushforward rollout and noise injection.

    Assume the model follows an autoregressive pattern: x_{t+1} = model(x_t, conditions).

    Key features:
    1. Multi-step pushforward: Unrolls the model for k steps during training.
    2. Noise injection: Adds Gaussian noise to inputs to improve stability.
    3. Curriculum schedule: Dynamically adjusts rollout steps and noise deviation during training.

    Configs:
    1. Default optimizer: AdamW
    2. Default scheduler: CosineAnnealingLR
    3. Default criterion: NMSE + Sobolev Gradient Penalty
    4. Default curriculum: Adapts rollout steps and noise deviation based on val loss stability
    """

    def __init__(self, model: nn.Module,
                 # optimization params
                 lr: float = 1e-3, weight_decay: float = 1e-5, max_epochs: int = 100, eta_min: float = 1e-6,
                 # curriculum params
                 max_rollout_steps: int = 5, curr_patience: int = 10, curr_sensitivity: float = 0.01,
                 noise_std_init: float = 0.05, noise_decay: float = 0.9,
                 # base params
                 **kwargs):
        """
        Args:
            model (nn.Module): The neural network.
            lr, weight_decay: AdamW parameters.
            max_epochs, eta_min: CosineAnnealingLR parameters.
            max_rollout_steps (int): Max steps for autoregressive rollout.
            curr_patience (int): Epochs of stable loss to trigger curriculum advance.
            curr_sensitivity (float): Threshold for loss improvement.
            noise_std_init (float): Initial noise injection std dev.
            noise_decay (float): Multiplier for noise when curriculum advances.
            **kwargs: Arguments passed to BaseTrainer.
        """

        # 1. Initialize BaseTrainer with defaults
        optimizer = kwargs.pop("optimizer", None)
        scheduler = kwargs.pop("scheduler", None)
        criterion = kwargs.pop("criterion", None)

        # default optimizer: AdamW
        if optimizer is None:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # default scheduler: CosineAnnealingLR
        if scheduler is None:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=eta_min
            )

        # default criterion: NMSE
        if criterion is None:
            criterion = NMSELoss()

        super().__init__(model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, **kwargs)

        # 2. Store hyperparameters
        self.max_rollout_steps = max_rollout_steps
        self.curr_patience = curr_patience
        self.curr_sensitivity = curr_sensitivity
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay

        # 3. Initialize curriculum state
        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init
        self.stable_epochs_counter = 0
        self.prev_val_loss = float("inf")

    def _update_curriculum(self, val_loss: float) -> None:
        """
        Update curriculum state based on validation loss trajectory.
        Strategy: If loss stablizes, we increase difficulty (steps) and decrease assistance (noise).
        """

        # calculate relative improvement
        if self.prev_val_loss == float("inf") or self.prev_val_loss < 1e-9:
            rel_improv = 0.0
        else:
            rel_improv = self.prev_val_loss - val_loss / self.prev_val_loss

        self.prev_val_loss = val_loss

        is_stable = rel_improv < self.curr_sensitivity and val_loss < 1.0
        if is_stable:
            self.stable_epochs_counter += 1
        else:
            self.stable_epochs_counter = 0

        # trigger advance
        if self.stable_epochs_counter >= self.curr_patience:
            if self.current_rollout_steps < self.max_rollout_steps:
                self.current_rollout_steps += 1
                self.current_noise_std *= self.noise_decay
                self.stable_epochs_counter = 0
                self.prev_val_loss = float("inf")
                logger.info(f"{sl.y}curriculum update:{sl.q} "
                            f"steps = {sl.m}{self.current_rollout_steps}{sl.q}, "
                            f"noise = {sl.m}{self.current_noise_std:.4f}{sl.q}")

    def _on_epoch_end(self, val_loss: float) -> None:
        """
        Updates the curriculum parameters based on validation loss at the end of each epoch.

        Args:
            val_loss (float): Validation loss from the current epoch.
        """
        self._update_curriculum(val_loss)

    def _compute_loss(self, batch: Any) -> Tensor:
        """
        Computes the pushforward rollout loss with noise injection.

        Steps:
        1. parse augmented "super batch"
        2. initialize state x_0
        3. pushforward rollout:
            a. inject noise: tilde_x_t = x_t + epsilon
            b. predict: hat_x_t+1 = f(tilde_x_t)
            c. compute loss: L_t = loss(hat_x_t+1, gt_x_t+1)
            d. update state: x_t+1 = hat_x_t+1

        Args:
        - batch (Any): Data batch containing sequence and optionally coords.
                       Sequence shape: (batch_size * win_size, win_len, num_nodes, num_channels)
                       Coordinates shape: (batch_size * win_size, num_nodes, spatial_dim)

        Returns:
        - Tensor: Scalar loss averaged over rollout steps. Shape (1,)
        """

        # 1. parse augmented "super batch"
        seq_window, coords_window = batch

        # 2. intialize state x_0
        input_state = seq_window[:, 0]  # t = 0
        loss = torch.tensor(0.0, device=self.device)

        # 3. pushforward rollout
        for t in range(self.current_rollout_steps):
            # a. inject noise
            if self.model.training and self.current_noise_std > 1e-6:
                input_state = input_state + torch.rand_like(input_state) * self.current_noise_std

            # b. predict
            if coords_window is not None:
                pred_state = self.model(input_state, coords_window)
            else:
                pred_state = self.model(input_state)

            # c. compute step loss
            target_state = seq_window[:, t + 1]
            loss += self.criterion(pred_state, target_state)

            # d. update state
            input_state = pred_state

        return loss / self.current_rollout_steps
