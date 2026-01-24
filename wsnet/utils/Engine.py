# Deep Learning Engine
# Author: Shengning Wang

import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer, Adam, lr_scheduler
from tqdm.auto import tqdm


# ======================================================================
# 1. Utilities & Reproducibility
# ======================================================================

# Config Logging
# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%H:%M:%S',
#     level=logging.INFO
# )
# logger = logging.getLogger(__name__)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(tqdm_handler)
logger.propagate = False


def seed_everything(seed: int = 42) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
    - seed (int): The seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f'Global seed set to {seed}')


# ======================================================================
# 2. Data Processing & Normalization
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
        - x (Tensor): Data tensor. Shape: (N, ...)
        - feature_dim (int): The dimension representing features/channels.
                             Statistics are computed over all OTHER dimensions.

        Returns:
        - TensorScaler: Self instance for method chaining.
        """
        # Ensure positive index
        feature_dim = feature_dim % x.ndim

        # Aggregate over all dimensions except the feature dimension
        dims = [d for d in range(x.ndim) if d != feature_dim]

        self.mean = x.mean(dim=dims, keepdim=True)
        self.std = x.std(dim=dims, keepdim=True)

        # Handle constant values (std=0) to avoid NaN
        self.std[self.std < 1e-7] = 1.0
        self.device = x.device

        return self

    def transform(self, x: Tensor) -> Tensor:
        """
        Standardizes input tensor.

        Args:
        - x (Tensor): Data to transform. Shape matches input to fit().

        Returns:
        - Tensor: Scaled data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Scaler has not been fitted.')

        # Ensure scaler stats are on the same device as input
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Scales back the data to the original representation.

        Args:
        - x (Tensor): Scaled data.

        Returns:
        - Tensor: Original scale data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError('Scaler has not been fitted.')

        # Ensure scaler stats are on the same device as input
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
# 3. Core Engine: The Base Trainer
# ======================================================================

class BaseTrainer:
    """
    Base class encapsulating the training loop, checkpointing, and evaluation logic.
    Subclasses must implement 'compute_loss' to define specific task logic.
    """

    def __init__(self, model: nn.Module, output_dir: Optional[Union[str, Path]] = './runs',
                 optimizer: Optional[Optimizer] = None, scheduler: Optional[lr_scheduler._LRScheduler] = None,
                 criterion: Optional[nn.Module] = None, scalers: Optional[Dict[str, TensorScaler]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_epochs: int = 100, lr: float = 1e-3, patience: int = 10):
        """
        Initialize the Trainer.

        Args:
        - model (nn.Module): The neural network.
        - output_dir (Union[str, Path]): Directory to save artifacts. Defaults to './runs'
        - optimizer (Optional[Optimizer]): Optimizer instance. Defaults to Adam.
        - scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        - criterion (Optional[nn.Module]): Loss function. Default to MSELoss.
        - scalers (Optional[Dict[str, TensorScaler]]): Dictionary of scalers to save.
        - device (str): Computation device.
        - max_epochs (int): Maximum training epochs.
        - lr (float): Initial learning rate for the optimizer, default 1e-3.
        - patience (int): Epochs to wait before early stopping if no improvement.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer if optimizer else Adam(self.model.parameters(), lr=lr)
        self.scheduler = scheduler if scheduler else lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=5, min_lr=1e-6)
        self.scaler_state = {k: v.state_dict() for k, v in scalers.items()} if scalers else None

        self.max_epochs = max_epochs
        self.patience = patience
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history: List[Dict[str, Any]] = []

    def compute_loss(self, batch: Any) -> Tensor:
        """
        Abstract method: Calculate the loss for a single batch.
        Must be implemented by subclasses.

        Args:
        - batch (Any): Data batch from DataLoader.

        Returns:
        - Tensor: Scalar loss tensor (Attached to graph). Shape: (1,)
        """
        raise NotImplementedError('Subclasses must implement compute_loss.')

    def _run_epoch(self, loader: DataLoader, is_training: bool) -> float:
        """
        Runs a single epoch of training or validation.

        Args:
        - loader (DataLoader): The data loader.
        - is_training (bool): Whether gradients should be computed.

        Returns:
        - float: Average loss for the epoch.
        """
        self.model.train(is_training)
        losses = []

        context = torch.enable_grad() if is_training else torch.no_grad()

        with context:
            pbar = tqdm(loader, desc='Training' if is_training else 'Validating', leave=False)
            for batch in pbar:
                # Move batch to device (handling lists/tuples generic logic)
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, Tensor) else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

                loss = self.compute_loss(batch)

                if is_training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # Gradient Clipping
                    self.optimizer.step()

                loss_val = loss.item()
                losses.append(loss_val)
                pbar.set_postfix({'loss': f'{loss_val:.4e}'})

        return float(np.mean(losses))

    def save_checkpoint(self, val_loss: float, is_best: bool = False) -> None:
        """Save the training state."""
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'scaler_state': getattr(self, 'scaler_state', None)
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save generic 'last' checkpoint
        torch.save(state, self.output_dir / f'ckpt.pt')

        if is_best:
            torch.save(state, self.output_dir / 'model.pt')
            logger.info(f'New best model saved at epoch {self.current_epoch} with loss {val_loss:.4e}')

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Resumes training from a checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Checkpoint not found at {path}')

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('val_loss', float('inf'))
        logger.info(f'Resumed from epoch {self.current_epoch}, best loss: {self.best_loss:.4e}')

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Main training loop.
        """
        logger.info(f'Starting training on {self.device} for {self.max_epochs} epochs.')
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch + 1
            ep_start = time.time()

            # Training
            train_loss = self._run_epoch(train_loader, is_training=True)

            # Validation
            val_loss = None
            val_str = ''
            if val_loader:
                val_loss = self._run_epoch(val_loader, is_training=False)
                val_str = f' | Val Loss: {val_loss:.4e}'

                if self.scheduler:
                    if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Checkpointing logic
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(val_loss, is_best=True)
                else:
                    patience_counter += 1
                    self.save_checkpoint(val_loss, is_best=False)

                if patience_counter >= self.patience:
                    logger.info(f'Early stopping triggered at epoch {self.current_epoch}')
                    break

            # Logging
            duration = time.time() - ep_start
            logger.info(f'Epoch {self.current_epoch:03d} | Time: {duration:.1f}s '
                        f'| Train Loss: {train_loss:.4e}{val_str}')

            self.history.append({'epoch': self.current_epoch, 'train_loss': train_loss,
                                 'val_loss': val_loss, 'lr': self.optimizer.param_groups[0]['lr']})

        # Save History
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f'Training finished in {time.time() - start_time:.1f}s.')


# ======================================================================
# 4. Concrete Implementations
# ======================================================================

class SupervisedTrainer(BaseTrainer):
    """
    Standard Trainer for mapping X -> Y (One-to-One).
    """
    def compute_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, targets = batch
        preds = self.model(inputs)
        return self.criterion(preds, targets)

class AutoregressiveTrainer(BaseTrainer):
    """
    Trainer for autoregressive sequence generation with pushforward rollout and noise injection.

    Assume the model follows an autoregressive pattern: x_{t+1} = model(x_t, conditions).

    Key features:
    1. Multi-step pushforward: Unrolls the model for k steps during training.
    2. Curriculum schedule: Dynamically adjusts rollout length during training.
    3. Noise injection: Adds Gaussian noise to inputs during rollout to improve stability.
    4. Sobolev Loss: Adds spatial gradient penalty.
    """

    def __init__(self, model: nn.Module, pushforward_steps: int = 5,
                 curriculum_schedule: Optional[Callable[[int], int]] = None,
                 noise_std: float = 0.0, sobolev_beta: float = 0.1, **kwargs):
        """
        Args:
        - model (nn.Module): The neural network.
        - pushforward_steps (int): The number of autoregressive steps to unroll during training.
                                   If curriculum_schedule is provided, this is the maximum steps.
        - curriculum_schedule (Optional[Callable[[int], int]]): A function mapping epoch -> steps.
        - noise_std (float): Standard deviation of Gaussian noise added to inputs during training.
        - sobolev_beta (float): Weight for the spatial gradient penalty.
        - **kwargs: Arguments passed to BaseTrainer (optimizer, criterion, etc.).
        """
        super().__init__(model, **kwargs)
        self.pushforward_steps = pushforward_steps
        self.curriculum_schedule = curriculum_schedule
        self.noise_std = noise_std
        self.sobolev_beta = sobolev_beta

    def _get_current_steps(self) -> int:
        """Determines the number of rollout steps for the current epoch."""
        if self.curriculum_schedule:
            steps = self.curriculum_schedule(self.current_epoch)
            return max(1, min(steps, self.pushforward_steps))
        return self.pushforward_steps

    def compute_loss(self, batch: Any) -> Tensor:
        """
        Computes the pushforward rollout loss with noise injection.

        Steps:
        1. Initialize state x_0
        2. For t = 0 to k:
            a. Inject noise: tilde_x_t = x_t + epsilon
            b. Predict: hat_x_t+1 = f(tilde_x_t)
            c. Compute Sobolev loss: L_t = ||hat_x_t+1 - gt_x_t+1||^2 + beta * ||grad(hat_x_t+1) - grad(gt_x_t+1)||^2
            d. Update state: x_t+1 = hat_x_t+1

        Args:
        - batch (Any): Data batch containing sequence and optionally coords.
                       Sequence shape: (batch_size * win_size, win_len, num_nodes, num_channels)
                       Coordinates shape: (batch_size * win_size, num_nodes, spatial_dim)

        Returns:
        - Tensor: Scalar loss averaged over rollout steps. Shape (1,)
        """
        # 1. Determine rollout horizon (Curriculum schedule)
        rollout_steps = self._get_current_steps()

        # 2. Parse the "super batch"
        # seq_window shape: (batch_size * win_size, win_len, num_nodes, num_channels)
        # coords_window shape: (batch_size * win_size, num_nodes, spatial_dim)
        seq_window, coords_window = batch

        # 3. Autoregressive Unrolling
        current_state = seq_window[:, 0]  # t = 0
        loss = torch.tensor(0.0, device=self.device)

        for t in range(rollout_steps):
            # A. Noise injection
            if self.model.training and self.noise_std > 0:
                noise = torch.randn_like(current_state) * self.noise_std
                input_state = current_state + noise
            else:
                input_state = current_state

            # B. Forward pass
            if coords_window is not None:
                pred_state = self.model(input_state, coords_window)
            else:
                pred_state = self.model(input_state)

            # C. Compute step Sobolev loss
            target_state = seq_window[:, t + 1]
            grad_pred = pred_state[:, 1:, :] - pred_state[:, :-1, :]
            grad_target = target_state[:, 1:, :] - target_state[:, :-1, :]

            primary_loss = self.criterion(pred_state, target_state)
            penalty_loss = self.criterion(grad_pred, grad_target)

            loss += (primary_loss + self.sobolev_beta * penalty_loss)

            # D. Pushforward
            current_state = pred_state

        return loss / rollout_steps


# ======================================================================
# 5. Example Usage
# ======================================================================

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for demonstration purposes.
    Includes a dedicated prediction method for inference.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Performs inference on new inputs.
        
        Args:
        - inputs (Tensor): Input tensor. Shape: (num_samples, in_dim)
        
        Returns:
        - Tensor: Predictions. Shape: (num_samples, out_dim)
        """
        self.eval()
        # Ensure input is on the correct device
        device = next(self.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            preds = self.forward(inputs)

        return preds


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Setup & Configuration
    # ------------------------------------------------------------------
    seed_everything(42)

    # ------------------------------------------------------------------
    # 2. Data Generation
    # ------------------------------------------------------------------
    # Task: Regression y = 2x + 1
    N_SAMPLES = 2000
    DIM_IN = 5
    DIM_OUT = 1

    # Generate synthetic data
    X_raw = torch.randn(N_SAMPLES, DIM_IN)
    # Let's say target depends strictly on the first feature for simplicity
    weights = torch.tensor([2.0] + [0.0] * (DIM_IN - 1)).unsqueeze(1) # (5, 1)
    Y_raw = X_raw @ weights + 1.0

    # ------------------------------------------------------------------
    # 3. Data Splitting (Train: 70%, Val: 15%, Test: 15%)
    # ------------------------------------------------------------------
    n_train = int(0.7 * N_SAMPLES)
    n_val = int(0.15 * N_SAMPLES)
    n_test = N_SAMPLES - n_train - n_val

    # Create indices and split
    indices = torch.randperm(N_SAMPLES)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, Y_train = X_raw[train_idx], Y_raw[train_idx]
    X_val, Y_val = X_raw[val_idx], Y_raw[val_idx]
    X_test, Y_test = X_raw[test_idx], Y_raw[test_idx]

    logger.info(f'Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}')

    # ------------------------------------------------------------------
    # 4. Data Preprocessing (Normalization)
    # ------------------------------------------------------------------
    # It is industry standard to fit scalers ONLY on training data
    input_scaler = TensorScaler().fit(X_train)
    target_scaler = TensorScaler().fit(Y_train)

    # Transform all sets
    X_train_s = input_scaler.transform(X_train)
    Y_train_s = target_scaler.transform(Y_train)

    X_val_s = input_scaler.transform(X_val)
    Y_val_s = target_scaler.transform(Y_val)

    scalers = {'input': input_scaler, 'target': target_scaler}

    # Create Datasets and Loaders
    train_loader = DataLoader(BaseDataset(X_train_s, Y_train_s), batch_size=64, shuffle=True)
    val_loader = DataLoader(BaseDataset(X_val_s, Y_val_s), batch_size=64, shuffle=False)

    # ------------------------------------------------------------------
    # 5. Model Initialization
    # ------------------------------------------------------------------
    model = SimpleMLP(in_dim=DIM_IN, out_dim=DIM_OUT, hidden_dim=128)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model has {num_params} parameters')

    # ------------------------------------------------------------------
    # 6. Training Loop
    # ------------------------------------------------------------------
    trainer = SupervisedTrainer(model=model, scalers=scalers, max_epochs=500, lr=1e-3, patience=10)

    logger.info('\n--- Starting Training ---')
    trainer.fit(train_loader, val_loader)

    # ------------------------------------------------------------------
    # 7. Testing & Evaluation
    # ------------------------------------------------------------------
    logger.info('\n--- Starting Evaluation ---')

    # Load trained model
    checkpoint = torch.load('./runs/model.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    input_scaler = TensorScaler()
    input_scaler.load_state_dict(checkpoint['scaler_state']['input'])
    target_scaler = TensorScaler()
    target_scaler.load_state_dict(checkpoint['scaler_state']['target'])

    X_test_s = input_scaler.transform(X_test)

    # Inference using the custom predict() method
    Y_preds_s = model.predict(X_test_s)

    # Inverse transform predictions to original scale
    Y_preds = target_scaler.inverse_transform(Y_preds_s)

    # Move to CPU for metric calculation
    Y_preds_cpu = Y_preds.cpu()
    Y_test_cpu = Y_test.cpu()

    # Calculate Metrics (MSE & MAE)
    mse = torch.mean((Y_preds_cpu - Y_test_cpu) ** 2).item()
    mae = torch.mean(torch.abs(Y_preds_cpu - Y_test_cpu)).item()

    logger.info(f'Test Set MSE: {mse:.4f}')
    logger.info(f'Test Set MAE: {mae:.4f}')

    # Visual Check (First 3 samples)
    logger.info('\nVisual Check (First 3 samples):')
    logger.info(f'Truth: {Y_test_cpu[:3].numpy().flatten()}')
    logger.info(f'Preds: {Y_preds_cpu[:3].numpy().flatten()}')
