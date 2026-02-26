# Rollout Trainer with Pushforward Rollout and Curriculum Learning
# Author: Shengning Wang

import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any

from wsnet.training.base_trainer import BaseTrainer
from wsnet.training.base_criterion import NMSECriterion
from wsnet.training.flow_criterion import FlowCriterion
from wsnet.utils.hue_logger import hue, logger


class RolloutTrainer(BaseTrainer):
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
    3. Default criterion: NMSE + Physics Informed Penalty
    4. Default curriculum: Adapts rollout steps and noise deviation based on val loss stability
    """

    def __init__(self, model: nn.Module,
                 # optimization params
                 lr: float = 1e-3, max_epochs: int = 500,
                 weight_decay: float = 1e-5, eta_min: float = 1e-6,
                 # curriculum params
                 max_rollout_steps: int = 5, rollout_patience: int = 10,
                 noise_std_init: float = 0.05, noise_decay: float = 0.9,
                 # physics params
                 use_physics_loss: bool = False,
                 lambda_physics: float = 0.1,
                 lambda_mass: float = 1.0, lambda_momentum: float = 1.0, lambda_energy: float = 1.0,
                 latent_grid_size: Any = None,
                 # base params
                 **kwargs):
        """
        Args:
            model (nn.Module): The neural network.
            lr, weight_decay: AdamW parameters.
            max_epochs, eta_min: CosineAnnealingLR parameters.
            max_rollout_steps (int): Max steps for autoregressive rollout.
            rollout_patience (int): Epochs of stable loss to trigger curriculum advance.
            noise_std_init (float): Initial noise injection std dev.
            noise_decay (float): Multiplier for noise when curriculum advances.
            use_physics_loss (bool): Whether to use physics-informed loss.
            lambda_physics (float): Weight of physics loss vs data loss.
            lambda_mass (float): Sub-weight for mass conservation residual.
            lambda_momentum (float): Sub-weight for momentum conservation residual.
            lambda_energy (float): Sub-weight for energy conservation residual.
            latent_grid_size: [L1, L2] for FD grid in physics loss.
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
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

        # default criterion: pure data-driven or physics-informed
        if criterion is None:
            if use_physics_loss:
                criterion = FlowCriterion(
                    lambda_physics=lambda_physics,
                    lambda_mass=lambda_mass,
                    lambda_momentum=lambda_momentum,
                    lambda_energy=lambda_energy
                )
            else:
                criterion = NMSECriterion()

        super().__init__(model, lr=lr, max_epochs=max_epochs,
                         optimizer=optimizer, scheduler=scheduler, criterion=criterion, **kwargs)

        # 2. Store hyperparameters
        self.max_rollout_steps = max_rollout_steps
        self.rollout_patience = rollout_patience
        self.noise_std_init = noise_std_init
        self.noise_decay = noise_decay
        self.latent_grid_size = latent_grid_size

        # 3. Initialize curriculum state
        self.rollout_counter = 0
        self.log_update_info = False
        self.current_rollout_steps = 1
        self.current_noise_std = noise_std_init

    def _update_curriculum(self) -> None:
        """
        Update curriculum state.
        Strategy: If loss stablizes, we increase difficulty (steps) and decrease assistance (noise).
        """
        self.rollout_counter += 1

        if self.rollout_counter >= self.rollout_patience:
            if self.current_rollout_steps < self.max_rollout_steps:
                self.current_rollout_steps += 1
                self.current_noise_std *= self.noise_decay
                self.rollout_counter = 0
                self.log_update_info = True

        if self.log_update_info and self.rollout_counter == 1:
            logger.info(f"{hue.y}curriculum update:{hue.q} "
                        f"steps = {hue.m}{self.current_rollout_steps}{hue.q}, "
                        f"noise = {hue.m}{self.current_noise_std:.4f}{hue.q}")
            self.log_update_info = False

    def _on_epoch_end(self, train_loss=None, val_loss=None) -> None:
        """
        Updates the curriculum parameters based at the start of each epoch.
        """
        self._update_curriculum()

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
        seq, coords = batch

        # 2. intialize state x_0
        input_state = seq[:, 0]  # t = 0
        loss = torch.tensor(0.0, device=self.device)

        # 3. pushforward rollout
        for t in range(self.current_rollout_steps):
            # a. inject noise
            clean_input = input_state
            if self.model.training and self.current_noise_std > 1e-6:
                input_state = clean_input + torch.randn_like(clean_input) * self.current_noise_std

            # b. predict
            if coords is not None:
                if hasattr(self.model, "time_encoder"):
                    pred_state = self.model(input_state, coords, step=t)
                else:
                    pred_state = self.model(input_state, coords)
            else:
                pred_state = self.model(input_state)

            # c. compute step loss
            target_state = seq[:, t + 1]

            if isinstance(self.criterion, FlowCriterion):
                loss += self.criterion(pred_state, target_state,
                                       prev=clean_input,
                                       coords=coords,
                                       latent_grid_size=self.latent_grid_size)
            else:
                loss += self.criterion(pred_state, target_state)

            # d. update state
            input_state = pred_state

        return loss / self.current_rollout_steps
