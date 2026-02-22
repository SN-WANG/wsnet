# Standard Trainer for One-to-One Mapping Tasks
# Author: Shengning Wang

from torch import Tensor
from typing import Tuple

from wsnet.training.base_trainer import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard Trainer for one-to-one mapping tasks X -> Y.
    """
    def _compute_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        inputs, targets = batch
        preds = self.model(inputs)
        return self.criterion(preds, targets)
