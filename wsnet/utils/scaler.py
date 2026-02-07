# Scalers for Standardization and Normalization
# Author: Shengning Wang

import numpy as np
from typing import Optional, Dict, Literal


class BaseScaler:
    """
    Base class for NumPy-based data scalers.

    This class defines the common interface for all scalers,
    including fit, transform, and inverse_transform.

    Subclasses must implement the core logic.
    """

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "BaseScaler":
        """
        Fits the scaler statistics from data.

        Args:
            x (np.ndarray): Input data array. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            BaseScaler: Self instance for method chaining.
        """
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transforms input data using fitted statistics.

        Args:
            x (np.ndarray): Input data array. Shape must match data passed to fit().

        Returns:
            np.ndarray: Scaled data with the same shape and dtype as input.
        """
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Inversely transforms scaled data back to original space.

        Args:
            x (np.ndarray): Scaled data array. Shape must match data passed to transform().

        Returns:
            np.ndarray: Data restored to original scale.
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns scaler internal state for serialization.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing scaler parameters.
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        """
        Loads scaler internal state from dictionary.

        Args:
            state_dict (Dict[str, np.ndarray]): State dictionary.
        """
        raise NotImplementedError


# ======================================================================
# Standardization Scaler
# ======================================================================

class StandardScalerNP(BaseScaler):
    """
    NumPy implementation of standardization scaler.

    Performs channel-wise standardization:
        x_scaled = (x - mean) / std

    Statistics are computed over all dimensions except channel_dim.
    """

    def __init__(self, eps: float = 1e-7):
        """
        Initializes the StandardScalerNP.

        Args:
            eps (float): Small constant to avoid division by zero.
        """
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps
        self.channel_dim: Optional[int] = None

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "StandardScalerNP":
        """
        Computes mean and standard deviation for standardization.

        Args:
            x (np.ndarray): Input data. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            StandardScalerNP: Self instance.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim

        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        self.mean = np.mean(x, axis=reduce_dims, keepdims=True)
        self.std = np.std(x, axis=reduce_dims, keepdims=True)

        # Avoid numerical instability
        self.std[self.std < self.eps] = 1.0

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applies standardization to input data.

        Args:
            x (np.ndarray): Input data. Shape matches fit() input.

        Returns:
            np.ndarray: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restores standardized data to original scale.

        Args:
            x (np.ndarray): Standardized data. Shape matches transform() input.

        Returns:
            np.ndarray: Original scale data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")

        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "mean": self.mean,
            "std": self.std,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.channel_dim = int(state_dict["channel_dim"])


# ======================================================================
# Min-Max Normalization Scaler
# ======================================================================

class MinMaxScalerNP(BaseScaler):
    """
    NumPy implementation of min-max normalization scaler.

    Supports normalization to:
        - [0, 1]
        - [-1, 1]

    Scaling rule:
        x_scaled = a + (x - min) * (b - a) / (max - min)
    """

    def __init__(self, feature_range: Literal["unit", "bipolar"] = "unit", eps: float = 1e-7):
        """
        Initializes the MinMaxScalerNP.

        Args:
            feature_range (Literal): Target normalization range.
                - "unit" -> [0, 1]
                - "bipolar" -> [-1, 1]
            eps (float): Small constant to avoid division by zero.
        """
        self.feature_range = feature_range
        self.eps = eps
        self.data_min: Optional[np.ndarray] = None
        self.data_max: Optional[np.ndarray] = None
        self.channel_dim: Optional[int] = None

        if feature_range == "unit":
            self.a, self.b = 0.0, 1.0
        elif feature_range == "bipolar":
            self.a, self.b = -1.0, 1.0
        else:
            raise ValueError("Invalid feature_range.")

    def fit(self, x: np.ndarray, channel_dim: int = -1) -> "MinMaxScalerNP":
        """
        Computes min and max statistics for normalization.

        Args:
            x (np.ndarray): Input data. Shape: (num_samples, ..., num_channels)
            channel_dim (int): Dimension index representing channel dimension.

        Returns:
            MinMaxScalerNP: Self instance.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")

        self.channel_dim = channel_dim % x.ndim

        reduce_dims = tuple(d for d in range(x.ndim) if d != self.channel_dim)

        self.data_min = np.min(x, axis=reduce_dims, keepdims=True)
        self.data_max = np.max(x, axis=reduce_dims, keepdims=True)

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Applies min-max normalization.

        Args:
            x (np.ndarray): Input data. Shape matches fit() input.

        Returns:
            np.ndarray: Normalized data.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0

        return self.a + (x - self.data_min) * (self.b - self.a) / scale

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Restores normalized data to original scale.

        Args:
            x (np.ndarray): Normalized data. Shape matches transform() input.

        Returns:
            np.ndarray: Original scale data.
        """
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Scaler has not been fitted.")

        scale = self.data_max - self.data_min
        scale[scale < self.eps] = 1.0

        return (x - self.a) * scale / (self.b - self.a) + self.data_min

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            "data_min": self.data_min,
            "data_max": self.data_max,
            "channel_dim": np.array(self.channel_dim, dtype=np.int64),
            "feature_range": np.array(
                0 if self.feature_range == "unit" else 1, dtype=np.int64
            ),
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.data_min = state_dict["data_min"]
        self.data_max = state_dict["data_max"]
        self.channel_dim = int(state_dict["channel_dim"])
        self.feature_range = (
            "unit" if int(state_dict["feature_range"]) == 0 else "bipolar"
        )
        self.a, self.b = (0.0, 1.0) if self.feature_range == "unit" else (-1.0, 1.0)
