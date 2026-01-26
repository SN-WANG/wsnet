# wsnet/utils/__init__.py
"""
wsnet.utils: Workflow utilities for deep learning and surrogate modeling.
Includes Deep Learning Engine, DoE, CFD Data Parsing, and CFD Data Rendering.
"""

# Hoist from engine (Deep Learning Engine)
from .engine import (
    # 1. Logger & Seed Setter
    sl, logger, seed_everything,
    # 2. Metrics & Loss Functions
    NMSELoss, compute_ar_metrics,
    # 3. Data Processing & Standardization
    BaseDataset, TensorScaler,
    # 4. Core Engine: The Base Trainer
    BaseTrainer,
    # 5. Concrete Implementations
    SupervisedTrainer, AutoregressiveTrainer
)

# Hoist from DoE (Design of Experiments)
from .doe import lhs_design

# Hoist from CFDataParser (CFD Data Parsing)
from .CFDParser import CFDataset, find_case_dirs

# Hoist from CFDRender (CFD Data Rendering)
from .CFDRender import CFDAnimation


__all__ = [
    # Deep Learning Engine
    # 1. Logger & Seed Setter
    "sl", "logger", "seed_everything",
    # 2. Metrics & Loss Functions
    "NMSELoss", "compute_ar_metrics",
    # 3. Data Processing & Standardization
    "BaseDataset", "TensorScaler",
    # 4. Core Engine: The Base Trainer
    "BaseTrainer",
    # 5. Concrete Implementations
    "SupervisedTrainer", "AutoregressiveTrainer",

    # DoE
    "lhs_design"

    # CFDataParser
    "CFDataset", "find_case_dirs"

    # CFDataRenderer
    "CFDAnimation"
]
