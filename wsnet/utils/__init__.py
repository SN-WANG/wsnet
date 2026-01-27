# wsnet/utils/__init__.py
"""
wsnet.utils: Workflow utilities for deep learning and surrogate modeling.
Includes:
    Deep Learning Engine (engine.py),
    Design of Experiments (doe.py),
    ANSYS Fluent Simulation Sequence Dataset Module (flow_data.py),
    Flow Sequence Visualization Module (flow_vis.py),
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

# Hoist from doe (Design of Experiments)
from .doe import lhs_design

# Hoist from flow_data (ANSYS Fluent Simulation Sequence Dataset Module),
from .flow_data import FlowData

# Hoist from flow_vis (Flow Sequence Visualization Module)
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

    # Design of Experiments
    "lhs_design"

    # ANSYS Fluent Simulation Sequence Dataset Module
    "FlowData"

    # Flow Sequence Visualization Module
    "CFDAnimation"
]
