# wsnet/utils/__init__.py
"""
wsnet.utils: Engineering utilities for deep learning and surrogate modeling.
Includes Deep Learning Engine, DoE, CFD Data Parsing, and CFD Data Rendering.
"""

# Hoist from Engine (Deep Learning Engine)
from .Engine import (
    # 1. Utilities & Reproducibility
    logger, seed_everything,
    # 2. Data Processing & Normalization
    BaseDataset, TensorScaler,
    # 3. Core Engine: The Base Trainer
    BaseTrainer,
    # 4. Concrete Implementations
    SupervisedTrainer, AutoregressiveTrainer
)

# Hoist from DoE (Design of Experiments)
from .DoE import lhs_design

# Hoist from CFDataParser (CFD Data Parsing)
from .CFDParser import CFDataset, find_case_dirs

# Hoist from CFDRender (CFD Data Rendering)
from .CFDRender import CFDAnimation


__version__ = '0.1.0'
__author__ = 'Shengning Wang (王晟宁)'
__email__ = 'snwang2023@163.com'


__all__ = [
    # Engine
    # 1. Utilities & Reproducibility
    'logger', 'seed_everything',
    # 2. Data Processing & Normalization
    'BaseDataset', 'TensorScaler',
    # 3. Core Engine: The Base Trainer
    'BaseTrainer',
    # 4. Concrete Implementations
    'SupervisedTrainer', 'AutoregressiveTrainer',

    # DoE
    'lhs_design'

    # CFDataParser
    'CFDataset', 'find_case_dirs'

    # CFDataRenderer
    'CFDAnimation'
]
