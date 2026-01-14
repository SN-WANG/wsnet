# wsnet/nets/__init__.py
"""
wsnet.nets: A unified toolbox for functional approximation.
Organized into Surfaces, Baselines, and Operators.
"""

# Hoist from Surfaces (Response Surface Surrogate Models)
from .surfaces import (
    PRS,
    RBF,
    KRG,
    SVR
)

# Hoist from Baselines (Standard Neural Network Surrogate Models)
from .baselines import (
    MLP
)

# Hoist from Operators (Neural Operator Surrogate Models)
from .operators import (
    DeepONet,
    GeoFNO
)

__version__ = '1.3.1'
__author__ = 'Shengning Wang (王晟宁)'
__email__ = 'snwang2023@163.com'


__all__ = [
    # Surfaces
    'PRS',
    'RBF',
    'KRG',
    'SVR',

    # Baselines
    'MLP',

    # Operators
    'DeepONet',
    'GeoFNO',
]
