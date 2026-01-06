# wsnet/nets/__init__.py
"""
wsnet.nets: A unified toolbox for functional approximation.
Organized into Surfaces, Baselines, and Operators.
"""

# Hoist from Surfaces (Response Surface Surrogate Models)
from .surfaces import (
    train_prs_model, test_prs_model,
    train_rbf_model, test_rbf_model,
    train_krg_model, test_krg_model,
    reg_constant_term, reg_linear_term, reg_quadratic_term,
    kernel_exponential, kernel_exponential_general, kernel_gaussian,
    kernel_linear, kernel_spherical, kernel_cubic, kernel_spline,
    train_svr_model, test_svr_model
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

__version__ = '0.1.0'
__author__ = 'Shengning Wang (王晟宁)'
__email__ = 'snwang2023@163.com'


__all__ = [
    # Surfaces
    'train_prs_model', 'test_prs_model',
    'train_rbf_model', 'test_rbf_model',
    'train_krg_model', 'test_krg_model',
    'reg_constant_term', 'reg_linear_term', 'reg_quadratic_term',
    'kernel_exponential', 'kernel_exponential_general', 'kernel_gaussian',
    'kernel_linear', 'kernel_spherical', 'kernel_cubic', 'kernel_spline',
    'train_svr_model', 'test_svr_model',

    # Baselines
    'MLP',

    # Operators
    'DeepONet',
    'GeoFNO'
]
