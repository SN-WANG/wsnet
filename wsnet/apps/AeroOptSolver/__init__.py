# wsnet/apps/AeroOptSolver/__init__.py
"""
wsnet.apps.AeroOptSolver: An Aero Optimization Solver Platform.
"""

# Hoist from Hybrid Surrogate Models
from .t_ahs import TAHS

# Hoist from Multi-Fidelity Surrogate Models
from .mfs_mls import MFSMLS


__all__ = [
    # Hybrid Surrogate Models
    'TAHS',

    # Multi-Fidelity Surrogate Models
    'MFSMLS'
]
