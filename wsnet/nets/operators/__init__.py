# wsnet/nets/operators/__init__.py
"""
Subpackage for neural operator surrogate models.
"""

from .DeepONet import DeepONet
from .GeoFNO import GeoFNO

__all__ = [
    'DeepONet',
    'GeoFNO'
]
