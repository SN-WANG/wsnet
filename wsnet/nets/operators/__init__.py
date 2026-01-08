# wsnet/nets/operators/__init__.py
"""
Subpackage for neural operator surrogate models.
"""

from .deeponet import DeepONet
from .geofno import GeoFNO

__all__ = [
    'DeepONet',
    'GeoFNO'
]
