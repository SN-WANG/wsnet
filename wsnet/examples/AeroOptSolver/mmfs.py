# Modified Multi-Fidelity Surrogate Model based on RBF with Adaptive Scale Factor (MMFS)
# Paper site: https://doi.org/10.1186/s10033-022-00742-z
# Paper Author: Yin Liu, Shuo Wang, Qi Zhou, Liye Lv, Wei Sun, Xueguan Song
# Code Author: Shengning Wang

import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, Tuple, Union, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wsnet.nets import RBF
import wsnet.utils.Engine as E


class MMFS:
    """
    Modified Multi-fidelity Surrogate Model based on RBF with Adaptive Scale Factor.

    Implements the modified multi-fidelity surrogate model using Radial Basis Function (RBF)
    with adaptive scale factor, supporting multi-input multi-output (MIMO) scenarios.

    Attributes:
    - adaptive_scale_params: Dictionary of parameters for adaptive scale factor calculation.
    - rbf_kernel: Type of RBF kernel.
    - rbf_epsilon: Epsilon parameter for RBF kernel.
    """