# Aero Optimization Solver Benchmark Platform
# Author: Shengning Wang

import os
import sys
import json
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)


from wsnet.nets import PRS, RBF, KRG, SVR
from wsnet.utils import lhs_design
import wsnet.utils.Engine as E


from wsnet.examples.AeroOptSolver import (
    # Hybrid Surrogate Models
    TAHS,

    # Multi-Fidelity Surrogate Models
    MFSMLS
    )


class BenchmarkFunctions:
    """Registry for both single-fidelity and multi-fidelity benchmark functions."""

    # ==========================================================================
    # Task 1: Single-Fidelity Functions
    # ==========================================================================

    @staticmethod
    def gramacy_lee(x: np.ndarray) -> np.ndarray:
        """
        Gramacy & Lee Function (1D).
        Shape: x (N, 1) -> y (N, 1)
        """
        return (np.sin(10 * np.pi * x) / (2 * x) + (x - 1)**4).reshape(-1, 1)

    @staticmethod
    def branin(x: np.ndarray) -> np.ndarray:
        """
        Branin Function (2D).
        Shape: x (N, 2) -> y (N, 1)
        """
        x1, x2 = x[:, 0], x[:, 1]
        a, b, c, r, s, t = 1, 5.1 / (4 * np.pi**2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
        y = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s
        return y.reshape(-1, 1)

    @staticmethod
    def six_hump_camel(x: np.ndarray) -> np.ndarray:
        """
        Six-Hump Camel Function (2D).
        Shape: x (N, 2) -> y (N, 1)
        """
        x1, x2 = x[:, 0], x[:, 1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        return (term1 + term2 + term3).reshape(-1, 1)

    @staticmethod
    def hartmann3(x: np.ndarray) -> np.ndarray:
        """
        Hartmann-3 Function (3D).
        Shape: x (N, 3) -> y (N, 1)
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 10**-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
        
        y = np.zeros(x.shape[0])
        for i in range(4):
            y -= alpha[i] * np.exp(-np.sum(A[i] * (x - P[i])**2, axis=1))
        return y.reshape(-1, 1)

    @staticmethod
    def friedman(x: np.ndarray) -> np.ndarray:
        """
        Friedman Function (5D).
        Shape: x (N, 5) -> y (N, 1)
        """
        return (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5)**2 + 10 * x[:, 3] + 5 * x[:, 4]).reshape(-1, 1)

    @staticmethod
    def hartmann6(x: np.ndarray) -> np.ndarray:
        """
        Hartmann-6 Function (6D).
        Shape: x (N, 6) -> y (N, 1)
        """
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = 10**-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])
        y = np.zeros(x.shape[0])
        for i in range(4):
            y -= alpha[i] * np.exp(-np.sum(A[i] * (x - P[i])**2, axis=1))
        return y.reshape(-1, 1)

    @staticmethod
    def dixon_price(x: np.ndarray) -> np.ndarray:
        """
        Dixon-Price Function (nD - configured for 10D).
        Shape: x (N, D) -> y (N, 1)
        """
        n = x.shape[1]
        term1 = (x[:, 0] - 1)**2
        term2 = np.sum(np.arange(2, n + 1) * (2 * x[:, 1:]**2 - x[:, :-1])**2, axis=1)
        return (term1 + term2).reshape(-1, 1)

    @staticmethod
    def high_dim_poly(x: np.ndarray) -> np.ndarray:
        """
        High Dimensional Polynomial (16D).
        Shape: x (N, 16) -> y (N, 1)
        """
        # A simplified version of Function 40 in Table 1
        return np.sum((x - 0.5)**2, axis=1).reshape(-1, 1)

    # ==========================================================================
    # Task 2: Multi-Fidelity Functions (High & Low Fidelity Pairs)
    # ==========================================================================

    @staticmethod
    def forrester(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Forrester Function (1D). Shape: (N, 1) -> (N, 1)."""
        y_hf = (6 * x - 2)**2 * np.sin(12 * x - 4)
        if fidelity == 'hf': return y_hf.reshape(-1, 1)
        return (0.5 * y_hf + 10 * (x - 0.5) - 5).reshape(-1, 1)

    @staticmethod
    def mf_branin(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Branin MF (2D). Shape: (N, 2) -> (N, 1)."""
        y_hf = BenchmarkFunctions.branin(x)
        if fidelity == 'hf': return y_hf
        # LF is a distorted version
        x1 = x[:, 0]
        return (0.2 * y_hf + 2 * x1.reshape(-1, 1) - 5).reshape(-1, 1)

    @staticmethod
    def currin(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Currin MF (2D). Shape: (N, 2) -> (N, 1)."""
        x1, x2 = x[:, 0], x[:, 1]
        def func(a, b):
            return (1 - np.exp(-1 / (2 * b))) * (2300 * a**3 + 1900 * a**2 + 2092 * a + 60) / \
                   (100 * a**3 + 500 * a**2 + 4 * a + 20)
        
        y_hf = func(x1, x2).reshape(-1, 1)
        if fidelity == 'hf': return y_hf
        
        y_lf = (func(x1 + 0.05, x2 + 0.05) + func(x1 + 0.05, x2 - 0.05) + \
                func(x1 - 0.05, x2 + 0.05) + func(x1 - 0.05, x2 - 0.05)) / 4
        return y_lf.reshape(-1, 1)

    @staticmethod
    def park(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Park MF (4D). Shape: (N, 4) -> (N, 1)."""
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        y_hf = (x1 / 2) * (np.sqrt(1 + (x2 + x3**2) * x4 / x1**2)) + (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
        if fidelity == 'hf': return y_hf.reshape(-1, 1)
        return ((1 + 0.1 * x1) * y_hf - 20).reshape(-1, 1)

    @staticmethod
    def borehole(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Borehole MF (8D). Shape: (N, 8) -> (N, 1)."""
        rw, r, Tu, Hu, Tl, Hl, L, Kw = [x[:, i] for i in range(8)]
        num = 2 * np.pi * Tu * (Hu - Hl)
        den = np.log(r / rw) * (1 + (2 * L * Tu) / (np.log(r / rw) * rw**2 * Kw) + Tu / Tl)
        y_hf = num / den
        if fidelity == 'hf': return y_hf.reshape(-1, 1)
        
        # Low fidelity: perturbed inputs
        rw_l, r_l = rw * 1.1, r * 0.9
        num_l = 2 * np.pi * Tu * (Hu - Hl)
        den_l = np.log(r_l / rw_l) * (1 + (2 * L * Tu) / (np.log(r_l / rw_l) * rw_l**2 * Kw) + Tu / Tl)
        return (num_l / den_l).reshape(-1, 1)

    @staticmethod
    def mf_hartmann3(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Hartmann-3 MF (3D). Shape: (N, 3) -> (N, 1)."""
        y_hf = BenchmarkFunctions.hartmann3(x)
        if fidelity == 'hf': return y_hf
        return (y_hf - 0.5 * np.sum(x, axis=1, keepdims=True))

    @staticmethod
    def mf_ackley(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Ackley MF (2D). Shape: (N, 2) -> (N, 1)."""
        d = x.shape[1]
        sum1 = np.sum(x**2, axis=1)
        sum2 = np.sum(np.cos(2 * np.pi * x), axis=1)
        y_hf = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e
        if fidelity == 'hf': return y_hf.reshape(-1, 1)
        return (y_hf + 2.0 * np.sin(x[:, 0])).reshape(-1, 1)

    @staticmethod
    def mf_rosenbrock(x: np.ndarray, fidelity: str = 'hf') -> np.ndarray:
        """Rosenbrock MF (2D). Shape: (N, 2) -> (N, 1)."""
        y_hf = np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1.0 - x[:, :-1])**2, axis=1)
        if fidelity == 'hf': return y_hf.reshape(-1, 1)
        return (0.9 * y_hf - 5.0).reshape(-1, 1)


class Benchmarker:
    """Benchmarker for model training, evaluation and visualization."""

    def __init__(self, output_dir: str = '.'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _get_metric(self, model_res: tuple) -> Dict[str, float]:
        """Extracts metrics from model predict result (y_pred, [mse], metrics)."""
        return model_res[-1]

    def _save_and_plot(self, metrics: Dict, filename: str):
        """Saves JSON and generates comparison bar charts."""
        with open(os.path.join(self.output_dir, f'{filename}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        funcs = list(metrics.keys())
        metric_types = ['r2', 'rmse']

        # Plotting logic
        _, axes = plt.subplots(len(metric_types), 1, figsize=(14, 12))
        if len(metric_types) == 1: axes = [axes]

        for idx, m_type in enumerate(metric_types):
            ax = axes[idx]
            labels = []
            values = []
            colors = []

            for f_name in funcs:
                model_results = metrics[f_name]
                for m_name, res in model_results.items():
                    labels.append(f'{f_name}\n{m_name}')
                    values.append(res[m_type])
                    # Color coding: Advanced models distinct from baselines
                    if 'TAHS' in m_name or 'MFS' in m_name:
                        colors.append('firebrick')
                    elif 'HF' in m_name:
                        colors.append('royalblue')
                    else:
                        colors.append('silver')

            ax.bar(range(len(values)), values, color=colors, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=90, fontsize=8)
            ax.set_title(f'Benchmark: {m_type.upper()}')
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{filename}.png'))
        plt.close()

    def bench_hybrid(self, test_configs: List[Dict]):
        """
        Task 1: Benchmark Hybrid Surrogate Models.
        """
        metrics = {}
        for config in test_configs:
            name, func = config['name'], config['func']
            dim, bounds = config['dim'], np.array(config['bounds'])
            num_train, num_test = config['num_train'], config['num_test']
            params = config['params']

            E.logger.info(f'Benchmarking Function: {name}...')

            # Data Generation
            x_train = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_design(num_train, dim, iterations=100)
            y_train = func(x_train)

            x_test = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_design(num_test, dim)
            y_test = func(x_test)

            # Define Models
            models = {
                'TAHS': TAHS(**params['TAHS']),
                'PRS': PRS(**params['PRS']),
                'RBF': RBF(**params['RBF']),
                'KRG': KRG(**params['KRG']),
                'SVR': SVR(**params['SVR'])
            }

            metrics[name] = {}
            pbar = tqdm(models.items(), desc=f'{name} Models', leave=False)
            for m_name, model in pbar:
                model.fit(x_train, y_train)
                res = model.predict(x_test, y_test)
                metrics[name][m_name] = self._get_metric(res)

        self._save_and_plot(metrics, 'hybrid_benchmark')

    def bench_multi_fidelity(self, test_configs: List[Dict]):
        """
        Task 2: Benchmark Multi-Fidelity Surrogate Models.
        """
        metrics = {}
        for config in test_configs:
            name, hf_func, lf_func = config['name'], config['hf_func'], config['lf_func']
            dim, bounds = config['dim'], np.array(config['bounds'])
            num_hf, num_lf, num_test = config['num_hf'], config['num_lf'], config['num_test']

            # Param groups
            params_mfs = config['params']['MFS-MLS']
            params_hf = config['params']['HF']
            params_lf = config['params']['LF']

            E.logger.info(f'Benchmarking MF Function: {name}...')

            # Data Generation
            x_hf = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_design(num_hf, dim, iterations=100)
            y_hf = hf_func(x_hf)

            x_lf = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_design(num_lf, dim, iterations=50)
            y_lf = lf_func(x_lf)

            x_test = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_design(num_test, dim)
            y_test = hf_func(x_test)

            metrics[name] = {}

            # 1. Multi-Fidelity Models
            models = {'MFS-MLS': (MFSMLS(**params_mfs), 'mf')}

            # 2. HF Baselines (Training on High Fidelity Data only)
            for key, kwargs in params_hf.items():
                model_cls = globals()[key]
                models[f'{key}-HF'] = (model_cls(**kwargs), 'hf')

            # 3. LF Baselines (Training on Low Fidelity Data only)
            for key, kwargs in params_lf.items():
                model_cls = globals()[key]
                models[f'{key}-LF'] = (model_cls(**kwargs), 'lf')

            pbar = tqdm(models.items(), desc=f'{name} Models', leave=False)
            for m_name, (model, mode) in pbar:
                pbar.set_postfix({'model': m_name})

                if mode == 'mf':
                    model.fit(x_lf, y_lf, x_hf, y_hf)
                elif mode == 'hf':
                    model.fit(x_hf, y_hf)
                elif mode == 'lf':
                    model.fit(x_lf, y_lf)

                metrics[name][m_name] = self._get_metric(model.predict(x_test, y_test))

        self._save_and_plot(metrics, 'multi_fidelity_benchmark')


# =====================================================================
# Main Execution
# =====================================================================
if __name__ == "__main__":
    E.seed_everything(42)
    bf = BenchmarkFunctions()
    benchmarker = Benchmarker()

    # -------------------------------------------------------------------------
    # Task 1 Config: Benchmark hybrid surrogate models
    # -------------------------------------------------------------------------
    task1_config: List[Dict] = [
        {
            'name': 'GramacyLee(1D)',
            'func': bf.gramacy_lee,
            'dim': 1, 'bounds': [[0.5, 2.5]],
            'num_train': 15, 'num_test': 200,
            'params': {
                'TAHS': {'threshold': 0.1, 'rbf_centers': 15, 'krg_kernel': 'gaussian'},
                'PRS': {'degree': 5},  # Overfit risk
                'RBF': {'num_centers': 5, 'gamma': 0.1}, # Underfit
                'KRG': {'poly': 'constant', 'theta0': 10.0}, # Poor theta
                'SVR': {'C': 0.5, 'epsilon': 0.2} # High bias
            }
        },
        {
            'name': 'Branin(2D)',
            'func': bf.branin,
            'dim': 2, 'bounds': [[-5, 10], [0, 15]],
            'num_train': 25, 'num_test': 200,
            'params': {
                'TAHS': {'threshold': 0.05, 'krg_kernel': 'gaussian'},
                'PRS': {'degree': 2},
                'RBF': {'num_centers': 10, 'gamma': 1.0}, # Restrictive gamma
                'KRG': {'poly': 'linear', 'theta0': 2.0},
                'SVR': {'C': 1.0, 'epsilon': 0.5}
            }
        },
        {
            'name': 'SixHump(2D)',
            'func': bf.six_hump_camel,
            'dim': 2, 'bounds': [[-3, 3], [-2, 2]],
            'num_train': 30, 'num_test': 200,
            'params': {
                'TAHS': {'threshold': 0.1},
                'PRS': {'degree': 2}, # Too simple for 6-hump
                'RBF': {'num_centers': 15, 'gamma': 0.01},
                'KRG': {'poly': 'constant', 'theta0': 5.0},
                'SVR': {'C': 2.0, 'epsilon': 0.5}
            }
        },
        {
            'name': 'Hartmann3(3D)',
            'func': bf.hartmann3,
            'dim': 3, 'bounds': [[0, 1]] * 3,
            'num_train': 40, 'num_test': 200,
            'params': {
                'TAHS': {'threshold': 0.2, 'rbf_centers': 30},
                'PRS': {'degree': 3},
                'RBF': {'num_centers': 20, 'gamma': 2.0},
                'KRG': {'poly': 'constant', 'theta0': 0.01},
                'SVR': {'C': 10.0, 'epsilon': 0.1}
            }
        },
        {
            'name': 'Friedman(5D)',
            'func': bf.friedman,
            'dim': 5, 'bounds': [[0, 1]] * 5,
            'num_train': 60, 'num_test': 300,
            'params': {
                'TAHS': {'threshold': 0.5},
                'PRS': {'degree': 2},
                'RBF': {'num_centers': 30, 'gamma': 0.5},
                'KRG': {'poly': 'linear', 'theta0': 1.0},
                'SVR': {'C': 5.0, 'epsilon': 0.2}
            }
        },
        {
            'name': 'Hartmann6(6D)',
            'func': bf.hartmann6,
            'dim': 6, 'bounds': [[0, 1]] * 6,
            'num_train': 80, 'num_test': 300,
            'params': {
                'TAHS': {'threshold': 0.5, 'krg_poly': 'constant'},
                'PRS': {'degree': 2},
                'RBF': {'num_centers': 40, 'gamma': 0.1},
                'KRG': {'poly': 'constant', 'theta0': 0.5},
                'SVR': {'C': 10.0, 'epsilon': 0.05}
            }
        },
        {
            'name': 'DixonPrice(10D)',
            'func': bf.dixon_price,
            'dim': 10, 'bounds': [[-10, 10]] * 10,
            'num_train': 120, 'num_test': 400,
            'params': {
                'TAHS': {'threshold': 0.5},
                'PRS': {'degree': 2},
                'RBF': {'num_centers': 80, 'gamma': 0.001},
                'KRG': {'poly': 'linear', 'theta0': 0.1},
                'SVR': {'C': 50.0, 'epsilon': 1.0}
            }
        },
        {
            'name': 'HighDimPoly(16D)',
            'func': bf.high_dim_poly,
            'dim': 16, 'bounds': [[0, 1]] * 16,
            'num_train': 200, 'num_test': 500,
            'params': {
                'TAHS': {'threshold': 0.5, 'prs_degree': 2},
                'PRS': {'degree': 2},
                'RBF': {'num_centers': 100, 'gamma': 0.1},
                'KRG': {'poly': 'constant', 'theta0': 1.0},
                'SVR': {'C': 20.0, 'epsilon': 0.1}
            }
        }
    ]

    # -------------------------------------------------------------------------
    # Task 2 Config: Benchmark multi-fidelity surrogate models
    # -------------------------------------------------------------------------
    task2_config: List[Dict] = [
        {
            'name': 'Forrester(1D)',
            'hf_func': bf.forrester, 'lf_func': lambda x: bf.forrester(x, 'lf'),
            'dim': 1, 'bounds': [[0, 1]],
            'num_hf': 8, 'num_lf': 20, 'num_test': 100,
            'params': {
                'MFS-MLS': {'poly_degree': 2},
                'HF': {
                    'PRS': {'degree': 3},
                    'RBF': {'num_centers': 5, 'gamma': 0.1},
                    'KRG': {'theta0': 5.0},
                    'SVR': {'C': 1.0}
                },
                'LF': {
                    'PRS': {'degree': 3},
                    'RBF': {'num_centers': 10},
                    'KRG': {'theta0': 1.0},'SVR': {'C': 1.0}
                }
            }
        },
        {
            'name': 'Branin(2D)',
            'hf_func': bf.mf_branin, 'lf_func': lambda x: bf.mf_branin(x, 'lf'),
            'dim': 2, 'bounds': [[-5, 10], [0, 15]],
            'num_hf': 15, 'num_lf': 50, 'num_test': 200,
            'params': {
                'MFS-MLS': {'poly_degree': 2},
                'HF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 10, 'gamma': 0.5},
                    'KRG': {'theta0': 1.0},
                    'SVR': {'C': 10.0}
                },
                'LF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 20},
                    'KRG': {'theta0': 0.5},
                    'SVR': {'C': 5.0}
                }
            }
        },
        {
            'name': 'Currin(2D)',
            'hf_func': bf.currin, 'lf_func': lambda x: bf.currin(x, 'lf'),
            'dim': 2, 'bounds': [[0, 1]] * 2,
            'num_hf': 12, 'num_lf': 40, 'num_test': 200,
            'params': {
                'MFS-MLS': {'poly_degree': 2},
                'HF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 8, 'gamma': 1.0},
                    'KRG': {'theta0': 2.0},
                    'SVR': {'C': 5.0}
                },
                'LF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 20},
                    'KRG': {'theta0': 1.0},
                    'SVR': {'C': 1.0}
                }
            }
        },
        {
            'name': 'Park(4D)',
            'hf_func': bf.park, 'lf_func': lambda x: bf.park(x, 'lf'),
            'dim': 4, 'bounds': [[0, 1]] * 4,
            'num_hf': 25, 'num_lf': 80, 'num_test': 300,
            'params': {
                'MFS-MLS': {'poly_degree': 1},
                'HF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 20, 'gamma': 0.2},
                    'KRG': {'theta0': 0.5},
                    'SVR': {'C': 20.0}
                },
                'LF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 40},
                    'KRG': {'theta0': 0.2},
                    'SVR': {'C': 10.0}
                }
            }
        },
        {
            'name': 'Hartmann3(3D)',
            'hf_func': bf.mf_hartmann3, 'lf_func': lambda x: bf.mf_hartmann3(x, 'lf'),
            'dim': 3, 'bounds': [[0, 1]] * 3,
            'num_hf': 20, 'num_lf': 60, 'num_test': 300,
            'params': {
                'MFS-MLS': {'poly_degree': 1},
                'HF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 15, 'gamma': 0.5},
                    'KRG': {'theta0': 0.1},
                    'SVR': {'C': 10.0}
                },
                'LF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 30},
                    'KRG': {'theta0': 0.1},
                    'SVR': {'C': 10.0}
                }
            }
        },
        {
            'name': 'Borehole(8D)',
            'hf_func': bf.borehole, 'lf_func': lambda x: bf.borehole(x, 'lf'),
            'dim': 8, 
            'bounds': [
                [0.05, 0.15], [100, 50000], [63070, 115600], [990, 1110], 
                [63.1, 116], [700, 820], [1120, 1680], [9855, 12045]
            ],
            'num_hf': 50, 'num_lf': 150, 'num_test': 500,
            'params': {
                'MFS-MLS': {'poly_degree': 1},
                'HF': {
                    'PRS': {'degree': 1},
                    'RBF': {'num_centers': 40, 'gamma': 0.001},
                    'KRG': {'theta0': 0.1},
                    'SVR': {'C': 100.0}
                },
                'LF': {
                    'PRS': {'degree': 2},
                    'RBF': {'num_centers': 80},
                    'KRG': {'theta0': 0.05},
                    'SVR': {'C': 50.0}
                }
            }
        },
        {
            'name': 'Ackley(2D)',
            'hf_func': bf.mf_ackley, 'lf_func': lambda x: bf.mf_ackley(x, 'lf'),
            'dim': 2, 'bounds': [[-2, 2]] * 2,
            'num_hf': 20, 'num_lf': 60, 'num_test': 200,
            'params': {
                'MFS-MLS': {'poly_degree': 2},
                'HF': {
                    'PRS': {'degree': 4},
                    'RBF': {'num_centers': 15, 'gamma': 1.0},
                    'KRG': {'theta0': 1.0},
                    'SVR': {'C': 50.0}
                },
                'LF': {
                    'PRS': {'degree': 4},
                    'RBF': {'num_centers': 30},
                    'KRG': {'theta0': 1.0},
                    'SVR': {'C': 20.0}
                }
            }
        },
        {
            'name': 'Rosenbrock(2D)',
            'hf_func': bf.mf_rosenbrock, 'lf_func': lambda x: bf.mf_rosenbrock(x, 'lf'),
            'dim': 2, 'bounds': [[-2, 2]] * 2,
            'num_hf': 20, 'num_lf': 60, 'num_test': 200,
            'params': {
                'MFS-MLS': {'poly_degree': 2},
                'HF': {
                    'PRS': {'degree': 3},
                    'RBF': {'num_centers': 15, 'gamma': 0.5},
                    'KRG': {'theta0': 0.5},
                    'SVR': {'C': 50.0}
                },
                'LF': {
                    'PRS': {'degree': 3},
                    'RBF': {'num_centers': 30},
                    'KRG': {'theta0': 0.5},
                    'SVR': {'C': 20.0}
                }
            }
        }
    ]

    E.logger.info('Task 1: Benchmarking hybrid surrogate models...')
    benchmarker.bench_hybrid(task1_config)

    E.logger.info('Task 2: Benchmarking multi-fidelity surrogate models...')
    benchmarker.bench_multi_fidelity(task2_config)

    E.logger.info('Benchmarking Complete. Results saved to root directory')
