"""Coulomb force search strategy based Dragonfly optimizer (CFSSDA).

This module provides a SciPy-like optimization entry point named
`dragonfly_optimize` for constrained/unconstrained and single/multi-objective
problems. The implementation only depends on NumPy and SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult, minimize
from scipy.special import gamma


EPS = 1e-12


@dataclass
class _ObjectiveValue:
    """Container for objective evaluation."""

    scalar: float
    vector: np.ndarray


def _parse_bounds(bounds: Union[Bounds, Sequence[Tuple[float, float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """Parse bounds to two vectors."""
    if isinstance(bounds, Bounds):
        lower = np.asarray(bounds.lb, dtype=float)
        upper = np.asarray(bounds.ub, dtype=float)
    else:
        arr = np.asarray(bounds, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dim, 2)")
        lower = arr[:, 0]
        upper = arr[:, 1]

    if lower.shape != upper.shape:
        raise ValueError("lower and upper bounds must have the same shape")
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("all bounds must be finite for dragonfly_optimize")
    if np.any(upper <= lower):
        raise ValueError("each upper bound must be strictly greater than lower bound")
    return lower, upper


def _levy_flight(rng: np.random.Generator, n_dim: int, beta: float = 1.5) -> np.ndarray:
    """Generate a Levy-flight random vector with Mantegna's algorithm."""
    sigma_u = (
        gamma(1.0 + beta) * np.sin(np.pi * beta / 2.0) /
        (gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0)))
    ) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=n_dim)
    v = rng.normal(0.0, 1.0, size=n_dim)
    step = u / ((np.abs(v) + EPS) ** (1.0 / beta))
    return 0.01 * step


def _constraint_violation(
    x: np.ndarray,
    constraints: Sequence[Any],
    args: Tuple[Any, ...],
) -> float:
    """Compute aggregated non-negative constraint violation."""
    if not constraints:
        return 0.0

    violation = 0.0
    for c in constraints:
        if isinstance(c, LinearConstraint):
            values = np.atleast_1d(c.A @ x)
            lb = np.atleast_1d(c.lb).astype(float)
            ub = np.atleast_1d(c.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))
        elif isinstance(c, NonlinearConstraint):
            values = np.atleast_1d(c.fun(x))
            lb = np.atleast_1d(c.lb).astype(float)
            ub = np.atleast_1d(c.ub).astype(float)
            low_v = np.maximum(lb - values, 0.0)
            up_v = np.maximum(values - ub, 0.0)
            low_v[~np.isfinite(lb)] = 0.0
            up_v[~np.isfinite(ub)] = 0.0
            violation += float(np.sum(low_v + up_v))
        elif isinstance(c, dict):
            c_type = c.get("type", "").lower()
            c_fun = c.get("fun")
            c_args = c.get("args", args)
            if c_fun is None:
                raise ValueError("constraint dict must contain key 'fun'")
            values = np.atleast_1d(c_fun(x, *c_args))
            if c_type == "ineq":
                violation += float(np.sum(np.maximum(-values, 0.0)))
            elif c_type == "eq":
                violation += float(np.sum(np.abs(values)))
            else:
                raise ValueError(f"unsupported constraint dict type: {c_type}")
        else:
            raise TypeError(
                "constraints must be LinearConstraint, NonlinearConstraint, or SLSQP-style dict"
            )
    return violation


def _nondominated_indices(objective_matrix: np.ndarray) -> np.ndarray:
    """Return indices of Pareto non-dominated points for minimization."""
    n = objective_matrix.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        better_or_equal = np.all(objective_matrix <= objective_matrix[i], axis=1)
        strictly_better = np.any(objective_matrix < objective_matrix[i], axis=1)
        dominates_i = better_or_equal & strictly_better
        dominates_i[i] = False
        if np.any(dominates_i):
            is_dominated[i] = True
    return np.where(~is_dominated)[0]


def _reflect_bounds(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Reflect then clip points into bounds."""
    y = x.copy()
    y = np.where(y < lower, 2.0 * lower - y, y)
    y = np.where(y > upper, 2.0 * upper - y, y)
    return np.clip(y, lower, upper)


def dragonfly_optimize(
    func: Callable[..., ArrayLike],
    bounds: Union[Bounds, Sequence[Tuple[float, float]]],
    args: Tuple[Any, ...] = (),
    maxiter: int = 200,
    popsize: int = 30,
    tol: float = 1e-6,
    mutation: Optional[Union[float, Tuple[float, float]]] = None,
    recombination: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    callback: Optional[Callable[[np.ndarray, float], bool]] = None,
    disp: bool = False,
    polish: bool = False,
    init: Union[str, np.ndarray] = "random",
    atol: float = 0.0,
    updating: str = "immediate",
    workers: int = 1,
    constraints: Union[Sequence[Any], Any] = (),
    x0: Optional[ArrayLike] = None,
    integrality: Optional[ArrayLike] = None,
    vectorized: bool = False,
    *,
    multi_objective: bool = False,
    objective_weights: Optional[ArrayLike] = None,
    scalarization: str = "weighted_sum",
    return_pareto: bool = False,
    penalty_start: float = 10.0,
    penalty_growth: float = 1.05,
    c_max: float = 2.0,
    c_min: float = 0.2,
    inertia_start: float = 0.9,
    inertia_end: float = 0.2,
    neighbor_radius_start: Optional[float] = None,
    neighbor_radius_end: float = 0.0,
    coulomb_alpha_mean: float = 2.0,
    coulomb_alpha_std: float = 0.25,
    k0: float = 1.0,
    levy_beta: float = 1.5,
) -> OptimizeResult:
    """Optimize an objective with CFSSDA and a SciPy-like interface.

    Args:
        func: Objective function. Input shape (n_dim,) and returns scalar or vector.
        bounds: Box constraints for each variable.
        args: Extra positional args passed to the objective.
        maxiter: Maximum number of iterations.
        popsize: Base population multiplier. Actual population is popsize * n_dim.
        tol: Relative tolerance for convergence on population energies.
        mutation: Accepted for compatibility, not used in CFSSDA.
        recombination: Accepted for compatibility, not used in CFSSDA.
        seed: Random seed or ``np.random.Generator``.
        callback: Optional callback callback(xk, convergence). Return True to stop.
        disp: Whether to print optimization progress.
        polish: Whether to run a local minimization from best point at the end.
        init: "random" or custom array with shape (n_pop, n_dim).
        atol: Absolute tolerance for convergence.
        updating: Accepted for compatibility, currently informational.
        workers: Accepted for compatibility, current implementation uses single process.
        constraints: Sequence of SciPy constraints or SLSQP-style dict constraints.
        x0: Optional initial guess inserted into population[0].
        integrality: Accepted for compatibility, currently not enforced.
        vectorized: If True, objective can accept (n_pop, n_dim).
        multi_objective: Whether objective returns multiple objectives.
        objective_weights: Weights for weighted_sum scalarization.
        scalarization: "weighted_sum" or "tchebycheff".
        return_pareto: If True and multi_objective=True, return Pareto archive.
        penalty_start: Initial penalty multiplier for constraints.
        penalty_growth: Geometric growth rate of penalty multiplier.
        c_max: Maximum behavior coefficient.
        c_min: Minimum behavior coefficient.
        inertia_start: Initial inertia weight.
        inertia_end: Final inertia weight.
        neighbor_radius_start: Initial neighborhood radius. Auto if None.
        neighbor_radius_end: Final neighborhood radius.
        coulomb_alpha_mean: Mean of normal variable in exponential k(t) decay.
        coulomb_alpha_std: Std of normal variable in exponential k(t) decay.
        k0: Initial Coulomb-search coefficient.
        levy_beta: Beta parameter for Levy flight.

    Returns:
        OptimizeResult: SciPy-compatible result object.
    """
    del mutation, recombination, updating, workers, integrality

    lower, upper = _parse_bounds(bounds)
    n_dim = lower.size
    n_pop = max(20, int(popsize) * n_dim)
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")
    if n_pop < 2:
        raise ValueError("population size must be >= 2")
    if scalarization not in {"weighted_sum", "tchebycheff"}:
        raise ValueError("scalarization must be 'weighted_sum' or 'tchebycheff'")

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    constraints_seq: Sequence[Any]
    if constraints is None:
        constraints_seq = ()
    elif isinstance(constraints, (list, tuple)):
        constraints_seq = constraints
    else:
        constraints_seq = (constraints,)

    span = upper - lower
    if neighbor_radius_start is None:
        neighbor_radius_start = 0.25 * float(np.linalg.norm(span))
    neighbor_radius_start = max(neighbor_radius_start, EPS)
    neighbor_radius_end = max(neighbor_radius_end, 0.0)

    if isinstance(init, str):
        if init.lower() != "random":
            raise ValueError("only init='random' or custom array are supported")
        population = rng.uniform(lower, upper, size=(n_pop, n_dim))
    else:
        population = np.asarray(init, dtype=float)
        if population.ndim != 2 or population.shape[1] != n_dim:
            raise ValueError("custom init must have shape (n_pop, n_dim)")
        n_pop = population.shape[0]

    if x0 is not None:
        x0_arr = np.asarray(x0, dtype=float).reshape(-1)
        if x0_arr.size != n_dim:
            raise ValueError("x0 dimension does not match bounds")
        population[0] = np.clip(x0_arr, lower, upper)

    delta_x = rng.uniform(-0.1, 0.1, size=(n_pop, n_dim)) * span

    objective_vectors: List[np.ndarray] = []
    objective_points: List[np.ndarray] = []
    nfev = 0

    def evaluate_objective(x: np.ndarray) -> _ObjectiveValue:
        nonlocal nfev
        raw = func(x, *args)
        nfev += 1
        vec = np.atleast_1d(np.asarray(raw, dtype=float)).reshape(-1)
        if not multi_objective:
            if vec.size != 1:
                raise ValueError("objective returned a vector; set multi_objective=True")
            scalar = float(vec[0])
        else:
            if vec.size < 2:
                raise ValueError("multi_objective=True requires objective vector with length >= 2")
            if objective_weights is None:
                weights = np.full(vec.size, 1.0 / vec.size)
            else:
                weights = np.asarray(objective_weights, dtype=float).reshape(-1)
                if weights.size != vec.size:
                    raise ValueError("objective_weights length must match number of objectives")
                if np.any(weights < 0):
                    raise ValueError("objective_weights must be non-negative")
                weight_sum = np.sum(weights)
                if weight_sum <= 0:
                    raise ValueError("objective_weights sum must be positive")
                weights = weights / weight_sum

            if scalarization == "weighted_sum":
                scalar = float(np.dot(weights, vec))
            else:
                ideal = np.min(obj_matrix, axis=0) if (obj_matrix := np.asarray(objective_vectors)).size else vec
                scalar = float(np.max(weights * np.abs(vec - ideal)))

            if return_pareto:
                objective_vectors.append(vec.copy())
                objective_points.append(x.copy())
        return _ObjectiveValue(scalar=scalar, vector=vec)

    if vectorized:
        # Keep deterministic nfev accounting per point while still allowing batched call.
        objective_values = np.asarray(func(population, *args), dtype=float)
        if objective_values.ndim == 1:
            objective_values = objective_values[:, None]
        if objective_values.shape[0] != n_pop:
            raise ValueError("vectorized objective must return shape (n_pop,) or (n_pop, n_obj)")
        nfev += n_pop
        if not multi_objective and objective_values.shape[1] != 1:
            raise ValueError("objective returned multiple values; set multi_objective=True")
        if multi_objective and objective_values.shape[1] < 2:
            raise ValueError("multi_objective=True requires objective vector with length >= 2")
        if multi_objective:
            if objective_weights is None:
                weights = np.full(objective_values.shape[1], 1.0 / objective_values.shape[1])
            else:
                weights = np.asarray(objective_weights, dtype=float).reshape(-1)
                if weights.size != objective_values.shape[1]:
                    raise ValueError("objective_weights length must match number of objectives")
                weights = weights / (np.sum(weights) + EPS)
            if scalarization == "weighted_sum":
                objective_scalars = objective_values @ weights
            else:
                ideal = np.min(objective_values, axis=0)
                objective_scalars = np.max(weights[None, :] * np.abs(objective_values - ideal), axis=1)
            if return_pareto:
                objective_vectors.extend([row.copy() for row in objective_values])
                objective_points.extend([row.copy() for row in population])
        else:
            objective_scalars = objective_values[:, 0]
        objective_vectors_arr = objective_values
    else:
        values = [evaluate_objective(population[i]) for i in range(n_pop)]
        objective_scalars = np.asarray([v.scalar for v in values], dtype=float)
        objective_vectors_arr = np.vstack([v.vector for v in values])

    penalties = np.array(
        [_constraint_violation(population[i], constraints_seq, args) for i in range(n_pop)],
        dtype=float,
    )
    penalty_factor = float(penalty_start)
    energies = objective_scalars + penalty_factor * penalties

    best_idx = int(np.argmin(energies))
    best_x = population[best_idx].copy()
    best_energy = float(energies[best_idx])
    best_obj_vec = objective_vectors_arr[best_idx].copy()
    success = False
    message = "Maximum number of iterations reached."

    for it in range(1, maxiter + 1):
        progress = (it - 1) / max(maxiter - 1, 1)
        inertia = inertia_start + (inertia_end - inertia_start) * progress
        behavior_base = c_max + (c_min - c_max) * progress
        neighborhood_radius = neighbor_radius_start + (neighbor_radius_end - neighbor_radius_start) * progress

        # Mass terms from fitness ranking (minimization).
        curr_best = float(np.min(energies))
        curr_worst = float(np.max(energies))
        mass_raw = (energies - curr_worst) / (curr_best - curr_worst + EPS)
        mass_raw = np.maximum(mass_raw, EPS)
        mass = mass_raw / (np.sum(mass_raw) + EPS)
        gamma_w = c_min + (c_max - c_min) * mass_raw
        fit_g = gamma_w * mass

        order = np.argsort(mass)[::-1]
        kbest_count = max(1, int(np.ceil(n_pop - (n_pop - 1) * progress)))
        kbest = order[:kbest_count]

        food_idx = int(np.argmin(energies))
        enemy_idx = int(np.argmax(energies))
        food_pos = population[food_idx]
        enemy_pos = population[enemy_idx]

        alpha_hat = abs(rng.normal(coulomb_alpha_mean, coulomb_alpha_std))
        k_t = k0 * np.exp(-alpha_hat * it / maxiter)

        new_population = population.copy()
        new_delta_x = delta_x.copy()
        max_step = 0.2 * span

        for i in range(n_pop):
            distances = np.linalg.norm(population - population[i], axis=1)
            neighbors = np.where((distances > 0.0) & (distances <= neighborhood_radius))[0]

            if neighbors.size == 0:
                levy_step = _levy_flight(rng, n_dim, beta=levy_beta) * span
                new_delta_x[i] = inertia * delta_x[i] + levy_step
                new_population[i] = population[i] + new_delta_x[i]
                continue

            separation = -np.sum(population[i] - population[neighbors], axis=0)
            alignment = np.mean(delta_x[neighbors], axis=0)
            cohesion = np.mean(population[neighbors], axis=0) - population[i]
            food_attr = food_pos - population[i]
            enemy_avoid = enemy_pos + population[i]

            total_force = np.zeros(n_dim, dtype=float)
            for j in kbest:
                if j == i:
                    continue
                diff = population[j] - population[i]
                dist = np.linalg.norm(diff) + EPS
                total_force += rng.uniform() * k_t * mass[i] * mass[j] * (diff / dist)

            enemy_diff = enemy_pos - population[i]
            enemy_dist = np.linalg.norm(enemy_diff) + EPS
            total_force += rng.uniform() * (-k_t) * mass[i] * mass[enemy_idx] * (enemy_diff / enemy_dist)
            acceleration = total_force / (fit_g[i] + EPS)

            s_w = behavior_base * rng.uniform()
            a_w = behavior_base * rng.uniform()
            c_w = behavior_base * rng.uniform()
            f_w = 2.0 * rng.uniform()
            e_w = behavior_base * rng.uniform()

            candidate_delta = (
                inertia * delta_x[i]
                + s_w * separation
                + a_w * alignment
                + c_w * cohesion
                + f_w * food_attr
                + e_w * enemy_avoid
                + acceleration
            )
            candidate_delta = np.clip(candidate_delta, -max_step, max_step)
            new_delta_x[i] = candidate_delta
            new_population[i] = population[i] + candidate_delta

        new_population = _reflect_bounds(new_population, lower, upper)

        if vectorized:
            new_obj_values = np.asarray(func(new_population, *args), dtype=float)
            if new_obj_values.ndim == 1:
                new_obj_values = new_obj_values[:, None]
            nfev += n_pop
            if multi_objective:
                if objective_weights is None:
                    weights = np.full(new_obj_values.shape[1], 1.0 / new_obj_values.shape[1])
                else:
                    weights = np.asarray(objective_weights, dtype=float).reshape(-1)
                    weights = weights / (np.sum(weights) + EPS)
                if scalarization == "weighted_sum":
                    new_obj_scalars = new_obj_values @ weights
                else:
                    ideal = np.minimum(np.min(objective_vectors_arr, axis=0), np.min(new_obj_values, axis=0))
                    new_obj_scalars = np.max(weights[None, :] * np.abs(new_obj_values - ideal), axis=1)
                if return_pareto:
                    objective_vectors.extend([row.copy() for row in new_obj_values])
                    objective_points.extend([row.copy() for row in new_population])
            else:
                new_obj_scalars = new_obj_values[:, 0]
            new_obj_vectors = new_obj_values
        else:
            values = [evaluate_objective(new_population[i]) for i in range(n_pop)]
            new_obj_scalars = np.asarray([v.scalar for v in values], dtype=float)
            new_obj_vectors = np.vstack([v.vector for v in values])

        new_penalties = np.array(
            [_constraint_violation(new_population[i], constraints_seq, args) for i in range(n_pop)],
            dtype=float,
        )
        penalty_factor *= penalty_growth
        new_energies = new_obj_scalars + penalty_factor * new_penalties

        improved = new_energies < energies
        population[improved] = new_population[improved]
        delta_x[improved] = new_delta_x[improved]
        objective_scalars[improved] = new_obj_scalars[improved]
        objective_vectors_arr[improved] = new_obj_vectors[improved]
        penalties[improved] = new_penalties[improved]
        energies[improved] = new_energies[improved]

        curr_idx = int(np.argmin(energies))
        if energies[curr_idx] < best_energy:
            best_idx = curr_idx
            best_x = population[curr_idx].copy()
            best_energy = float(energies[curr_idx])
            best_obj_vec = objective_vectors_arr[curr_idx].copy()

        convergence = float(np.std(energies))
        threshold = float(atol + tol * abs(np.mean(energies)))
        if disp:
            print(
                f"[CFSSDA] iter={it:4d} best={best_energy:.8e} "
                f"mean={np.mean(energies):.8e} std={convergence:.8e}"
            )

        if callback is not None:
            if callback(best_x.copy(), convergence):
                success = False
                message = "Stopped by callback."
                break

        if convergence <= threshold:
            success = True
            message = "Optimization converged."
            break

    if polish:
        def local_obj(x_local: np.ndarray) -> float:
            obj = np.atleast_1d(np.asarray(func(x_local, *args), dtype=float))
            if obj.size == 1:
                obj_scalar = float(obj[0])
            else:
                if objective_weights is None:
                    weights_local = np.full(obj.size, 1.0 / obj.size)
                else:
                    weights_local = np.asarray(objective_weights, dtype=float).reshape(-1)
                    weights_local = weights_local / (np.sum(weights_local) + EPS)
                if scalarization == "weighted_sum":
                    obj_scalar = float(np.dot(weights_local, obj))
                else:
                    obj_scalar = float(np.max(weights_local * np.abs(obj - best_obj_vec)))
            pen = _constraint_violation(x_local, constraints_seq, args)
            return obj_scalar + penalty_factor * pen

        method = "L-BFGS-B" if not constraints_seq else "SLSQP"
        local_res = minimize(
            local_obj,
            x0=best_x,
            method=method,
            bounds=Bounds(lower, upper),
            constraints=constraints_seq if constraints_seq else (),
        )
        nfev += int(getattr(local_res, "nfev", 0))
        if local_res.fun < best_energy:
            best_x = np.asarray(local_res.x, dtype=float)
            best_energy = float(local_res.fun)
            val = np.atleast_1d(np.asarray(func(best_x, *args), dtype=float))
            nfev += 1
            best_obj_vec = val

    result = OptimizeResult()
    result.x = best_x
    result.fun = float(best_obj_vec[0] if best_obj_vec.size == 1 else np.dot(
        np.asarray(objective_weights if objective_weights is not None else np.full(best_obj_vec.size, 1.0 / best_obj_vec.size), dtype=float),
        best_obj_vec
    ))
    result.success = bool(success)
    result.message = message
    result.nit = int(it if "it" in locals() else 0)
    result.nfev = int(nfev)
    result.population = population.copy()
    result.population_energies = energies.copy()
    result.constraint_violation = float(_constraint_violation(best_x, constraints_seq, args))
    result.objective_vector = best_obj_vec.copy()
    result.penalized_fun = float(best_energy)
    result.optimizer = "CFSSDA"

    if multi_objective:
        result.fun_vector = best_obj_vec.copy()
        if return_pareto and len(objective_vectors) > 0:
            archive_f = np.asarray(objective_vectors, dtype=float)
            archive_x = np.asarray(objective_points, dtype=float)
            nd_idx = _nondominated_indices(archive_f)
            result.pareto_f = archive_f[nd_idx]
            result.pareto_x = archive_x[nd_idx]

    return result
