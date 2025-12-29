# solve_quantum.py
from __future__ import annotations

from typing import Dict, Sequence, Tuple


def _workload_for_budget(num_vars: int, time_budget_s: float) -> Tuple[int, int, int]:
    """Heuristic QAOA settings that stay under ~1 minute but do real work."""

    heavy_budget = time_budget_s >= 45.0
    reps = 2 if num_vars <= 20 and heavy_budget else 1
    shots = 1024 if heavy_budget else 512
    maxiter = 150 if heavy_budget else 90

    if time_budget_s < 30.0:
        shots = min(shots, 512)
        maxiter = min(maxiter, 70)

    # Tie max iterations to the wall-clock budget to avoid runaway solves.
    maxiter = max(25, int(min(maxiter, time_budget_s * 3.0)))

    return reps, shots, maxiter


def solve_qaoa(
    Q: Sequence[Sequence[float]], max_variables: int = 20, time_budget_s: float = 60.0
) -> Tuple[Sequence[int], float, Dict[str, int]]:
    """Solve the QUBO with QAOA, optionally skipping oversized instances.

    The import of Qiskit components is deferred so the classical path can run
    without heavyweight dependencies. If the QUBO is larger than the configured
    limit or Qiskit is unavailable, a ``ValueError`` is raised and callers can
    fall back to classical execution. The internal workload is chosen to
    maximize work within the provided time budget.
    """

    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit.primitives import Sampler
    except ImportError:
        raise ValueError("Qiskit not installed; skipping quantum solve")

    num_vars = len(Q)
    if any(len(row) != num_vars for row in Q):
        raise ValueError("QUBO matrix must be square for QAOA conversion")
    if num_vars > max_variables:
        raise ValueError(
            f"QAOA demo capped at {max_variables} variables; received {num_vars}."
        )

    reps, shots, maxiter = _workload_for_budget(num_vars, time_budget_s)

    qp = QuadraticProgram()

    for i in range(num_vars):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": Q[i][i] for i in range(num_vars)}
    quadratic = {}

    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            if Q[i][j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = Q[i][j]

    qp.minimize(linear=linear, quadratic=quadratic)

    # Use a shot-based sampler and cap optimizer iterations to keep runtime small.
    qaoa = QAOA(sampler=Sampler(shots=shots), optimizer=COBYLA(maxiter=maxiter), reps=reps)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    x = [int(result.variables_dict[f"x{i}"]) for i in range(num_vars)]

    config = {"shots": shots, "maxiter": maxiter, "reps": reps}
    return x, result.fval, config
