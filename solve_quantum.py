# solve_quantum.py
from __future__ import annotations

from typing import Sequence


def solve_qaoa(Q: Sequence[Sequence[float]], max_variables: int = 20):
    """Solve the QUBO with QAOA, optionally skipping oversized instances.

    The import of Qiskit components is deferred so the classical path can run
    without heavyweight dependencies. If the QUBO is larger than the configured
    limit or Qiskit is unavailable, a ``ValueError`` is raised and callers can
    fall back to classical execution.
    """

    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit.primitives import StatevectorSampler as Sampler
    except ImportError:
        raise ValueError("Qiskit not installed; skipping quantum solve")

    T = len(Q)
    if T > max_variables:
        raise ValueError(
            f"QAOA demo capped at {max_variables} variables; received {T}."
        )

    qp = QuadraticProgram()

    for i in range(T):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": Q[i][i] for i in range(T)}
    quadratic = {}

    for i in range(T):
        for j in range(i + 1, T):
            if Q[i][j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = Q[i][j]

    qp.minimize(linear=linear, quadratic=quadratic)

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    x = [int(result.variables_dict[f"x{i}"]) for i in range(T)]

    return x, result.fval
