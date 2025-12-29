# solve_quantum.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple


def _workload_for_budget(num_vars: int, time_budget_s: float) -> Tuple[int, int, int]:
    """Heuristic QAOA settings tuned for fast demo iterations (typically 10s budget)."""

    # For the small demo (8-18 variables), keep reps=1 and ultra-low shots/iterations.
    reps = 1
    shots = 64  # Very small number of shots for quick feedback
    maxiter = 12  # Light optimizer iterations

    # Tie max iterations to the wall-clock budget but keep it reasonably bounded.
    # For a 10s budget, this gives us ~30-60 iterations depending on circuit size.
    maxiter = max(6, int(min(maxiter, time_budget_s * 1.2)))

    return reps, shots, maxiter


def _maybe_extend_sys_path_from_local_venv() -> None:
    """If a local ``venv`` exists, add its site-packages to ``sys.path``."""

    root = Path(__file__).resolve().parent
    for candidate in sorted(root.glob("venv/lib/python*/site-packages")):
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_qiskit_components():
    """Import Qiskit pieces, retrying with a local venv path if needed.

    This helper is intentionally thin so it can adapt to breaking changes in
    Qiskit without touching the rest of the code. For Qiskit 2.x the
    ``qiskit.primitives.Sampler`` class was removed in favour of the V2
    primitives such as :class:`StatevectorSampler`, which implement the
    ``BaseSamplerV2`` interface expected by ``qiskit_algorithms.QAOA``.
    """

    try:
        from qiskit_algorithms import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit.primitives import StatevectorSampler
        return QAOA, COBYLA, QuadraticProgram, MinimumEigenOptimizer, StatevectorSampler
    except ImportError:
        _maybe_extend_sys_path_from_local_venv()
        try:
            from qiskit_algorithms import QAOA
            from qiskit_algorithms.optimizers import COBYLA
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.algorithms import MinimumEigenOptimizer
            from qiskit.primitives import StatevectorSampler
            return QAOA, COBYLA, QuadraticProgram, MinimumEigenOptimizer, StatevectorSampler
        except ImportError as exc:
            raise ValueError(
                "Qiskit not available for the current interpreter "
                f"({sys.executable}). Install qiskit or run with ./venv/bin/python."
            ) from exc


def _count_pauli_terms(Q: Sequence[Sequence[float]]) -> int:
    """Estimate the number of ZZ + Z Pauli terms induced by the QUBO."""

    num_vars = len(Q)
    diag = sum(1 for i in range(num_vars) if Q[i][i] != 0)
    offdiag = sum(1 for i in range(num_vars) for j in range(i + 1, num_vars) if Q[i][j] != 0)
    return diag + offdiag


def solve_qaoa(
    Q: Sequence[Sequence[float]],
    max_variables: int = 20,
    time_budget_s: float = 60.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_pauli_terms: Optional[int] = 120,
) -> Tuple[Sequence[int], float, Dict[str, int]]:
    """Solve the QUBO with QAOA, optionally skipping oversized instances.

    The import of Qiskit components is deferred so the classical path can run
    without heavyweight dependencies. If the QUBO is larger than the configured
    limit or Qiskit is unavailable, a ``ValueError`` is raised and callers can
    fall back to classical execution. The internal workload is chosen to
    maximize work within the provided time budget.
    """

    QAOA, COBYLA, QuadraticProgram, MinimumEigenOptimizer, StatevectorSampler = _load_qiskit_components()

    num_vars = len(Q)
    if any(len(row) != num_vars for row in Q):
        raise ValueError("QUBO matrix must be square for QAOA conversion")
    if num_vars > max_variables:
        raise ValueError(
            f"QAOA demo capped at {max_variables} variables; received {num_vars}."
        )

    pauli_terms = _count_pauli_terms(Q)
    if max_pauli_terms is not None and pauli_terms > max_pauli_terms:
        raise ValueError(
            "QAOA skipped: the QUBO is too dense for the demo settings "
            f"({pauli_terms} Pauli terms > limit {max_pauli_terms}). "
            "Reduce the horizon/penalties or call solve_qaoa(..., max_pauli_terms=None) to force it."
        )

    reps, shots, maxiter = _workload_for_budget(num_vars, time_budget_s)

    # Bridge QAOA's callback API (eval_count, params, value, metadata) to the
    # simple two-argument ``progress_callback(current_step, total_steps)``
    # expected by the runner.
    def _qaoa_callback(eval_count, _params, _mean, _metadata):
        if progress_callback is None:
            return
        try:
            current = int(eval_count)
        except (TypeError, ValueError):
            current = 0
        progress_callback(min(current, maxiter), maxiter)

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

    # Use a statevector-based sampler (V2 primitives) and cap optimizer iterations
    # to keep runtime small. ``default_shots`` controls the sampling resolution
    # used internally by QAOA.
    sampler = StatevectorSampler(default_shots=shots)
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        callback=_qaoa_callback if progress_callback is not None else None,
    )
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    x = [int(result.variables_dict[f"x{i}"]) for i in range(num_vars)]

    config = {"shots": shots, "maxiter": maxiter, "reps": reps}
    return x, result.fval, config
