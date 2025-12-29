# solve_classical.py
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple


def _energy(Q: Sequence[Sequence[float]], x: Sequence[int]) -> float:
    """Compute the QUBO objective value for assignment ``x``."""

    total = 0.0
    n = len(x)
    for i in range(n):
        total += Q[i][i] * x[i]
        for j in range(i + 1, n):
            if Q[i][j] != 0:
                total += Q[i][j] * x[i] * x[j]
    return total


def solve_simulated_annealing(
    Q: Sequence[Sequence[float]],
    steps: int = 5000,
    temperature: float = 1.0,
    cooling: float = 0.995,
    seed: int = 7,
) -> Tuple[List[int], float]:
    """Solve a QUBO with a lightweight simulated annealing routine.

    This implementation avoids external dependencies so that the demo runs even
    in constrained environments. It returns the best binary assignment and its
    objective value.
    """

    n = len(Q)
    rng = random.Random(seed)

    current = [rng.randrange(2) for _ in range(n)]
    current_energy = _energy(Q, current)
    best = list(current)
    best_energy = current_energy

    T = temperature
    for _ in range(steps):
        idx = rng.randrange(n)
        candidate = list(current)
        candidate[idx] = 1 - candidate[idx]
        candidate_energy = _energy(Q, candidate)

        delta = candidate_energy - current_energy
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            current = candidate
            current_energy = candidate_energy
            if candidate_energy < best_energy:
                best = list(candidate)
                best_energy = candidate_energy

        T *= cooling

    return best, best_energy
