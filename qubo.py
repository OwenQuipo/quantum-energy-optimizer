# qubo.py
"""QUBO construction utilities.

This module now builds a formulation that captures both energy charges and a
"true" demand charge through an explicit peak variable. The construction uses
auxiliary slack bits to encode the inequality ``peak >= net_load_t`` for every
time step, which makes the peak cost structurally correct rather than a soft
approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class QuboMeta:
    """Metadata needed to interpret a QUBO solution vector."""

    T: int
    peak_bits: int
    peak_step: float
    energy_slack_bits: int
    slack_bits: int
    demand_rate: float

    @property
    def num_variables(self) -> int:
        return self.T + self.peak_bits + self.energy_slack_bits + self.T * self.slack_bits


def _add_penalty_square(
    Q: List[List[float]], penalty: float, constant: float, terms: List[Tuple[int, float]]
) -> None:
    """Add a squared penalty ``penalty * (constant + sum(a_i x_i))^2`` to ``Q``.

    Parameters
    ----------
    Q:
        The QUBO matrix to update in-place.
    penalty:
        Scaling factor for the penalty term.
    constant:
        Constant offset inside the squared expression.
    terms:
        List of ``(index, coefficient)`` pairs representing ``a_i x_i``.
    """

    for i, (idx_i, weight_i) in enumerate(terms):
        # Diagonal contributions: weight_i^2 * x_i^2 == weight_i^2 * x_i
        Q[idx_i][idx_i] += penalty * (weight_i**2 + 2 * constant * weight_i)

        # Cross terms 2 * weight_i * weight_j * x_i x_j
        for idx_j, weight_j in terms[i + 1 :]:
            Q[idx_i][idx_j] += 2 * penalty * weight_i * weight_j


def build_qubo_demand(
    price: Sequence[float],
    demand: Sequence[float],
    SOC_0: float,
    P_MAX: float,
    dt: float = 1.0,
    demand_rate: float = 15.0,
    lambda_energy: float = 50.0,
    lambda_peak: float = 50.0,
    peak_step: float = 1.0,
) -> Tuple[List[List[float]], QuboMeta]:
    """Build a QUBO that models energy plus demand charges.

    Decision variables (all binary):
    - ``y_t``: discharge at hour ``t`` (at max power ``P_MAX``)
    - ``p_b``: bits encoding the demand-charge peak magnitude
    - ``s_e``: bits representing slack for the energy budget inequality
    - ``s_{t,b}``: slack bits enforcing ``peak >= net_load[t]``

    Returns
    -------
    Q:
        Quadratic matrix for the QUBO objective.
    meta:
        ``QuboMeta`` describing how to decode a solution vector.
    """

    price = [float(p) for p in price]
    demand = [float(d) for d in demand]

    if len(price) != len(demand):
        raise ValueError("price and demand must have the same length")

    T = len(price)
    max_peak = int(math.ceil(max(demand)))
    peak_bits = max(1, math.ceil(math.log2(max_peak / peak_step + 1)))
    slack_bits = peak_bits

    # Energy budget: sum(y_t) <= K
    K = int(math.floor(SOC_0 / (P_MAX * dt)))
    energy_slack_bits = max(1, math.ceil(math.log2(K + 1)))

    meta = QuboMeta(
        T=T,
        peak_bits=peak_bits,
        peak_step=peak_step,
        energy_slack_bits=energy_slack_bits,
        slack_bits=slack_bits,
        demand_rate=demand_rate,
    )

    N = meta.num_variables
    Q = [[0.0 for _ in range(N)] for _ in range(N)]

    # Index bookkeeping
    idx_y = list(range(T))
    idx_peak = list(range(T, T + peak_bits))
    idx_energy_slack = list(
        range(idx_peak[-1] + 1, idx_peak[-1] + 1 + energy_slack_bits)
    )
    idx_slack_start = idx_energy_slack[-1] + 1
    idx_slack = []
    for t in range(T):
        start = idx_slack_start + t * slack_bits
        idx_slack.append(list(range(start, start + slack_bits)))

    # 1) Energy cost reduction from discharging
    for t, idx in enumerate(idx_y):
        energy_savings = price[t] * P_MAX * dt
        Q[idx][idx] += -energy_savings

    # 2) Demand charge linear term: demand_rate * peak
    for bit, idx in enumerate(idx_peak):
        Q[idx][idx] += demand_rate * peak_step * (2**bit)

    # 3) Energy budget inequality: sum(y) + slack_e = K
    energy_terms = [(idx, 1.0) for idx in idx_y]
    for bit, idx in enumerate(idx_energy_slack):
        energy_terms.append((idx, float(2**bit)))
    _add_penalty_square(Q, lambda_energy, -float(K), energy_terms)

    # 4) Peak inequality: peak - (demand[t] - P_MAX*y_t) - slack_t = 0
    peak_weights = [(idx, peak_step * (2**bit)) for bit, idx in enumerate(idx_peak)]
    for t in range(T):
        terms: List[Tuple[int, float]] = []
        # Peak bits contribute positively
        terms.extend(peak_weights)
        # Discharge reduces net load, so contributes +P_MAX*y_t to the equality
        terms.append((idx_y[t], P_MAX))
        # Slack bits (non-negative) absorb the remaining margin
        for bit, idx in enumerate(idx_slack[t]):
            terms.append((idx, -peak_step * (2**bit)))

        _add_penalty_square(Q, lambda_peak, -float(demand[t]), terms)

    return Q, meta


def decode_solution(x: Sequence[int], meta: QuboMeta) -> Dict[str, object]:
    """Decode a binary solution vector using ``meta``."""

    pointer = 0
    discharge = [int(val) for val in x[pointer : pointer + meta.T]]
    pointer += meta.T

    peak_bits = [int(val) for val in x[pointer : pointer + meta.peak_bits]]
    pointer += meta.peak_bits
    energy_slack_bits = [int(val) for val in x[pointer : pointer + meta.energy_slack_bits]]
    pointer += meta.energy_slack_bits

    slack = []
    for _ in range(meta.T):
        slack_bits = [int(val) for val in x[pointer : pointer + meta.slack_bits]]
        slack.append(slack_bits)
        pointer += meta.slack_bits

    peak_value = meta.peak_step * sum(bit * (2**idx) for idx, bit in enumerate(peak_bits))
    energy_slack_value = sum(bit * (2**idx) for idx, bit in enumerate(energy_slack_bits))
    slack_values = [sum(bit * (2**idx) for idx, bit in enumerate(bits)) * meta.peak_step for bits in slack]

    return {
        "discharge": discharge,
        "peak_kW": peak_value,
        "energy_slack": energy_slack_value,
        "slack_by_hour": slack_values,
    }
