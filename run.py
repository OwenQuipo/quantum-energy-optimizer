# run.py
from __future__ import annotations

import warnings

from data import P_MAX, SOC_0, demand, price
from qubo import build_qubo_demand, decode_solution
from solve_classical import solve_simulated_annealing
from solve_quantum import solve_qaoa

warnings.filterwarnings("ignore", category=FutureWarning)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def summarize_cost(solution, meta):
    discharge = solution["discharge"]
    peak_kW = solution["peak_kW"]

    energy_cost = dot(price, demand) - dot(price, discharge) * P_MAX
    demand_charge = peak_kW * meta.demand_rate

    return energy_cost + demand_charge, energy_cost, demand_charge


def main():
    Q, meta = build_qubo_demand(price, demand, SOC_0, P_MAX)

    x_classical, c_classical = solve_simulated_annealing(Q)
    decoded_classical = decode_solution(x_classical, meta)

    try:
        x_quantum, c_quantum = solve_qaoa(Q)
        decoded_quantum = decode_solution(x_quantum, meta)
        quantum_failed = False
    except ValueError as exc:
        decoded_quantum = None
        c_quantum = None
        quantum_failed = True
        quantum_error = str(exc)

    total_c, e_c, d_c = summarize_cost(decoded_classical, meta)
    print("Classical schedule:", decoded_classical)
    print(f"  Energy cost: ${e_c:.2f}, Demand charge: ${d_c:.2f}, Total: ${total_c:.2f}")

    if quantum_failed:
        print("Quantum run skipped:", quantum_error)
    else:
        total_q, e_q, d_q = summarize_cost(decoded_quantum, meta)
        print("Quantum schedule:", decoded_quantum)
        print(
            f"  Energy cost: ${e_q:.2f}, Demand charge: ${d_q:.2f}, Total: ${total_q:.2f}"
        )


if __name__ == "__main__":
    main()
