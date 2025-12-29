# run.py
from __future__ import annotations

import sys
import threading
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from data import E_MAX, P_MAX, SOC_0, demand, price
from qubo import build_qubo_hybrid, decode_solution, encode_schedule_to_binary
from solve_classical import compute_qubo_energy, solve_simulated_annealing
from solve_quantum import solve_qaoa

warnings.filterwarnings("ignore", category=FutureWarning)

# SciPy emits ``SparseEfficiencyWarning`` from deep inside the QAOA stack when it
# implicitly converts matrices to CSC format. This is harmless for this demo but
# very noisy, so we silence just that warning category if SciPy is available.
try:  # pragma: no cover - optional dependency handling
    from scipy.sparse import SparseEfficiencyWarning  # type: ignore[import]
except Exception:  # SciPy not installed or incompatible
    SparseEfficiencyWarning = None  # type: ignore[assignment]

if SparseEfficiencyWarning is not None:
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def evaluate_schedule(
    discharge_levels: Sequence[float],
    charge: Sequence[int] | None,
    demand_rate: float,
    peak_hours: Sequence[int] | None = None,
    previous_peak: float = 0.0,
) -> Dict[str, float]:
    """Compute exact cost metrics for a fixed dispatch schedule."""

    net_demand = []
    for t, base in enumerate(demand):
        flow = -discharge_levels[t] * P_MAX
        if charge:
            flow += charge[t] * P_MAX
        net = base + flow
        net_demand.append(net)

    if peak_hours:
        peak_domain = [net_demand[t] for t in peak_hours if t < len(net_demand)]
    else:
        peak_domain = net_demand
    peak_kW = max([previous_peak] + peak_domain)
    energy_cost = dot(price, net_demand)
    demand_charge = peak_kW * demand_rate
    total = energy_cost + demand_charge
    return {
        "net_demand": net_demand,
        "peak_kW": peak_kW,
        "energy_cost": energy_cost,
        "demand_charge": demand_charge,
        "total": total,
    }


def explain_schedule(discharge_levels: Sequence[float]) -> List[Dict[str, float]]:
    """Provide marginal savings and peak relief intuition per hour."""

    baseline_peak = max(demand)
    explanations: List[Dict[str, float]] = []
    for t, level in enumerate(discharge_levels):
        marginal_savings = price[t] * level * P_MAX
        projected_peak = max(
            max(demand[:t] + demand[t + 1 :]) if len(demand) > 1 else 0,
            demand[t] - level * P_MAX,
        )
        peak_drop = max(0.0, baseline_peak - projected_peak)
        explanations.append(
            {
                "hour": t,
                "level_P": level * P_MAX,
                "marginal_energy_savings": marginal_savings,
                "marginal_peak_relief": peak_drop,
            }
        )
    return explanations


def enumerate_feasible_schedules(meta) -> Iterable[Tuple[List[float], int]]:
    """Generate all dispatch schedules that respect the energy budget."""

    level_options = [0.0, meta.discharge_unit, 2 * meta.discharge_unit]
    for choice in product(level_options, repeat=meta.T):
        units_used = sum(int(round(lvl / meta.discharge_unit)) for lvl in choice)
        if units_used <= meta.discharge_capacity_units:
            yield list(choice), units_used


def brute_force_optimal(
    Q,
    meta,
    progress_event: Optional["threading.Event"] = None,
    progress_reporter: Optional[Callable[[int, int], None]] = None,
    progress_interval: Optional[int] = None,
):
    """Enumerate every feasible schedule and return the QUBO optimum."""

    best_schedule: List[float] | None = None
    best_energy = float("inf")
    total_candidates = 3 ** meta.T

    for idx, (schedule, units) in enumerate(enumerate_feasible_schedules(meta), start=1):
        if progress_event and progress_event.is_set():
            if progress_reporter:
                progress_reporter(idx - 1, total_candidates)
            progress_event.clear()

        x = encode_schedule_to_binary(schedule, meta)
        energy = compute_qubo_energy(Q, x)
        if energy < best_energy:
            best_energy = energy
            best_schedule = schedule

        if progress_event and progress_event.is_set():
            if progress_reporter:
                progress_reporter(idx, total_candidates)
            progress_event.clear()
        elif progress_reporter and progress_interval and idx % progress_interval == 0:
            progress_reporter(idx, total_candidates)

    if best_schedule is None:
        raise RuntimeError("No feasible schedules found")
    return best_schedule, best_energy


def greedy_baseline(meta) -> List[float]:
    """Simple heuristic: pick the best hours by weighted price/peak proxy."""

    scores = []
    for t, (p, d) in enumerate(zip(price, demand)):
        # Benefit per unit of discharge energy (half-step granularity)
        benefit = (p * P_MAX * meta.discharge_unit) + meta.peak_weight * d * meta.discharge_unit
        scores.append((benefit, t))

    scores.sort(reverse=True)
    remaining_units = meta.discharge_capacity_units
    dispatch = [0.0 for _ in range(meta.T)]
    for _, t in scores:
        if remaining_units >= 2:
            dispatch[t] = 2 * meta.discharge_unit
            remaining_units -= 2
        elif remaining_units == 1:
            dispatch[t] = meta.discharge_unit
            remaining_units -= 1
    return dispatch


def rolling_horizon(
    horizon_hours: int, include_charge: bool = False, soc0: float = SOC_0
) -> List[float]:
    """Re-optimize over a moving window and execute the first decision only."""

    remaining_soc = soc0
    executed: List[float] = []
    for start in range(len(price)):
        window_price = price[start : start + horizon_hours]
        window_demand = demand[start : start + horizon_hours]
        if not window_price:
            break
        Q, meta = build_qubo_hybrid(
            window_price,
            window_demand,
            remaining_soc,
            E_MAX,
            P_MAX,
            include_charge=include_charge,
        )
        schedule, _ = brute_force_optimal(Q, meta)
        first_action = schedule[0]
        executed.append(first_action)
        remaining_soc = max(0.0, remaining_soc - first_action * P_MAX)
    return executed


def main(include_charge: bool = False, horizon: int | None = None, max_bruteforce_hours: int = 10):
    progress_event = threading.Event()

    def make_reporter(label: str):
        """Return a clean in-place progress bar printer; minimal clutter."""

        bar_width = 20

        def _report(current_step: int, total_steps: int) -> None:
            total = max(int(total_steps), 1)
            current = min(max(int(current_step), 0), total)
            fraction = current / total
            filled = int(bar_width * fraction)
            bar = "█" * filled + " " * (bar_width - filled)
            msg = f"\r{label}: |{bar}| {fraction*100:4.1f}%"
            print(msg, end="", flush=True)
            if current >= total:
                print()

        return _report

    def listen_for_enter() -> None:
        """Watch stdin for Enter presses and request a progress update."""

        for line in sys.stdin:
            if line.strip() == "":
                progress_event.set()

    listener = threading.Thread(target=listen_for_enter, daemon=True)
    listener.start()

    Q, meta = build_qubo_hybrid(price, demand, SOC_0, E_MAX, P_MAX, include_charge=include_charge)

    if meta.T > max_bruteforce_hours:
        brute_schedule = None
        brute_energy = None
        print(
            f"Skipping brute-force search for T={meta.T} (> {max_bruteforce_hours}) to avoid long runtimes."
        )
    else:
        brute_schedule, brute_energy = brute_force_optimal(
            Q,
            meta,
            progress_event=progress_event,
            progress_reporter=make_reporter("Brute force"),
            progress_interval=5000,
        )
    greedy_schedule = greedy_baseline(meta)

    x_classical, _ = solve_simulated_annealing(
        Q,
        progress_event=progress_event,
        progress_reporter=make_reporter("Simulated annealing"),
        progress_interval=500,
    )
    decoded_classical = decode_solution(x_classical, meta)

    try:
        if meta.num_variables > 20:
            raise ValueError(
                f"QAOA demo capped at 20 variables; received {meta.num_variables}."
            )
        x_quantum, _, qaoa_config = solve_qaoa(
            Q,
            time_budget_s=10.0,
            progress_callback=make_reporter("QAOA"),
        )
        decoded_quantum = decode_solution(x_quantum, meta)
        quantum_failed = False
    except ValueError as exc:
        decoded_quantum = None
        quantum_failed = True
        quantum_error = str(exc)
        qaoa_config = None

    def score(schedule: Sequence[float]):
        return evaluate_schedule(schedule, None, meta.demand_rate)

    if brute_schedule is not None:
        brute_score = score(brute_schedule)
    greedy_score = score(greedy_schedule)
    classical_score = score(decoded_classical["discharge"])

    if brute_schedule is not None:
        print("Brute-force optimal schedule (ground truth):", brute_schedule)
        print(
            f"  Energy cost: ${brute_score['energy_cost']:.2f}, Demand charge: ${brute_score['demand_charge']:.2f}, Total: ${brute_score['total']:.2f}"
        )
    else:
        print("Brute-force search skipped; relying on greedy and annealing heuristics instead.")

    if quantum_failed:
        print("Quantum run skipped:", quantum_error)
    else:
        quant_score = score(decoded_quantum["discharge"])
        print("QAOA schedule:", decoded_quantum["discharge"])
        print(
            f"  Energy cost: ${quant_score['energy_cost']:.2f}, Demand charge: ${quant_score['demand_charge']:.2f}, Total: ${quant_score['total']:.2f}"
        )
        if qaoa_config:
            print(
                f"  QAOA workload: reps={qaoa_config['reps']}, shots={qaoa_config['shots']}, maxiter={qaoa_config['maxiter']} (targeting ~1 minute)"
            )

    print("Greedy baseline schedule:", greedy_schedule)
    print(
        f"  Energy cost: ${greedy_score['energy_cost']:.2f}, Demand charge: ${greedy_score['demand_charge']:.2f}, Total: ${greedy_score['total']:.2f}"
    )

    print("Simulated annealing schedule:", decoded_classical["discharge"])
    print(
        f"  Energy cost: ${classical_score['energy_cost']:.2f}, Demand charge: ${classical_score['demand_charge']:.2f}, Total: ${classical_score['total']:.2f}"
    )

    if brute_schedule is not None:
        print("Explainability (per hour):")
        for entry in explain_schedule(brute_schedule):
            print(
                f"  Hour {entry['hour']}: level={entry['level_P']:.1f} kW, energy savings=${entry['marginal_energy_savings']:.2f}, peak relief={entry['marginal_peak_relief']:.1f} kW"
            )

    if horizon:
        rh_schedule = rolling_horizon(horizon, include_charge)
        rh_score = score(rh_schedule)
        print(f"Rolling-horizon first-step policy (window={horizon}h):", rh_schedule)
        print(
            f"  Energy cost: ${rh_score['energy_cost']:.2f}, Demand charge: ${rh_score['demand_charge']:.2f}, Total: ${rh_score['total']:.2f}"
        )


if __name__ == "__main__":
    main()
