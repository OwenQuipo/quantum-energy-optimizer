# Quantum Battery Dispatch (v0.1)

This repository contains a **quantum-native optimization engine** for battery
charge / discharge scheduling.

The core decision problem is formulated as a **QUBO (Quadratic Unconstrained
Binary Optimization)** and solved using **QAOA** via Qiskit, with a classical
brute-force solver used for verification on small instances.

## What this does
Given:
- hourly electricity prices
- a battery with limited energy and power
- an initial state of charge

The system computes an **optimal dispatch schedule**:
- CHARGE
- DISCHARGE
- IDLE

The quantum solver directly participates in selecting the schedule.

## Why quantum?
Battery dispatch decisions are:
- discrete
- globally coupled in time
- constraint-heavy

This makes them a natural fit for quantum optimization methods, rather than
continuous relaxations or heuristic rules.

## Repository structure
#### quantum-battery/
#### data.py # Hardcoded example scenario
#### qubo.py # QUBO construction (quantum-native core)
#### solve_classical.py # Classical simulated annealing solver for QUBOs
#### solve_quantum.py # QAOA-based solver using Qiskit (optional)
#### run.py # Entry point
#### README.md

## Status
- v0.2: explicit demand-charge modeling with a peak variable
- Next: longer horizons and rolling optimization

## Running the demo
The classical path has **no third-party dependencies**; invoke it directly:

```
python run.py
```

If Qiskit is installed, a QAOA run will be attempted as well. Otherwise the
quantum step is skipped with a clear message while the classical result is
still produced.

This is an early research-driven prototype.

