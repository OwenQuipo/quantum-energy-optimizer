# solve_quantum.py
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_algorithms.optimizers import COBYLA
import numpy as np

def solve_qaoa(Q):
    T = Q.shape[0]
    qp = QuadraticProgram()

    for i in range(T):
        qp.binary_var(f"x{i}")

    linear = {f"x{i}": Q[i, i] for i in range(T)}
    quadratic = {}

    for i in range(T):
        for j in range(i + 1, T):
            if Q[i, j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = Q[i, j]

    qp.minimize(linear=linear, quadratic=quadratic)

    qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qp)
    x = np.array([result.variables_dict[f"x{i}"] for i in range(T)])

    return x, result.fval
