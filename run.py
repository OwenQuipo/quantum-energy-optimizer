# run.py
from data import price, SOC_0, P_MAX
from qubo import build_qubo_subset
from solve_classical import solve_bruteforce
from solve_quantum import solve_qaoa
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


Q, K = build_qubo_subset(price, SOC_0, P_MAX, dt=1.0, lambda_k=10.0)

x_classical, c_classical = solve_bruteforce(Q)
x_quantum, c_quantum = solve_qaoa(Q)

print("K (discharge hours allowed):", K)
print("Classical solution:", x_classical, c_classical, "sum=", x_classical.sum())
print("Quantum solution:  ", x_quantum, c_quantum, "sum=", x_quantum.sum())
