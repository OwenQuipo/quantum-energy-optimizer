# solve_classical.py
import itertools
import numpy as np

def solve_bruteforce(Q):
    T = Q.shape[0]
    best_x = None
    best_cost = float("inf")

    for bits in itertools.product([0, 1], repeat=T):
        x = np.array(bits)
        cost = x @ Q @ x
        if cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x, best_cost
