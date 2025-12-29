# qubo.py
import numpy as np

def build_qubo_subset(price, SOC_0, P_MAX, dt=1.0, lambda_k=10.0):
    """
    QUBO for: choose exactly K discharge hours to maximize savings.
    y_t = 1 => discharge at hour t by P_MAX for dt hours

    Objective (min form):
      minimize  -sum_t (price[t] * P_MAX * dt) * y_t  +  lambda_k * (sum_t y_t - K)^2
    """
    price = np.array(price, dtype=float)
    T = len(price)

    # How many discharge-hours worth of energy do we have?
    K = int(np.floor(SOC_0 / (P_MAX * dt)))
    K = max(0, min(K, T))

    # Savings weight per hour if we discharge
    w = price * P_MAX * dt

    Q = np.zeros((T, T), dtype=float)

    # Linear term: -w_t y_t  (put on diagonal)
    for t in range(T):
        Q[t, t] += -w[t]

    # Penalty: lambda * (sum y - K)^2
    # Expand: lambda*(sum y)^2 - 2*lambda*K*(sum y) + lambda*K^2
    # Constant lambda*K^2 can be ignored.
    # (sum y)^2 = sum y_i + 2*sum_{i<j} y_i y_j
    for i in range(T):
        Q[i, i] += lambda_k * (1 - 2 * K)   # from lambda*sum y_i  and -2lambdaK*sum y_i
        for j in range(i + 1, T):
            Q[i, j] += 2 * lambda_k         # from 2*lambda*y_i*y_j

    return Q, K
