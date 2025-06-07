import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(2341)

# OU simulation via Euler-Maruyama
def simulate_ou(theta, mu, sigma, X0, T, dt):
    N = int(T / dt)
    t = np.linspace(0, T, N)
    X = np.zeros(N)
    X[0] = X0
    increments = np.sqrt(dt) * np.random.normal(size=N-1)
    for i in range(1, N):
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * increments[i-1]
    return t, X

# Common simulation settings
T = 200
dt = 1

# Parameter sets: (theta, mu, sigma, X0)
param_sets = [
    (0.7, 1.0, 0.5, 1.0),
    (0.7, 7.0, 0.5, 7.0),
]

plt.figure(figsize=(10, 6))
for theta_i, mu_i, sigma_i, X0_i in param_sets:
    t, X = simulate_ou(theta_i, mu_i, sigma_i, X0_i, T, dt)
    plt.plot(t, X, label=f"θ={theta_i}, μ={mu_i}, σ={sigma_i}")


plt.title("Ornstein–Uhlenbeck Simulations with Distinct Parameters")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend(loc="upper left")
plt.ylim([-0.5,10])
plt.show()
