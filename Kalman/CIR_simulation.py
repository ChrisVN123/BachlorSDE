import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(2341)

# CIR simulation function
def simulate_cir(kappa, theta, sigma, x0, dt, n):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + kappa * (theta - x[t-1]) * dt + sigma * np.sqrt(max(x[t-1], 0)) * np.sqrt(dt) * np.random.normal()
    return x

# Common settings
dt = 1
n = 200
x0 = 0

# Parameter sets chosen to separate the series
param_sets = [
    (0.7, 1.0, 0.5, 1.0),
    (0.7, 7.0, 0.5, 7.0),
]

# Plotting
plt.figure(figsize=(10, 6))
for kappa, theta, sigma, x0 in param_sets:
    path = simulate_cir(kappa, theta, sigma, x0, dt, n)
    plt.plot(path, label=f"θ={kappa}, mu={theta}, σ={sigma}")

plt.title("Simulation of CIR Paths")
plt.xlabel("Time Step")
plt.ylabel("Process Value")
plt.legend()
plt.show()
