import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100      # Initial stock price
mu = 0.05     # Drift (expected return)
sigma = 0.2   # Volatility
T = 1.0       # Total time in years
N = 252       # Number of time steps (daily steps if 252 trading days)
dt = T / N    # Time step size
np.random.seed(42)  # For reproducibility

# Generate Brownian motion (random increments)
dW = np.random.normal(0, np.sqrt(dt), size=N)  # Standard normal increments scaled by sqrt(dt)

# Simulate GBM path
S = np.zeros(N)
S[0] = S0  # Set initial condition

for t in range(1, N):
    S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])

# Plot the simulated path
plt.figure(figsize=(10,5))
plt.plot(S, label="Simulated GBM")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.title("Geometric Brownian Motion Simulation")
plt.legend()
plt.show()
