import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1       # Drift (expected return)
sigma = 0.2    # Volatility
S0 = 100       # Initial stock price
T = 1.0        # Time horizon (1 year)
N = 252        # Number of time steps (daily steps for a year)
dt = T / N     # Time step size

# Initialize arrays
t = np.linspace(0, T, N)
S = np.zeros(N)
S[0] = S0

# Simulate the process
for i in range(1, N):
    dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
    S[i] = S[i-1] + mu * S[i-1] * dt + sigma * S[i-1] * dW

# Plotting the simulated path
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Simulated GBM Path')
plt.title('Geometric Brownian Motion Simulation')
plt.xlabel('Time (Years)')
plt.ylabel('Asset Price')
plt.legend()
plt.grid(True)
plt.show()
