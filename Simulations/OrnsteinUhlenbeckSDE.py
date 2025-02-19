import numpy as np
import matplotlib.pyplot as plt 
'''
This PDE is called Ornstein-Uhlenbeck process
The model is a "revert to the mean" model

It is simulated using the Euler-Maruyama simulation method
'''



# Parameters
theta = 0.7   # Mean reversion strength
mu = 1.0      # Long-term mean
sigma = 0.5   # Volatility
X0 = 0        # Initial condition
T = 100        # Total time
dt = 0.01     # Time step
N = int(T / dt)  # Number of steps

# Time grid
t = np.linspace(0, T, N)

# Initialize solution array
X = np.zeros(N)
X[0] = X0

# Simulate Brownian motion
dB = np.sqrt(dt) * np.random.normal(0, 1, N-1)

# Euler-Maruyama simulation
for i in range(1, N):
    drift = theta * (mu - X[i-1]) * dt
    diffusion = sigma * dB[i-1]
    X[i] = X[i-1] + drift + diffusion

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(t, X, label="Ornstein-Uhlenbeck Process")
plt.axhline(mu, color='red', linestyle='--', label="Long-term mean (μ)")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Simulation of an SDE (Ornstein-Uhlenbeck Process)")
plt.legend()
plt.show()
