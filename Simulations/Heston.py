import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Heston model parameters
S0 = 100.0     # Initial asset price
V0 = 0.04      # Initial variance (volatility^2)
mu = 0.05      # Drift of the asset
kappa = 2.0    # Mean-reversion speed
theta = 0.04   # Long-term variance
sigma = 0.3    # Volatility of volatility
rho = -0.7     # Correlation between dW1 and dW2
T = 1.0        # Time horizon (1 year)
N = 252        # Number of time steps (e.g., trading days in a year)
dt = T / N

# Preallocate arrays for S and V
S = np.zeros(N+1)
V = np.zeros(N+1)
t_grid = np.linspace(0, T, N+1)

# Set initial values
S[0] = S0
V[0] = V0

# Generate standard normal increments
Z1 = np.random.normal(size=N)
Z2 = np.random.normal(size=N)

# Apply Cholesky to introduce correlation between W1 and W2
# We want corr(W1, W2) = rho, so we build a 2x2 covariance matrix
# [1,   rho
#  rho, 1  ]
# Then do matrix multiplication to create correlated increments
correlation_matrix = np.array([[1.0,     rho],
                               [rho, 1.0]])
L = np.linalg.cholesky(correlation_matrix)

# Transform the independent Z1, Z2 into correlated increments
W = L @ np.array([Z1, Z2])

# Extract correlated increments
dW1 = W[0, :]
dW2 = W[1, :]

# Euler–Maruyama simulation
for i in range(N):
    # Variance process
    V[i+1] = (V[i]
              + kappa * (theta - V[i]) * dt
              + sigma * np.sqrt(max(V[i], 0.0)) * np.sqrt(dt) * dW2[i])
    
    # Enforce non-negativity to avoid numerical issues
    V[i+1] = max(V[i+1], 0.0)
    
    # Asset price process
    S[i+1] = (S[i]
              + mu * S[i] * dt
              + np.sqrt(max(V[i], 0.0)) * S[i] * np.sqrt(dt) * dW1[i])

# Plot the paths
plt.figure()
plt.plot(t_grid, S)
plt.title("Simulated Asset Price Under Heston Model")
plt.xlabel("Time")
plt.ylabel("Asset Price")
plt.show()

plt.figure()
plt.plot(t_grid, V)
plt.title("Simulated Variance Under Heston Model")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.show()
