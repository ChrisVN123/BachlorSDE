# Import libraries
import matplotlib.pyplot as plt
import scipy.optimize as spop
import numpy as np
import pandas as pd

# Importing and preprocessing data
data = pd.read_csv('/Users/marcusring/Desktop/KU/Tredje år/DatFin/Python/Bahcelor_GARCH/priceDatacopy.csv',
                   delimiter=';')
data_filtered = data["Spot.price"].dropna()
data_filtered = data_filtered[data_filtered > 0]

# Interpolation
data_filtered = data_filtered.interpolate(method='linear')

# Printing dataset
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(data_filtered, color='Red', linewidth=1)
ax.set(title='Electricity prices', ylabel='Price')
plt.show()

# Calculating log returns
returns = np.log(np.array(data_filtered)[1:]/np.array(data_filtered)[:-1])

# Printing returns
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(returns, color='Red', linewidth=1)
ax.set(title='Returns', ylabel='Price')
plt.show()

# Starting parameters
mean = np.average(returns)
var = np.std(returns) ** 2

# Creating theoretical GARCH function
def garch_mle(params):
    mu, omega, alpha, beta = params
    # Ensuring positive parameters
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    residuals = returns - mu
    realised_vol = np.abs(residuals)
    conditional_vol = np.zeros(len(returns))
    # Implementation of the long-run volatility
    long_run = np.sqrt(omega / (1 - alpha - beta))
    conditional_vol[0] = max(long_run, 1e-8)  # Prevents zero volatility
    for i in range(1, len(returns)):
        conditional_vol[i] = np.sqrt(
            omega + alpha * residuals[i - 1] ** 2 + beta * conditional_vol[i - 1] ** 2
        )
        conditional_vol[i] = max(conditional_vol[i], 1e-8)  # Prevent zeros
    # Implementing the log-likelihood
    lh = (1 / (np.sqrt(2 * np.pi) * conditional_vol)) * np.exp(-realised_vol ** 2 / (2 * conditional_vol ** 2))
    # Handles very small likelihoods
    lh = np.where(lh <= 1e-12, 1e-12, lh)
    # Total log likelihood
    llh = np.sum(np.log(lh))
    return -llh

# Maximizing (minimizing) log-likelihood
result = spop.minimize(garch_mle, [mean, var, 0, 0], method='Nelder-Mead')

# Retrieving optimal parameters
params = result.x
mu = result.x[0]
omega = result.x[1]
alpha = result.x[2]
beta = result.x[3]
llh = -float(result.fun)

# Calculating the realised and conditional volatility for optimal parameters
long_run = (omega / (1 - alpha - beta)) ** (1 / 2)
residuals = returns - mu
realised_vol = abs(residuals)
conditional_vol = np.zeros(len(returns))
conditional_vol[0] = long_run
for i in range (1, len(returns)):
    conditional_vol[i] = (omega + alpha * residuals[i-1] ** 2 + beta * conditional_vol[i-1] ** 2) ** (1/2)

# Printing estimated parameters
print("GARCH model parameters")
print(f"mu:{mu}")
print(f"omega:{omega}")
print(f"alpha:{alpha}")
print(f"beta:{beta}")
print(f"persistence:{alpha + beta}")

# Plotting results
plt.figure(figsize=(12,6))
plt.rc('xtick', labelsize=10)
plt.plot(data_filtered.index[1:], realised_vol, label='Realised Volatility', linewidth=1)
plt.plot(data_filtered.index[1:], conditional_vol, label='Conditional Volatility (GARCH)', linewidth=1, color='Red')
plt.title("GARCH(1,1) Volatility Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


