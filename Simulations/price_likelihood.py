import numpy as np
from scipy.optimize import minimize
from stock import get_data, compute_drift_and_diffusion
import matplotlib.pyplot as plt
import pandas as pd


def simulate_gbm_show(S0=None, mu=None, sigma=None, T=1.0, N=None, seed=None):
    """
    Simulates a GBM path with given parameters.
    Returns array of length N+1 (including time 0).
    """
    if seed != None:
        np.random.seed(seed)
    dt = T / N
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        Z = np.random.normal(0, 1)
        S[i] = S[i-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    return S


def negative_log_likelihood(params, S, dt):
    """
    Returns the NEGATIVE log-likelihood for the increments of S.
    params = [mu, sigma].
    S: array of shape (N+1,), i.e. S[0], S[1], ..., S[N].
    dt: time step size T/N.
    """
    mu, sigma = params
    # Avoid negative or zero sigma
    if sigma <= 0:
        return np.inf

    # Compute log-increments X_i = log(S_{i+1}) - log(S_i)
    X = np.diff(np.log(S))

    # Theoretical mean of X_i
    mean_increment = (mu - 0.5*sigma**2) * dt

    # Theoretical variance of X_i
    var_increment = sigma**2 * dt

    # Sum log-likelihood of each increment
    # log-likelihood for X_i ~ Normal(mean_increment, var_increment)
    # ll_i = -0.5 * [ log(2*pi*var) + (X_i - mean)^2 / var ]
    n = len(X)
    ll = -0.5 * n * np.log(2 * np.pi * var_increment) \
         - 0.5 * np.sum((X - mean_increment)**2 / var_increment)

    # We return NEGATIVE log-likelihood
    return -ll



#____________________________________________________________________

# Test here
#____________________________________________________________________



data_raw = pd.read_csv("priceData.csv", sep=";")
df = data_raw["Spot.price"]

prices = np.abs(df.values[12000:12750])
prices = prices[~np.isnan(prices)]+1e-10

hours = df.index[12000:12750]


initial_guess = [0.2, 0.4]    # e.g. some guess
dt = 1

# 3) Optimize
bounds = [(None, None), (1e-12, None)]  # keep sigma > 0
result = minimize(
    fun=negative_log_likelihood,
    x0=initial_guess,
    args=(prices, dt),
    method='L-BFGS-B',
    bounds=bounds
)

mu_hat, sigma_hat = result.x
print("Estimated mu =", mu_hat)
print("Estimated sigma =", sigma_hat)

output = simulate_gbm_show(S0=0.49,mu=0.4, sigma=sigma_hat, N=len(prices))
print(prices.shape, output[:-1].shape, hours[:-2].shape)
plt.plot(hours, prices)
plt.plot(hours, output[:-1])
plt.show()

