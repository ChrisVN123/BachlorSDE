import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

def lamperti_negative_log_likelihood(params, Y, dt):
    """
    Vectorized negative log-likelihood for the Lamperti-transformed OU-SDE:
        dY = [theta * mu * exp(-Y) - theta - sigma^2/2]*dt + sigma*dW

    params = [theta, mu, sigma]
    Y      = log(price) array
    dt     = time step (assumed constant)
    """
    theta, mu, sigma = params
    
    # Penalize non-positive sigma
    if sigma <= 0.0:
        return np.inf

    # Compute the drift vector for all transitions
    # Note: For i=0,...,n-1 where n = len(Y)-1
    Y_curr = Y[:-1]
    Y_next = Y[1:]
    drift = theta * mu * np.exp(-Y_curr) - theta - 0.5 * sigma**2
    
    # Euler-Maruyama step: calculate the mean vector (M) of the transitions
    M = Y_curr + drift * dt
    
    # The variance is constant across all steps
    V = sigma**2 * dt
    
    # Calculate the log-likelihood contributions for all transitions at once
    # Each term: 0.5 * log(2*pi*V) + 0.5 * ((y_next - M)^2) / V
    nll = 0.5 * np.sum(np.log(2.0 * np.pi * V) + ((Y_next - M)**2) / V)
    
    return nll


def fit_multiplicative_ou_lamperti(csv_file="priceData.csv"):
    """
    1) Read the price data (assumed strictly positive).
    2) Transform via Y = log(S).
    3) Fit parameters [theta, mu, sigma] via maximum likelihood in the Y-domain.
    4) Return the original price series, log-transformed series, 
       estimated parameters and time step.
    """
    # Read data
    data_raw = pd.read_csv(csv_file, sep=";")
    # Ensure prices are positive and offset to avoid zeros or negatives
    spot_prices = data_raw["Spot.price"].dropna().values
    spot_prices = np.abs(spot_prices) + 1

    # Transform to log-domain
    Y = np.log(spot_prices)
    dt = 1 / (24 * 365)  # Hourly data converted to an annualized time step (approximately 1/8760)

    # Define bounds: ensuring theta, mu, and sigma are positive.
    bounds = [(1e-6, None),  # theta
              (1e-6, None),  # mu
              (1e-6, None)]  # sigma

    # Try multiple initial parameter guesses to reduce the risk of finding a local minimum
    best_res = None
    initial_guesses = [
        [0.1, 1.0, 0.1],
        [5.0, 1.6, 1.8],
        [5.0, 0.5, 0.5],
        [1.0, 0.1, 5.0],
        [7.0, 0.1, 7.0],
        # Add more if needed
    ]
    for guess in initial_guesses:
        res = minimize(
            lamperti_negative_log_likelihood,
            x0=guess,
            args=(Y, dt),
            method="L-BFGS-B",
            bounds=bounds
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    # Extract parameters from the best result
    theta_hat, mu_hat, sigma_hat = best_res.x
    
    return spot_prices, Y, theta_hat, mu_hat, sigma_hat, dt

def simulate_Y(theta, mu, sigma, Y0, dt, N):
    """
    Simulate the Lamperti-transformed process Y satisfying
      dY = [theta * mu * exp(-Y) - theta - 0.5*sigma^2]*dt + sigma*dW
    using Euler–Maruyama discretization.
    """
    Y = np.zeros(N)
    Y[0] = Y0
    for i in range(1, N):
        dW = np.sqrt(dt) * np.random.normal()
        drift = theta * mu * np.exp(-Y[i-1]) - theta - 0.5 * sigma**2
        Y[i] = Y[i-1] + drift * dt + sigma * dW
    return Y

def load_price_data(csv_file="priceData.csv"):
    """
    Utility function to load the price data from CSV.
    """
    data_raw = pd.read_csv(csv_file, sep=";")
    prices = data_raw["Spot.price"].dropna().values
    # Use a small constant to ensure positivity
    prices = np.abs(prices) + 1e-1
    return prices

# -------------------------- Main Script ------------------------------

# Fit the model to the data and retrieve parameters
spot_prices, Y_data, theta_hat, mu_hat, sigma_hat, dt = fit_multiplicative_ou_lamperti("priceData.csv")
N = len(spot_prices)
print("Optimal theta:", np.log(theta_hat))
print("Optimal mu:", np.log(mu_hat))
print("Optimal sigma:", np.log(sigma_hat))


print("Optimal mu:", np.mean(spot_prices))


# Simulate the log-price process using the estimated parameters
Y0 = Y_data[0]
Y_sim = simulate_Y(theta_hat, mu_hat, sigma_hat, Y0, dt, N)

# Convert simulated log-prices back to the price domain
S_sim = np.exp(Y_sim)

# Create a time index for plotting
time = np.arange(N)

# -------------------- Plotting --------------------
fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)

# Plot in the log-domain: Actual vs. Simulated Sample Path
axs[0].plot(time, Y_data, label='Actual Log-Prices')
axs[0].plot(time, Y_sim, label='Simulated Log-Prices', linestyle='--')
axs[0].set_title('Actual vs Simulated Log-Prices (OU Process)')
axs[0].legend()

# Plot in the original price domain: Actual vs. Simulated Price Path
axs[1].plot(time, spot_prices, label='Actual Prices')
axs[1].plot(time, S_sim, label='Simulated Prices', linestyle='--')
axs[1].set_title('Actual vs Simulated Prices')
axs[1].legend()

residuals = spot_prices - S_sim
axs[2].plot(residuals, label='Residals')
axs[2].set_title('Residual between simulation and true values')
axs[2].legend()

plt.xlabel('Time index')
plt.tight_layout()
plt.show()
