import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- OU model code ----

def simulate_ou(mu, theta, sigma, dt, X0, N=750):
    t = np.linspace(0, (N-1)*dt, N)
    X = np.zeros(N)
    X[0] = X0
    dB = np.sqrt(dt) * np.random.normal(0, 1, N-1)
    for i in range(1, N):
        drift = theta * (mu - X[i-1]) * dt
        diffusion = sigma * dB[i-1]
        X[i] = X[i-1] + drift + diffusion
    return X, t

def negative_log_likelihood(params, S, dt):
    mu, sigma, theta = params
    if sigma <= 0 or theta <= 0:
        return np.inf
    S_t = S[:-1]
    S_tp1 = S[1:]
    e_term = np.exp(-theta*dt)
    M = S_t * e_term + mu * (1 - e_term)
    V = (sigma**2)/(2.0*theta) * (1 - np.exp(-2.0*theta*dt))
    n = len(S_t)
    ll = (
        -0.5 * n * np.log(2.0 * np.pi * V)
        - 0.5 * np.sum((S_tp1 - M)**2 / V)
    )
    return -ll

# ---- Data prep ----

# 1) Read CSV data
data_raw = pd.read_csv("priceData.csv", sep=";")
df_prices = data_raw["Spot.price"]
df_workday = data_raw["dayOfWeek"]
df_dates = data_raw["t"]

# 2) Slice data, clean NAs by interpolation
prices = np.abs(df_prices.values[23:335])
dates = df_dates.values[23:335]
workday = df_workday.values[23:335]

# Interpolate missing prices
s = pd.Series(prices)
prices = s.interpolate(method='linear', limit_direction='both').to_numpy()

# 3) Split data into days
change_indices = np.where(workday[1:] != workday[:-1])[0] + 1
daily_prices = np.split(prices, change_indices)
daily_dates = np.split(dates, change_indices)

# ---- Predict each day using previous day's OU model ----
dt = 1.0  # hourly spacing
simulated_prices = []
ou_params = []

# Skip the first day (index 0)
for i in range(1, len(daily_prices)):
    prev_day_prices = daily_prices[i - 1]
    current_day_length = len(daily_prices[i])

    # Fit OU to previous day
    if len(prev_day_prices) < 2:
        simulated_prices.append(np.array([]))
        ou_params.append((np.nan, np.nan, np.nan))
        continue

    initial_guess = [np.mean(prev_day_prices), 1.0, 0.1]
    bounds = [(None, None), (1e-12, None), (1e-12, None)]
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(prev_day_prices, dt),
        method='L-BFGS-B',
        bounds=bounds
    )
    mu_hat, sigma_hat, theta_hat = result.x
    X0 = prev_day_prices[-1]
    sim_path, _ = simulate_ou(mu_hat, theta_hat, sigma_hat, dt, X0, current_day_length)

    simulated_prices.append(sim_path)
    ou_params.append((mu_hat, sigma_hat, theta_hat))

# ---- Plotting ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False, gridspec_kw={'height_ratios': [2.5, 1]})

# --- Plot 1: Prices and Predictions ---
ax1.plot(prices, label="Original Prices", linewidth=2)

for idx in change_indices:
    ax1.axvline(x=idx, color='red', linestyle='--', linewidth=0.8)

start_idx = len(daily_prices[0])  # skip first day
for i, sim in enumerate(simulated_prices):
    if len(sim) > 0:
        end_idx = start_idx + len(sim)
        ax1.plot(range(start_idx, end_idx), sim, label=f"Predicted Day {i+1}", alpha=0.7)
        start_idx = end_idx

xticks = np.concatenate(([0], change_indices))
xtick_labels = dates[xticks]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels, rotation=45, ha='right')

ax1.set_ylabel("Price")
ax1.set_title("OU Model Predictions (Each Day Predicted from Previous Day)")
ax1.grid(True)
ax1.legend()

# --- Plot 2: OU Parameters ---
mu_vals, sigma_vals, theta_vals = zip(*ou_params)

days = np.arange(1, len(ou_params) + 1)
ax2.plot(days, mu_vals, label="μ (mean reversion level)", marker='o')
ax2.plot(days, sigma_vals, label="σ (volatility)", marker='s')
ax2.plot(days, theta_vals, label="θ (mean reversion speed)", marker='^')

ax2.set_xlabel("Day Index (starting from 2nd day)")
ax2.set_ylabel("Parameter Value")
ax2.set_title("Fitted OU Parameters per Day")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
