import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data_raw = pd.read_csv("priceData.csv", sep=";")
df_prices = data_raw["Spot.price"]
df_dates = data_raw["t"]
df_workday = data_raw["dayOfWeek"]

# Slice & interpolate prices
prices = np.abs(df_prices.values[23:232])
dates = df_dates.values[23:232]

s = pd.Series(prices)
prices = s.interpolate(method='linear', limit_direction='both').to_numpy()

# Kalman Filter function for discrete-time OU process
def kalman_ou_filter(prices, mu, sigma, theta, dt):
    N = len(prices)
    phi = np.exp(-theta * dt)
    Q = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))  # process noise

    # Init arrays
    x_pred = np.zeros(N)
    P_pred = np.zeros(N)
    x_filt = np.zeros(N)
    P_filt = np.zeros(N)

    # Initial state
    x_filt[0] = prices[0]
    P_filt[0] = 1.0  # some uncertainty

    for t in range(1, N):
        # Prediction step
        x_pred[t] = phi * x_filt[t-1] + mu * (1 - phi)
        P_pred[t] = phi**2 * P_filt[t-1] + Q

        # Observation update step
        R = 1e-5  # small observation noise
        K = P_pred[t] / (P_pred[t] + R)  # Kalman gain
        x_filt[t] = x_pred[t] + K * (prices[t] - x_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]

    return x_filt, x_pred, P_filt[-1]

# OU parameters (manually defined or estimated beforehand)
mu = np.mean(prices)
sigma = 0.5
theta = 0.3
dt = 1/24

# Run Kalman filter
filtered_state, predicted_state, last_P = kalman_ou_filter(prices, mu, sigma, theta, dt)

# 1-day ahead prediction (24 hours)
phi = np.exp(-theta * dt)
Q = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

future_steps = 24
future_predictions = np.zeros(future_steps)
future_uncertainty = np.zeros(future_steps)

# Start from last filtered value
future_predictions[0] = phi * filtered_state[-1] + mu * (1 - phi)
future_uncertainty[0] = phi**2 * last_P + Q

for t in range(1, future_steps):
    future_predictions[t] = phi * future_predictions[t-1] + mu * (1 - phi)
    future_uncertainty[t] = phi**2 * future_uncertainty[t-1] + Q

# Build time axis
total_len = len(prices) + future_steps
time = np.arange(total_len)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(prices, label='Observed Prices', linewidth=1.5)
plt.plot(filtered_state, label='Filtered OU State', linewidth=2)
plt.plot(predicted_state, label='Predicted In-Sample', linestyle='--')
plt.plot(np.arange(len(prices), total_len), future_predictions, label='1-Day Ahead Prediction', linestyle='--', color='purple')
plt.xlabel('Time (hour index)')
plt.ylabel('Price')
plt.title('OU Kalman Filter with 1-Day Ahead Forecast')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
