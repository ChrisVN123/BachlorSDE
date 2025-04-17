import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

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



import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Your data (make sure it's a 1D array)
# prices = np.abs(df_prices.values[23:232])  # or longer for seasonal modeling

# Define the seasonal period (24 hours)
seasonal_period = 24

# Fit SARIMA: (p,d,q)x(P,D,Q,s)
# Here’s a common starting point: SARIMA(1,1,1)x(1,1,1,24)
model = sm.tsa.SARIMAX(prices, order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
results = model.fit()

# Forecast 24 steps ahead
forecast = results.get_forecast(steps=24)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()  # This may return a NumPy array

# ✅ Fix:
lower = conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0]
upper = conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(np.abs(df_prices.values[23:232+24]), label='Observed')
plt.plot(np.arange(len(prices), len(prices)+24), forecast_mean, label='Forecast', linestyle='--')
plt.fill_between(np.arange(len(prices), len(prices)+24), lower, upper, color='gray', alpha=0.3)
plt.title('SARIMA Forecast (Next 24 Hours)')
plt.xlabel('Time (Hourly)')
plt.ylabel('Electricity Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





def discretize_ou(theta, dt):
    phi = np.exp(-theta[0]*dt)
    mu = theta[1]
    Q = theta[2]**2/(2*theta[0])*(1-np.exp(-2*theta[0]*dt))
    R = theta[3]**2
    return phi, mu, Q, R


def kalman_filter(y, theta, dt):
    n = y.shape[0]
    phi, mu, Q, R = discretize_ou(theta, dt)

    x_k = np.zeros(n)
    P_k = np.zeros(n)
    S_k = np.zeros(n)
    K_k = np.zeros(n)
    L_k = np.zeros(n)

    # Initial guess
    x_k[0] = y[0]
    P_k[0] = 1
    S_k[0] = P_k[0] + R
    K_k[0] = P_k[0] / S_k[0]
    L_k[0] = -0.5 * (np.log(2*np.pi) + np.log(S_k[0]) + ((y[0] - x_k[0])**2) / S_k[0])

    for i in range(1, n):
        # Prediction
        x_pred = phi * x_k[i-1] + (1 - phi) * mu
        P_pred = phi**2 * P_k[i-1] + Q

        # Update
        S_k[i] = P_pred + R
        K_k[i] = P_pred / S_k[i]
        x_k[i] = x_pred + K_k[i] * (y[i] - x_pred)
        P_k[i] = (1 - K_k[i]) * P_pred

        # Log-likelihood
        L_k[i] = -0.5 * (np.log(2*np.pi) + np.log(S_k[i]) + ((y[i] - x_pred)**2) / S_k[i])

    return L_k, x_k

def negative_log_likelihood(theta, y, dt):
    L_k, _ = kalman_filter(y, theta, dt)
    return -np.sum(L_k)

dt = 1/24
initial_guess = np.array([0.5,0.3,1,0.12])
bounds = [(1e-5, None)] * 4  # enforce positive parameters

res = minimize(negative_log_likelihood, initial_guess, args=(prices, dt), bounds=bounds)
theta_hat = res.x

print("Estimated parameters:")
print(f"theta_1 (mean reversion): {theta_hat[0]:.4f}")
print(f"theta_2 (long-term mean): {theta_hat[1]:.4f}")
print(f"theta_3 (volatility):     {theta_hat[2]:.4f}")
print(f"sigma (obs noise std):    {theta_hat[3]:.4f}")


_, x_filtered = kalman_filter(prices, theta_hat, dt)

plt.plot(prices, label='Observed Prices')
plt.plot(x_filtered, label='Filtered State (OU estimate)', linestyle='--')
plt.legend()
plt.title("Observed vs Filtered Electricity Prices")
plt.show()


def forecast_ou(theta, last_x, last_P, steps_ahead, dt):
    phi, mu, Q, R = discretize_ou(theta, dt)
    
    x_forecast = np.zeros(steps_ahead)
    P_forecast = np.zeros(steps_ahead)

    x_forecast[0] = phi * last_x + (1 - phi) * mu
    P_forecast[0] = phi**2 * last_P + Q

    for i in range(1, steps_ahead):
        x_forecast[i] = phi * x_forecast[i-1] + (1 - phi) * mu
        P_forecast[i] = phi**2 * P_forecast[i-1] + Q

    return x_forecast, P_forecast

# From Kalman output
_, x_filtered = kalman_filter(prices, theta_hat, dt)
last_x = x_filtered[-1]
last_P = 1  # or use final filtered P_k if you tracked it

# Forecast next 24 hours (1 day)
x_future, P_future = forecast_ou(theta_hat, last_x, last_P, 24, dt)

# Plot forecast
plt.plot(np.arange(len(prices)+24), np.abs(df_prices.values[23:232+24]), label="Observed")
plt.plot(np.arange(len(prices), len(prices) + 24), x_future, label="Forecast", linestyle="--")
plt.fill_between(np.arange(len(prices), len(prices) + 24),
                 x_future - 1.96 * np.sqrt(P_future),
                 x_future + 1.96 * np.sqrt(P_future),
                 alpha=0.2, label="95% CI")
plt.legend()
plt.title("OU Forecast: Next 24 Hours")
plt.show()
