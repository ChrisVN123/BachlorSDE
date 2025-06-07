# Import libraries
import matplotlib.pyplot as plt
import scipy.optimize as spop
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats

# Importing and preprocessing data
data_raw = pd.read_csv('priceDatacopy.csv', sep=';')
df_prices = data_raw["Spot.price"].dropna()
df_prices = df_prices[df_prices > 0]

# Interpolation
df_prices = df_prices.interpolate(method='linear')

# Calculating log returns
returns = np.log(np.array(df_prices)[1:] / np.array(df_prices)[:-1])

# GARCH model fitting for train data
train_returns = returns[:-24]

# Creating empirical GARCH log-likelihood function
def garch_mle(params):
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return np.inf
    residuals = train_returns - mu
    cond_var = np.zeros(len(train_returns))
    cond_var[0] = np.var(train_returns)
    for t in range(1, len(train_returns)):
        cond_var[t] = omega + alpha * residuals[t-1]**2 + beta * cond_var[t-1]
    llh = -0.5 * (np.log(2 * np.pi) + np.log(cond_var) + residuals**2 / cond_var)
    return -np.sum(llh)

# Maximizing (minimizing) GARCH log-likelihood
initial_guess = [np.mean(train_returns), np.var(train_returns), 0.05, 0.9]
result = spop.minimize(garch_mle, initial_guess, method='Nelder-Mead')
mu, omega, alpha, beta = result.x
residuals_train = train_returns - mu
conditional_vol_train = np.zeros(len(train_returns))
conditional_vol_train[0] = np.sqrt(omega / (1 - alpha - beta))
for t in range(1, len(train_returns)):
    conditional_vol_train[t] = np.sqrt(
        omega + alpha * residuals_train[t-1]**2 + beta * conditional_vol_train[t-1]**2
    )

# Implementing the ARIMA model on returns
log_returns = returns
log_returns_sliced = log_returns[-100:]
# Using the statsmodel package to retrieve the SARIMA model
model = sm.tsa.SARIMAX(log_returns_sliced, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
results_arima = model.fit(disp=False)
arima_forecast = results_arima.get_forecast(steps=24)
arima_mean_forecast = arima_forecast.predicted_mean

# Simulation on combined ARIMA and GARCH
num_simulations = 1000
horizon = 24
simulated_paths = np.zeros((num_simulations, horizon))
last_sigma2 = conditional_vol_train[-1]**2
last_residual = residuals_train[-1]
np.random.seed(420)

# The combined ARIMA-GARCH model
for i in range(num_simulations):
    sigma2_t = last_sigma2
    res_t = last_residual
    for t in range(horizon):
        z = np.random.normal()
        sigma2_t = omega + alpha * res_t**2 + beta * sigma2_t
        sigma_t = np.sqrt(sigma2_t)
        r = arima_mean_forecast[t] + sigma_t * z
        simulated_paths[i, t] = r
        res_t = r - arima_mean_forecast[t]

# Computing mean and percentiles
mean_path = simulated_paths.mean(axis=0)
p05 = np.percentile(simulated_paths, 5, axis=0)
p95 = np.percentile(simulated_paths, 95, axis=0)

# GARCH-only 24-hour forecast
garch_forecast = np.zeros(horizon)
sigma2_t = conditional_vol_train[-1]**2
res_t = residuals_train[-1]
np.random.seed(420)
for t in range(horizon):
    sigma2_t = omega + alpha * res_t**2 + beta * sigma2_t
    sigma_t = np.sqrt(sigma2_t)
    r = 0 + sigma_t * np.random.normal()
    garch_forecast[t] = r
    res_t = r

# Plot with ARIMA+GARCH combined, GARCH-only, and actual returns
plt.figure(figsize=(12, 6))
plt.plot(mean_path, label="Combined ARIMA + GARCH forecast", color='blue')
plt.fill_between(range(horizon), p05, p95, color='blue', alpha=0.2, label="5%-95% Interval of the combined model")
plt.plot(garch_forecast, label="GARCH", color='red', linestyle='--')
plt.plot(log_returns[-24:], label="Actual returns (Last 24 hours)", color='black', linestyle=':')
plt.title("Forecasts: Combined ARIMA+GARCH, GARCH-only & actual returns (24 hours)")
plt.xlabel("Hour")
plt.ylabel("Log return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual analysis
# Garch residuals
residuals_train = train_returns - mu
standardized_residuals = residuals_train / conditional_vol_train

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Histogram
axs[0].hist(standardized_residuals, bins=30, color='blue', edgecolor='black', alpha=0.7)
axs[0].set_title("Histogram of GARCH standardized residuals")
axs[0].set_xlim(-10, 10)

# QQ plot
stats.probplot(standardized_residuals, dist="norm", plot=axs[1])
axs[1].set_title("QQ plot of GARCH standardized residuals")

# ACF plot
plot_acf(standardized_residuals, lags=24, alpha=0.05, ax=axs[2])
axs[2].set_title("ACF of GARCH standardized residuals")

plt.tight_layout()
plt.show()

# Combined ARIMA and GARCH residuals
true_returns = log_returns[-24:]
forecast_errors = simulated_paths.mean(axis=0) - true_returns
combined_residuals = forecast_errors

# SKill scores
# CRPS
def compute_crps(samples, true_value):
    samples = np.sort(samples)
    term1 = np.mean(np.abs(samples - true_value))
    term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return term1 - term2

# Using CRPS function on returns
true_returns = log_returns[-24:]

# Computing CRPS per time step
crps_scores = [compute_crps(simulated_paths[:, t], true_returns[t]) for t in range(horizon)]

# Mean of CRPS over 24 hours prediction
crps_total = np.mean(crps_scores)

# Printing CRPS
print("Final CRPS score (mean over 24 hours):", crps_total)

# Creating variogram function
def empirical_variogram(data, max_lag):
    max_valid_lag = len(data) - 1
    safe_lag = min(max_lag, max_valid_lag)
    return np.array([
        0.5 * np.mean((data[h:] - data[:-h])**2)
        for h in range(1, safe_lag + 1)
    ]), safe_lag

max_lag = 24

# Computing the variograms
resid_vario, lag_resid = empirical_variogram(standardized_residuals, max_lag)
resid_combined_vario, lag_combined_resid = empirical_variogram(combined_residuals, max_lag)
forecast_errors = simulated_paths.mean(axis=0) - true_returns
error_vario, lag_error = empirical_variogram(forecast_errors, max_lag)

# Creating subplot
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Plot residual GARCH variogram
axs[0].plot(np.linspace(0, 24, lag_resid), resid_vario, marker='o')
axs[0].set_title("Variogram of GARCH residuals")
axs[0].set_xlabel("Lag")
axs[0].set_ylabel("Semivariance")
axs[0].grid(True)

# Plot reisudal combined ARIMA-GARCH model variogram
axs[1].plot(np.linspace(0, 24, lag_combined_resid), resid_combined_vario, marker='o')
axs[1].set_title("Variogram of the combined model residuals")
axs[1].set_xlabel("Lag")
axs[1].set_ylabel("Semivariance")
axs[1].grid(True)

# Plot
plt.tight_layout()
plt.show()

# Stacking variograms into one array
variogram_scores = np.vstack([
    resid_vario[:min(len(resid_vario), len(resid_combined_vario), len(error_vario))],
    resid_combined_vario[:min(len(resid_vario), len(resid_combined_vario), len(error_vario))]
])

print(variogram_scores)

