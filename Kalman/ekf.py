import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pandas as pd
import scipy.stats as st
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Simulate CIR process
def simulate_cir(kappa, theta, sigma, x0, dt, n):
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + kappa * (theta - x[t-1]) * dt + sigma * np.sqrt(max(x[t-1], 0)) * np.sqrt(dt) * np.random.normal()
    return x

def get_data(start, end):
    # Load your data
    data_raw = pd.read_csv("priceData.csv", sep=";")
    df_prices = data_raw["Spot.price"]
    df_dates = data_raw["t"]
    df_workday = data_raw["dayOfWeek"]

    day = 24 # one day = 24 hours

    prices = np.abs(df_prices.values)
    dates = df_dates.values

    s = pd.Series(prices[start:end])
    interpolated_prices = s.interpolate(method='linear', limit_direction='both').to_numpy()

    # Do a train test split where the test set is just the following 24-hours
    train_prices = interpolated_prices[:-24]
    test_prices = interpolated_prices[-24:]
    return train_prices, test_prices, prices, dates

# EKF implementation
def ekf_cir(y, dt, kappa, theta, sigma, R):
    n = len(y)
    x_est = np.zeros(n)
    P      = np.zeros(n)
    mu_pred = np.zeros(n)   # μ_t|t-1
    S_pred  = np.zeros(n)   # Var(y_t | t-1)

    x_est[0], P[0] = y[0], 1.0
    loglik = 0.0

    for t in range(1, n):
        # –– prediction step
        x_p = x_est[t-1] + kappa*(theta - x_est[t-1])*dt
        F   = 1 - kappa*dt
        Q   = sigma**2 * max(x_est[t-1], 0) * dt
        P_p = F*P[t-1]*F + Q

        # record predictive distribution of y[t]
        mu_pred[t] = x_p
        S_pred[t]  = P_p + R

        # –– update step
        K = P_p / S_pred[t]
        x_est[t] = x_p + K*(y[t] - x_p)
        P[t]     = (1 - K)*P_p

        # –– log‐lik (unchanged)
        loglik += -0.5*(np.log(2*np.pi*S_pred[t]) + (y[t]-x_p)**2/S_pred[t])

    return x_est, loglik, mu_pred, S_pred


#minimize negative log-likelihood
def neg_log_likelihood(params, y, dt, R):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return np.inf
    _, log_likelihood,_,_ = ekf_cir(y, dt, kappa, theta, sigma, R)
    return -log_likelihood



def crps_gaussian(mu, sigma2, y):
    """
    CRPS of a Normal(mu, sigma2) forecast against scalar y.
    Closed‐form (Gneiting & Raftery, 2007).
    """
    sigma = np.sqrt(sigma2)
    z = (y - mu) / sigma
    pdf = st.norm.pdf(z)
    cdf = st.norm.cdf(z)
    return (
        sigma * (z * (2*cdf - 1) + 2*pdf - 1/np.sqrt(np.pi))
    )


# Parameters
n = 2500
dt = 1/n
np.random.seed(42)

# Simulate CIR process
train_prices, test_prices, prices ,dates = get_data(start=23,end=5000) #simulate_cir(true_kappa, true_theta, true_sigma, x0, dt, n)
R = 0.01  # Observation noise variance
y_obs = train_prices #+ np.random.normal(0, np.sqrt(R), size=n)


def empirical_ou_params(x, dt=1.0):
    """Moment-matching & regression OU parameter estimates.

    Parameters
    ----------
    x  : 1-D array-like
    dt : float, time step length in same units you want θ in
    """
    x = np.asarray(x, dtype=float)

    # ── 1. keep only rows where both x_k and x_{k+1} are observed ──
    valid_pairs = ~(np.isnan(x[:-1]) | np.isnan(x[1:]))
    x_lag   = x[:-1][valid_pairs]
    x_lead  = x[1:][valid_pairs]

    # ── 2. sample moments (using only non-NaN points) ──
    xbar = np.nanmean(x)                       # mean ignoring NaNs
    S2   = np.nanvar(x)                        # variance

    # lag-1 autocorr: use the filtered vectors
    r1   = np.corrcoef(x_lag, x_lead)[0, 1]

    # ── 3. MOM formulas ──
    theta_mom = -np.log(r1) / dt
    sigma_mom = np.sqrt(2 * theta_mom * S2)
    mu_mom    = xbar

    # ── 4. regression (Yule–Walker) ──
    reg = LinearRegression().fit(x_lag.reshape(-1, 1), x_lead)
    phi   = reg.coef_[0]
    c     = reg.intercept_
    theta_reg = -np.log(phi) / dt
    mu_reg    = c / (1 - phi)
    var_u     = np.mean((reg.predict(x_lag.reshape(-1, 1)) - x_lead) ** 2)
    sigma_reg = np.sqrt(2 * theta_reg * var_u / (1 - phi ** 2))

    return {
        "mom":  (mu_mom,   theta_mom,   sigma_mom),
        "reg":  (mu_reg,   theta_reg,   sigma_reg)
    }

print(empirical_ou_params(prices))


# Estimate parameters using EKF
initial_guess = [0.5, 1.0, 0.2]
bounds = [(1e-10, 100000), (1e-10, 100000), (1e-10, 100000)]
result = minimize(neg_log_likelihood, initial_guess, args=(y_obs, dt, R), bounds=bounds)

# Run EKF with estimated parameters
kappa_hat, theta_hat, sigma_hat = result.x
x_estimated, _, _, _ = ekf_cir(y_obs, dt, kappa_hat, theta_hat, sigma_hat, R)

print(f"Results give: {[kappa_hat, theta_hat, sigma_hat]}")  # Show estimated parameters






###################################################################################################
###################################################################################################
                                            #CRPS#
###################################################################################################
###################################################################################################
# fit parameters as before...
x_est, _, mu_pred, S_pred = ekf_cir(y_obs, dt, kappa_hat, theta_hat, sigma_hat, R)

# drop t=0 since we have no one-step-ahead pred
crps_vals = [crps_gaussian(mu_pred[t], S_pred[t], y_obs[t]) for t in range(1, len(y_obs))]


mean_crps = np.mean(crps_vals)
print(f"Mean CRPS (in‐sample EKF): {mean_crps:.4f}")


####################################################################################################
####################################################################################################

####################### FORCASTING THE NEXT 24-HOURS WITH ARIMA ####################################

####################################################################################################
####################################################################################################


from statsmodels.tsa.statespace.sarimax import SARIMAX

seasonal_period = 24
model = SARIMAX(train_prices[-48:], order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
results = model.fit()

forecast_object = results.get_forecast(steps=24)
arima_forecast = np.asarray(forecast_object.predicted_mean)


def ekf_cir_forecast_with_arima(train_prices, arima_forecast, dt, kappa, theta, sigma, R):
    n_train = len(train_prices)
    n_forecast = len(arima_forecast)
    total_steps = n_train + n_forecast

    x_est = np.zeros(total_steps)
    P = np.zeros(total_steps)
    residuals = np.zeros(n_train)

    x_est[0] = train_prices[0]
    P[0] = 1.0

    # Train phase (EKF updates with true data)
    for t in range(1, n_train):
        x_pred = x_est[t-1] + kappa * (theta - x_est[t-1]) * dt
        F = 1 - kappa * dt
        Q = sigma**2 * max(x_est[t-1], 0) * dt
        P_pred = F * P[t-1] * F + Q

        residuals[t] = train_prices[t] - x_pred
        S = P_pred + R
        K = P_pred / S
        x_est[t] = x_pred + K * (train_prices[t] - x_pred)
        P[t] = (1 - K) * P_pred

    # Forecast phase (EKF uses ARIMA forecast as observations)
    for t in range(n_forecast):
        i = n_train + t
        x_pred = x_est[i-1] + kappa * (theta - x_est[i-1]) * dt
        F = 1 - kappa * dt
        Q = sigma**2 * max(x_est[i-1], 0) * dt
        P_pred = F * P[i-1] * F + Q

        S = P_pred + R
        K = P_pred / S
        x_est[i] = x_pred + K * (arima_forecast[t] - x_pred)
        P[i] = (1 - K) * P_pred

        crps_vals.append(crps_gaussian(x_pred, S, test_prices[t]))


        #residuals[t] = arima_forecast[t] - x_pred  # one-step residual

    return x_est, P, residuals

# Apply the function
x_combined_forecast, P_combined_forecast, residuals = ekf_cir_forecast_with_arima(
    train_prices, arima_forecast, dt, kappa_hat, theta_hat, sigma_hat, R
)

# Plotting forecast using ARIMA as pseudo-observations in EKF
plt.figure(figsize=(14, 6))
plt.plot(np.concatenate([train_prices, test_prices]), label="Observed Prices (Train + Test)", marker='o')
plt.plot(np.concatenate([train_prices, arima_forecast]), label="ARIMA")
plt.plot(x_combined_forecast, label="EKF (Train) + ARIMA-Based Forecast (Test)", linestyle='--', color="purple")
plt.axvline(len(train_prices)-1, color='green', linestyle='--', label='Train/Test Split')
plt.title("CIR EKF Forecasting with ARIMA Correction (Used as Observations)")
plt.xlabel("Time (Hourly Index)")
plt.ylabel("Electricity Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

# … after you have your `residuals` array …

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1) Histogram
axes[0].hist(residuals, bins=30, edgecolor='black')
axes[0].axhline(0, color='black', linestyle='--')
axes[0].set_title("Residual Histogram")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Frequency")

# 2) QQ-plot
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("QQ-Plot")

# 3) ACF plot
sm.graphics.tsa.plot_acf(residuals, lags=30, ax=axes[2])
axes[2].set_title("ACF of Residuals")
axes[2].set_xlabel("Lag")
axes[2].set_ylabel("Autocorrelation")

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 4))
plt.hist(residuals, bins=30)
plt.axhline(0, color='black', linestyle='--')
plt.title("One-Step Forecast Residuals (ARIMA - EKF Prediction)")
plt.xlabel("Forecast Step")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()




###################################################################################################
###################################################################################################
                                        # VARIOGRAM #
###################################################################################################
###################################################################################################

def empirical_variogram(x, max_lag):
    """
    Compute the experimental semivariogram for lags 1…max_lag.
    Returns arrays (lags, gamma).
    """
    n = len(x)
    lags = np.arange(1, max_lag+1)
    gamma = np.zeros_like(lags, dtype=float)
    for i, h in enumerate(lags):
        diffs = x[h:] - x[:-h]
        gamma[i] = 0.5 * np.mean(diffs**2)
    return lags, gamma

# 1) Variogram of the *increments* of the original train_prices
increments = train_prices[1:] - train_prices[:-1]
lags, gamma_emp = empirical_variogram(increments, max_lag=50)

# 2) Theoretical OU variogram of increments at lag h:
#    γ_theory(h) = 0.5*Var[X(t+h)-X(t)] = (σ^2/(2κ))*(1 - exp(-κ*h*dt))
gamma_theory = (sigma_hat**2 / (2 * kappa_hat)) * (1 - np.exp(-kappa_hat * lags * dt))

plt.figure(figsize=(8,5))
#plt.plot(lags*dt, gamma_emp, 'o', label="Empirical variogram")
plt.plot(lags*dt, gamma_theory, '-', label="OU theory (fitted)")
plt.xlabel("Lag (time units)")
plt.ylabel("Semivariance")
plt.title("Increment variogram vs. OU theory")
plt.legend()
plt.show()

Var_inf = theta_hat * sigma_hat**2 / (2 * kappa_hat)
gamma_theo = Var_inf * (1 - np.exp(-kappa_hat * lags * dt))

plt.figure(figsize=(8,5))
plt.plot(lags*dt, gamma_emp, 'o', label="Empirical CIR Residual Variogram")
plt.plot(lags*dt, gamma_theo, '-', label="Theoretical CIR Semivariogram")
plt.xlabel("Lag (time units)")
plt.ylabel("Semivariance")
plt.title("CIR EKF Residual Variogram vs. Theory")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

errs = test_prices - x_combined_forecast[len(train_prices):]
l_e, g_e = empirical_variogram(errs,24)
plt.figure()
plt.plot(l_e, g_e, 'o-')
plt.title('Error Variogram (Out‐of‐Sample)')
plt.xlabel('Lag')
plt.ylabel('Semivariance')
plt.show()

# 3) Variogram of the EKF *residuals* to check for leftover structure
resid = train_prices - x_estimated
lags2, gamma_resid = empirical_variogram(resid, max_lag=50)
plt.figure(figsize=(8,5))
plt.plot(lags2*dt, gamma_resid, 'o-')
plt.xlabel("Lag")
plt.ylabel("Semivariance")
plt.title("Variogram of EKF residuals — should be flat (nugget)")
plt.show()
