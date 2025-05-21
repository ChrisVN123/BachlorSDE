import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm
import pylab 
import scipy.stats as stats
from scipy.optimize import minimize_scalar


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
    residuals = np.zeros(n)


    # Initial guess
    x_k[0] = y[0]
    P_k[0] = 1
    S_k[0] = P_k[0] + R
    K_k[0] = P_k[0] / S_k[0]
    L_k[0] = -0.5 * (np.log(2*np.pi) + np.log(S_k[0]) + ((y[0] - x_k[0])**2) / S_k[0])
    residuals[0] = 0

    for i in range(1, n):
        # Prediction
        x_pred = phi * x_k[i-1] + (1 - phi) * mu
        P_pred = phi**2 * P_k[i-1] + Q

        residuals[i] = (y[i] - x_pred)
        # Update
        S_k[i] = P_pred + R
        K_k[i] = P_pred / S_k[i]
        x_k[i] = x_pred + K_k[i] * (y[i] - x_pred)
        P_k[i] = (1 - K_k[i]) * P_pred

        # Log-likelihood
        L_k[i] = -0.5 * (np.log(2*np.pi) + np.log(S_k[i]) + ((y[i] - x_pred)**2) / S_k[i])

    return L_k, x_k, residuals
def kf_ou_forecast_with_arima(train_prices, arima_forecast, dt, theta):
    """
    Linear Kalman filter for OU process + ARIMA-based pseudo-observations.
    
    train_prices : array of length n_train      -- true observations
    arima_forecast: array of length n_forecast  -- ARIMA mean forecasts
    dt            : float                       -- time step (e.g. 1/24)
    theta         : [kappa, mu, sigma, R]
    """
    # unpack and discretize
    phi, mu, Q, R = discretize_ou(theta, dt)
    
    n_train = len(train_prices)
    n_fore = len(arima_forecast)
    T      = n_train + n_fore

    x = np.zeros(T)
    P = np.zeros(T)

    # init
    x[0] = train_prices[0]
    P[0] = 1.0

    # — Train phase (use real data) —
    for t in range(1, n_train):
        # predict
        x_pred = phi * x[t-1] + (1-phi) * mu
        P_pred = phi**2 * P[t-1] + Q

        # update
        S = P_pred + R
        K = P_pred / S
        x[t] = x_pred + K * (train_prices[t] - x_pred)
        P[t] = (1 - K) * P_pred

    # — Forecast phase (use ARIMA forecasts as “observations”) —
    for i in range(n_fore):
        idx = n_train + i
        x_pred = phi * x[idx-1] + (1-phi) * mu
        P_pred = phi**2 * P[idx-1] + Q

        S = P_pred + R
        K = P_pred / S
        x[idx] = x_pred + K * (arima_forecast[i] - x_pred)
        P[idx] = (1 - K) * P_pred

    return x, P

def negative_log_likelihood(theta, y, dt):
    L_k, _ , residuals = kalman_filter(y, theta, dt)
    return -np.sum(L_k)

def optimal_kalman(dt, initial_guess, train_prices, plotting):
    bounds = [(1e-5, None)] * 4  # enforce positive parameters

    res = minimize(negative_log_likelihood, initial_guess, args=(train_prices, dt), bounds=bounds)
    theta_hat = res.x

    print("Estimated parameters:")
    print(f"theta_1 (mean reversion): {theta_hat[0]:.4f}")
    print(f"theta_2 (long-term mean): {theta_hat[1]:.4f}")
    print(f"theta_3 (volatility):     {theta_hat[2]:.4f}")
    print(f"sigma (obs noise std):    {theta_hat[3]:.4f}")


    _, x_filtered, residuals = kalman_filter(train_prices, theta_hat, dt)

    if plotting:

        plt.plot(train_prices, label='Observed Prices')
        plt.plot(x_filtered, label='Filtered State (OU estimate)', linestyle='--')
        plt.legend()
        plt.title("Observed vs Filtered Electricity Prices")
        plt.show()

    return x_filtered, theta_hat, residuals


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


def main(model=True, forecast=True):


    dt = 1/1000
    initial_guess = np.array([0.5, 0.3, 1, 0.12])
    train_prices, test_prices, prices, dates = get_data(start=12500, end=13500)

    # From Kalman output
    x_filtered, theta_hat, residuals = optimal_kalman(dt, initial_guess, train_prices, plotting=False)

    last_x = x_filtered[-1]
    last_P = 1  # or use final filtered P_k if tracked

    # Forecast next 24 hours (1 day)
    x_future, P_future = forecast_ou(theta_hat, last_x, last_P, 24, dt)

    # === Forecasting === #
    if forecast:

        # Create the side-by-side plots
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))

        # Plot histogram of residuals
        axes[0].hist(residuals, bins=30, edgecolor='black')
        axes[0].set_title('Residuals')

        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
        axes[1].set_xlabel('Theoretical Quantiles')
        axes[1].set_ylabel('Sample Quantiles')

        axes[2].acorr(residuals, maxlags=24)
        axes[2].set_title('ACF Plot')
        axes[2].set_xlim(0, 24)


        # Adjust layout and show
        plt.tight_layout()
        #plt.show()

        seasonal_period = 24
        model = sm.tsa.SARIMAX(train_prices[-96*3:], order=(0, 1, 1), seasonal_order=(0, 1, 0, seasonal_period))
        results = model.fit()

        # Forecast 24 steps ahead
        forecast = results.get_forecast(steps=24)
        arima_forecast = np.asarray(forecast.predicted_mean)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        lower = conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0]
        upper = conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1]

        theta = theta_hat[0]
        mu = theta_hat[1]
        sigma = theta_hat[2]
        R = theta_hat[3]
        # after fitting theta_hat via optimal_kalman...

        # get ARIMA forecast as before
        # arima_forecast = ...

        # run the OU‐KF + ARIMA
        x_combined_forecast, P_combined = kf_ou_forecast_with_arima(
            train_prices,
            arima_forecast,
            dt,
            theta_hat
        )

        # x_combined now contains: [filtered train states; 24‐step forecast]

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


        # Optimize alpha (weighting OU vs SARIMA)
        def objective(alpha):
            combined = alpha * x_future + (1 - alpha) * forecast_mean
            return np.mean((combined - test_prices) ** 2)

        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        optimal_alpha = result.x
        min_mse = result.fun

        print(f"\nOptimal alpha: {optimal_alpha:.4f}")
        print(f"Minimum MSE: {min_mse:.4f}")

        optimal_combined = optimal_alpha * x_future + (1 - optimal_alpha) * forecast_mean

        # Forecast plot
        plt.figure(figsize=(12, 6))
        plt.plot(test_prices, label='Actual (Test Data)', marker='o')
        plt.plot(x_future, label='OU Forecast', linestyle='--')
        plt.plot(forecast_mean, label='SARIMA Forecast', linestyle='--')
        plt.plot(optimal_combined, label=f'Optimal Combined (alpha={optimal_alpha:.2f})', linestyle='--')
        plt.title('Test Data vs Forecasts (Optimal Alpha)')
        plt.xlabel('Time (Hourly)')
        plt.ylabel('Electricity Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Model + Plotting === #
    if model:
    # === Final combined plot: Kalman on train + Combined forecast vs actual test ===

        # Step 1: Extend predictions
        combined_kalman = np.concatenate([x_filtered, optimal_combined])
        combined_observed = np.concatenate([train_prices, test_prices])

        # Step 2: X-axis = time index
        time = np.arange(len(combined_observed))

        # Step 3: Plot
        plt.figure(figsize=(14, 6))
        plt.plot(time, combined_observed, label='Observed Prices (Train + Test)', marker='o')
        plt.plot(time, combined_kalman, label='Kalman (Train) + Kalman+SARIMA Forecast (Test)', linestyle='--', color="red")

        # Add train/test split line
        plt.axvline(len(train_prices)-1, color='green', linestyle='--', label='Train/Test Split')

        plt.title('Kalman Filter Fit and Forecast with SARIMA Correction')
        plt.xlabel('Time (Hourly Index)')
        plt.ylabel('Electricity Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


main(model=True, forecast=True)
