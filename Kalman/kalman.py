import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm



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

def optimal_kalman(dt, initial_guess, train_prices, plotting):
    bounds = [(1e-5, None)] * 4  # enforce positive parameters

    res = minimize(negative_log_likelihood, initial_guess, args=(train_prices, dt), bounds=bounds)
    theta_hat = res.x

    print("Estimated parameters:")
    print(f"theta_1 (mean reversion): {theta_hat[0]:.4f}")
    print(f"theta_2 (long-term mean): {theta_hat[1]:.4f}")
    print(f"theta_3 (volatility):     {theta_hat[2]:.4f}")
    print(f"sigma (obs noise std):    {theta_hat[3]:.4f}")


    _, x_filtered = kalman_filter(train_prices, theta_hat, dt)

    if plotting:

        plt.plot(train_prices, label='Observed Prices')
        plt.plot(x_filtered, label='Filtered State (OU estimate)', linestyle='--')
        plt.legend()
        plt.title("Observed vs Filtered Electricity Prices")
        plt.show()

    return x_filtered, theta_hat


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

    train_prices = interpolated_prices[:-24]
    test_prices = interpolated_prices[-24:]
    return train_prices, test_prices, prices


def main(model, forecast):
    dt = 1/24
    initial_guess = np.array([0.5, 0.3, 1, 0.12])
    train_prices, test_prices, prices = get_data(start=5000, end=6000)

    # From Kalman output
    x_filtered, theta_hat = optimal_kalman(dt, initial_guess, train_prices, plotting=False)

    last_x = x_filtered[-1]
    last_P = 1  # or use final filtered P_k if tracked

    # Forecast next 24 hours (1 day)
    x_future, P_future = forecast_ou(theta_hat, last_x, last_P, 24, dt)

    if model:
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=False)

        # First plot: Kalman Filter vs Train
        axs[0].plot(train_prices, label='Train Prices', marker='o')
        axs[0].plot(x_filtered, label='OU Filtered (Kalman)', linestyle='--')
        axs[0].set_title('Train Data vs OU Kalman Filter')
        axs[0].set_xlabel('Time (Hourly)')
        axs[0].set_ylabel('Electricity Price')
        axs[0].legend()
        axs[0].grid(True)

        # Second plot: All prices
        axs[1].plot(prices, label='All Prices (Train + Test)', marker='o')
        axs[1].set_title('Full Price Series')
        axs[1].set_xlabel('Time (Hourly)')
        axs[1].set_ylabel('Electricity Price')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()


    if forecast:
        #we define the seasonal period (24 hours)
        seasonal_period = 24

        # Fit SARIMA: (p,d,q)x(P,D,Q,s)
        # Here’s a common starting point: SARIMA(1,1,1)x(1,1,1,24)
        model = sm.tsa.SARIMAX(train_prices[-48:], order=(1,1,1), seasonal_order=(1,1,1,seasonal_period))
        results = model.fit()

        # Forecast 24 steps ahead
        forecast = results.get_forecast(steps=24)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()  # This may return a NumPy array
        lower = conf_int[:, 0] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 0]
        upper = conf_int[:, 1] if isinstance(conf_int, np.ndarray) else conf_int.iloc[:, 1]

        from scipy.optimize import minimize_scalar

        # Define the objective function: mean squared error between combined forecast and test data
        def objective(alpha):
            combined = alpha * x_future + (1 - alpha) * forecast_mean
            return np.mean((combined - test_prices) ** 2)

        # Minimize the objective function in the interval [0, 1]
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        optimal_alpha = result.x
        min_mse = result.fun

        print(f"\nOptimal alpha: {optimal_alpha:.4f}")
        print(f"Minimum MSE: {min_mse:.4f}")

        # Plot the optimal combined forecast
        optimal_combined = optimal_alpha * x_future + (1 - optimal_alpha) * forecast_mean

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


main(model=True, forecast=True)
