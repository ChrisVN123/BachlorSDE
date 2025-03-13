import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

def plot_full_history(df):
    """Plot the full historical closing prices."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label=f"Historical {df.name if hasattr(df, 'name') else ''} Close")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Historical Close Prices")
    plt.legend()
    plt.show()

def compute_drift_and_diffusion(prices, delta_t):
    """
    Compute the drift (μ) and diffusion (σ) from a stock price timeseries using log returns.
    
    Parameters:
        prices (np.array): Array of stock prices.
        delta_t (float): Time increment between observations.
    
    Returns:
        mu (float): Estimated drift (per unit time).
        sigma (float): Estimated volatility (per sqrt(time)).
    """
    # Compute log prices and their differences (log returns)
    log_prices = np.log(prices)
    dlog = np.diff(log_prices)  # length is len(prices) - 1
    n = dlog.size

    # Compute drift: μ ≈ (1/(n * Δt)) Σ Δ(log S)
    mu = np.sum(dlog) / (n * delta_t)

    # Compute volatility (diffusion): σ ≈ sqrt((1/(n * Δt)) Σ (Δ(log S) − μΔt)^2)
    sigma = np.sqrt(np.sum((dlog - mu * delta_t)**2) / (n * delta_t))
    return mu, sigma

def simulate_forecast(mu, sigma, historical_prices, forecast_steps, T=1, N=100):
    """
    Run N GBM simulations over the forecast horizon, compute the average simulated path,
    and plot it together with the true historical values.
    
    Parameters:
        mu (float): Drift parameter.
        sigma (float): Diffusion parameter.
        historical_prices (np.array): Array of historical stock prices.
        forecast_steps (int): Number of forecast steps.
        T (float): Total forecast time horizon (in the same time units as delta_t).
        N (int): Number of simulation paths.
    """
    dt = T / forecast_steps
    # Simulation starting point: price forecast_steps back from the end.
    start_index = -forecast_steps
    initial = historical_prices[start_index]

    # Pre-allocate array for simulations: shape (N, forecast_steps)
    simulations = np.zeros((N, forecast_steps))
    simulations[:, 0] = initial  # All simulations start at the same initial price

    # Set seed for reproducibility.
    np.random.seed(42)
    
    # Run N simulations.
    for i in range(N):
        # Generate Brownian increments for this simulation.
        dW = np.random.normal(0, np.sqrt(dt), size=forecast_steps)
        for t in range(1, forecast_steps):
            simulations[i, t] = simulations[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t-1])
    
    # Compute the average simulated path.
    avg_simulation = np.mean(simulations, axis=0)
    
    # Extract the true historical values for the forecast period.
    true_values = historical_prices[start_index:]
    
    # Create a timeline for the forecast period.
    time_axis = np.arange(forecast_steps)
    
    # Plot both the average simulated path and the true historical values.
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, true_values, label="True Historical", linestyle="--", marker="x")
    plt.plot(time_axis, avg_simulation, label=f"Average GBM (N={N})", marker="o")
    plt.xlabel("Forecast Steps")
    plt.ylabel("Stock Price")
    plt.title("Average GBM Forecast vs. True Historical Values")
    plt.legend()
    plt.show()

def get_data(ticker,num_years):
    """
    Fetch historical data for the given ticker over the past 5 years.
    
    Returns:
        prices (np.array): Closing prices as a NumPy array.
        df (DataFrame): The full historical data.
    """
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=f"{num_years}y")
    df.index = pd.to_datetime(df.index)
    # Optionally, set a name for the dataframe for labeling purposes.
    df.name = ticker.upper()
    prices = df["Close"].to_numpy()
    return prices, df


# prices_numpy, df = get_data("AAPL")
# #print(prices_numpy)

# # Define the time increment between observations.
# delta_t = 1  # adjust as needed (e.g., 1 for weekly or daily data)

# # Estimate the drift (mu) and volatility (sigma) using log returns.
# mu, sigma = compute_drift_and_diffusion(prices=prices_numpy, delta_t=delta_t)
# print("Estimated μ:", mu)
# print("Estimated σ:", sigma)

# # Define the forecast horizon: for example, forecast the last 100 steps.
# forecast_steps = 100
# T = forecast_steps * delta_t  # Total forecast time horizon

# # Run N simulations and plot the average GBM path with true historical values.
# simulate_forecast(mu, sigma, prices_numpy, forecast_steps, T, N=9)
