# Ornstein_prices.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# If you want to produce a Plotly figure, import plotly:
import plotly.graph_objs as go

def OrnsteinSimulated(mu, theta, sigma, dt, X0, N=750):
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
    if sigma <= 0:
        return np.inf
    S_t = S[:-1]
    S_tp1 = S[1:]
    e_term = np.exp(-theta*dt)
    M = S_t*e_term + mu*(1 - e_term)
    V = (sigma**2)/(2.0*theta)*(1.0 - np.exp(-2.0*theta*dt))
    n = len(S_t)
    ll = (
        -0.5 * n * np.log(2.0*np.pi*V)
        - 0.5 * np.sum((S_tp1 - M)**2 / V)
    )
    return -ll

def run_ou_estimation(csv_file="priceData.csv"):
    """
    Runs the OU parameter estimation using data from `csv_file`.
    Returns:
      fig: a Plotly figure comparing real vs. simulated data
      param_text: a string describing the fitted parameters
    """
    # 1) Read CSV data
    data_raw = pd.read_csv(csv_file, sep=";")
    df_prices = data_raw["Spot.price"]

    # 2) Slice data, drop NaNs
    prices = df_prices.values
    prices = prices[~np.isnan(prices)]
    hours = df_prices.index
    hours = hours[:len(prices)]  # ensure same length

    # 3) MLE
    dt = 1.0  # 1 hour steps
    initial_guess = [2, 2, 2]  # [mu, sigma, theta]
    bounds = [
        (None, None),    # mu
        (1e-12, None),   # sigma > 0
        (None, None)     # theta
    ]

    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(prices, dt),
        method='L-BFGS-B',
        bounds=bounds
    )

    mu_hat, sigma_hat, theta_hat = result.x

    # 4) Simulate OU path
    X0 = prices[0] if len(prices) > 0 else 0.4
    Nsim = len(prices)
    sim_path, sim_time = OrnsteinSimulated(
        mu=mu_hat,
        theta=theta_hat,
        sigma=sigma_hat,
        dt=dt,
        X0=X0,
        N=Nsim
    )

    # 5) Build a Plotly figure comparing real vs. simulated
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, 
        y=prices, 
        mode='lines+markers', 
        name='Observed Prices'
    ))
    fig.add_trace(go.Scatter(
        x=hours, 
        y=sim_path, 
        mode='lines+markers', 
        name='Simulated OU'
    ))
    fig.update_layout(title="Ornstein–Uhlenbeck Fit Comparison")

    # Generate parameter text
    param_text = (
        f"Fitted mu = {mu_hat:.4f}, "
        f"sigma = {sigma_hat:.4f}, "
        f"theta = {theta_hat:.4f}"
    )

    return fig, param_text


# If you want to test this file by itself, you can still do so:
if __name__ == "__main__":
    # Just test the function here (but it won’t produce a Dash figure):
    data_raw = pd.read_csv("priceData.csv", sep=";")
    print(data_raw.columns)
    df_prices = data_raw["Electricity.Co2.Emission"]    
    print(df_prices[:10])

    # fig, params = run_ou_estimation("priceData.csv")
    # print(params)
    # If you want to display a Plotly figure in a pop-up:
    # fig.show()
