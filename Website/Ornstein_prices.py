# Ornstein_prices.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from statsmodels.tsa.stattools import acf

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

def simple_ornstein(csv_file="priceData.csv"):
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
    return hours, prices, sim_path, mu_hat, sigma_hat, theta_hat

def run_ou_estimation(prices, hours, sim_path, mu_hat, sigma_hat, theta_hat):
    """
    Runs the OU parameter estimation using data from `csv_file`.
    Returns:
      fig: a Plotly figure comparing real vs. simulated data
      param_text: a string describing the fitted parameters
    """

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



def plot_residuals_and_acf(prices, simulated_prices, lags):
    residuals = prices - simulated_prices

    # Fit normal and t-distribution
    mu, std = norm.fit(residuals)
    t_df, t_loc, t_scale = t.fit(residuals)

    # Histogram of residuals
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        histnorm='probability density',
        name='Residuals',
        opacity=0.6
    ))

    x_range = np.linspace(residuals.min(), residuals.max(), 500)
    hist_fig.add_trace(go.Scatter(
        x=x_range,
        y=norm.pdf(x_range, mu, std),
        mode='lines',
        name='Normal Fit'
    ))
    hist_fig.add_trace(go.Scatter(
        x=x_range,
        y=t.pdf(x_range, t_df, t_loc, t_scale),
        mode='lines',
        name='Student-t Fit'
    ))

    hist_fig.update_layout(title='Histogram of Residuals with Fitted Distributions')

    # ACF of original and simulated prices
    acf_prices = acf(prices, nlags=lags, fft=True)
    acf_simulated = acf(simulated_prices, nlags=lags, fft=True)

    acf_fig = go.Figure()
    acf_fig.add_trace(go.Bar(
        x=list(range(len(acf_prices))),
        y=acf_prices,
        name='Original Prices ACF'
    ))
    acf_fig.add_trace(go.Bar(
        x=list(range(len(acf_simulated))),
        y=acf_simulated,
        name='Simulated Prices ACF'
    ))

    acf_fig.update_layout(
        title='ACF Comparison',
        barmode='group',
        xaxis_title='Lag',
        yaxis_title='ACF'
    )
    return hist_fig, acf_fig


# hours, prices, sim_path, mu_hat, sigma_hat, theta_hat = simple_ornstein(csv_file="priceData.csv")

# # Plot 1: Line plot of true prices and sim_path
# plt.figure()
# plt.plot(hours, prices, label='Actual Prices')
# plt.plot(hours, sim_path, label='Simulated Path', linestyle='--')
# plt.xlabel('Hour')
# plt.ylabel('Price')
# plt.title('Actual vs Simulated Prices')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 2: ACF comparison
# acf_prices = acf(prices, nlags=40)
# acf_sim = acf(sim_path, nlags=40)

# plt.figure()
# plt.plot(acf_prices, label='ACF of Prices')
# plt.plot(acf_sim, label='ACF of Simulated Path', linestyle='--')
# plt.xlabel('Lag')
# plt.ylabel('ACF')
# plt.title('Autocorrelation Comparison')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Plot 3: Histogram of difference + Normal and t-distribution fit
# diff = prices - sim_path
# x = np.linspace(min(diff), max(diff), 1000)

# # Fit Normal
# mu_norm, std_norm = norm.fit(diff)
# pdf_norm = norm.pdf(x, mu_norm, std_norm)

# # Fit t-distribution
# df_t, loc_t, scale_t = t.fit(diff)
# pdf_t = t.pdf(x, df_t, loc=loc_t, scale=scale_t)

# plt.figure()
# plt.hist(diff, bins=40, density=True, alpha=0.6, label='Difference Histogram')
# plt.plot(x, pdf_norm, label=f'Normal Fit\nμ={mu_norm:.2f}, σ={std_norm:.2f}')
# plt.plot(x, pdf_t, label=f't Fit\ndf={df_t:.2f}, loc={loc_t:.2f}, scale={scale_t:.2f}')
# plt.title('Distribution of Price Differences\n(Actual - Simulated)')
# plt.xlabel('Difference')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()
