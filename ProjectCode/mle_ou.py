import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import scipy.stats as stats
import statsmodels.api as sm


def crps_gaussian(mu, sigma2, y):
    """
    CRPS of a Normal(mu, sigma2) forecast against scalar y.
    """
    sigma = np.sqrt(sigma2)
    z     = (y - mu) / sigma
    pdf   = stats.norm.pdf(z)
    cdf   = stats.norm.cdf(z)
    return sigma * (z * (2*cdf - 1) + 2*pdf - 1/np.sqrt(np.pi))

def empirical_variogram(x, max_lag):
    """
    Compute the experimental semivariogram for lags 1…max_lag.
    Returns (lags, semivariance).
    """
    n = len(x)
    lags = np.arange(1, max_lag + 1)
    gamma = np.zeros_like(lags, dtype=float)
    for i, h in enumerate(lags):
        diffs = x[h:] - x[:-h]
        gamma[i] = 0.5 * np.mean(diffs**2)
    return lags, gamma


def OrnsteinSimulated(mu, theta, sigma, dt, X0, N=750):
    t = np.linspace(0, (N-1)*dt, N)
    X = np.zeros(N)
    X[0] = X0
    dB = np.sqrt(dt) * np.random.normal(0, 1, N-1)
    for i in range(1, N):
        drift    = theta * (mu - X[i-1]) * dt
        diffusion = sigma * dB[i-1]
        X[i] = X[i-1] + drift + diffusion
    return X, t

def negative_log_likelihood(params, S, dt):
    mu, sigma, theta = params
    if sigma <= 0 or theta <= 0:
        return np.inf
    S_t   = S[:-1]
    S_tp1 = S[1:]
    e_term = np.exp(-theta * dt)
    M      = S_t * e_term + mu * (1 - e_term)
    V      = sigma**2 / (2 * theta) * (1 - np.exp(-2 * theta * dt))
    n = len(S_t)
    ll = -0.5 * n * np.log(2 * np.pi * V) - 0.5 * np.sum((S_tp1 - M)**2 / V)
    return -ll

def simple_ornstein(csv_file="priceData.csv"):
    data_raw = pd.read_csv(csv_file, sep=";")

    prices   = data_raw["Spot.price"].values
    print(prices.shape)
    prices   = prices[~np.isnan(prices)]
    hours    = np.arange(len(prices))
    dt       = 1.0
    initial_guess = [np.mean(prices), np.std(prices), 1.0]
    bounds        = [(None, None), (1e-12, None), (1e-12, None)]
    res = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(prices, dt),
        method='L-BFGS-B',
        bounds=bounds
    )
    mu_hat, sigma_hat, theta_hat = res.x
    print(res.x)
    X0     = prices[0]
    sim_path, sim_time = OrnsteinSimulated(mu_hat, theta_hat, sigma_hat, dt, X0, N=len(prices))
    return hours, prices, sim_path, mu_hat, sigma_hat, theta_hat

def run_ou_estimation(prices, hours, sim_path, mu_hat, sigma_hat, theta_hat, dt=1.0):
    residuals = prices - sim_path

    # 1) Observed vs Simulated
    plt.figure(figsize=(10, 4))
    plt.plot(hours, prices,    'o-', label='Observed')
    plt.plot(hours, sim_path, 's--', label='Simulated OU')
    plt.title('Observed vs. Simulated OU Process')
    plt.xlabel('Time Index')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Residual histogram
    axes[0].hist(residuals, bins=30, edgecolor='black')
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_title("In‐Sample Residuals")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    
    # 2) QQ‐plot
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("QQ‐Plot of Residuals")
    axes[1].get_lines()[1].set_color('red')

    # 3) ACF plot
    sm.graphics.tsa.plot_acf(residuals, lags=24, ax=axes[2])
    axes[2].set_title("ACF of Residuals")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrelation")
    axes[2].set_ylim([-1, 1.1])

    plt.tight_layout()
    plt.show()

    e_term = np.exp(-theta_hat * dt)
    M      = sim_path[:-1] * e_term + mu_hat * (1 - e_term)
    V      = sigma_hat**2 / (2 * theta_hat) * (1 - np.exp(-2 * theta_hat * dt))
    crps_vals = [crps_gaussian(M[i], V, prices[i+1]) for i in range(len(M))]
    print(f"Mean one-step OU CRPS: {np.mean(crps_vals):.4f}")

    # 5) Residual variogram
    max_lag = 24
    lags_var, gamma_var = empirical_variogram(residuals, max_lag)
    plt.figure(figsize=(10, 4))
    plt.plot(lags_var, gamma_var, 'o-')
    plt.title('Residual Variogram')
    plt.xlabel('Lag (hours)')
    plt.ylabel('Semivariance')
    plt.tight_layout()
    plt.show()
    print(gamma_var)

if __name__ == '__main__':
    hours, prices, sim_path, mu_hat, sigma_hat, theta_hat = simple_ornstein()
    run_ou_estimation(prices, hours, sim_path, mu_hat, sigma_hat, theta_hat)
