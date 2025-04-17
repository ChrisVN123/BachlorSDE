import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from arch import arch_model

###############################################################################
# 1. OU Model Fitting and Simulation
###############################################################################

def simulate_ou(mu, theta, sigma, dt, X0, N=750):
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

def fit_ou_mle(dt,csv_file="priceData.csv"):
    data_raw = pd.read_csv(csv_file, sep=";")
    df_prices = data_raw["Spot.price"]
    prices = df_prices.dropna().values
    hours = df_prices.index[:len(prices)]
    initial_guess = [2, 2, 2]
    bounds = [(None, None), (1e-12, None), (None, None)]
    result = minimize(
        fun=negative_log_likelihood,
        x0=initial_guess,
        args=(prices, dt),
        method='L-BFGS-B',
        bounds=bounds
    )
    mu_hat, sigma_hat, theta_hat = result.x
    X0 = prices[0] if len(prices) > 0 else 0.4
    Nsim = len(prices)
    sim_path, sim_time = simulate_ou(mu_hat, theta_hat, sigma_hat, dt, X0, Nsim)
    return hours, prices, sim_path, mu_hat, sigma_hat, theta_hat

###############################################################################
# 2. GARCH on the Residuals + Time-Varying Volatility OU
###############################################################################

def compute_ou_residuals(prices, mu, sigma, theta, dt):
    """
    Given the fitted OU parameters (mu, sigma, theta) and the price series,
    return the standardized OU residuals.
    """
    e_term = np.exp(-theta * dt)  # e^(-theta*dt)
    # The OU model implies a constant one-step variance:
    V = (sigma**2)/(2.0*theta) * (1.0 - np.exp(-2.0*theta*dt))
    if V <= 0:
        raise ValueError("Computed OU variance is non-positive. Check parameters.")

    residuals = []
    for t in range(len(prices) - 1):
        S_t = prices[t]
        S_tp1 = prices[t+1]
        # Mean for step t->t+1
        M_t = S_t*e_term + mu*(1.0 - e_term)
        # Standardized residual
        r_t = (S_tp1 - M_t) / np.sqrt(V)
        residuals.append(r_t)
    
    return np.array(residuals)





def fit_garch(residuals):
    garch_model = arch_model(residuals, p=1, q=1, vol='GARCH', dist='normal')
    garch_fit = garch_model.fit(disp='off')
    omega = garch_fit.params['omega']
    alpha = garch_fit.params['alpha[1]']
    beta = garch_fit.params['beta[1]']
    return omega, alpha, beta

def simulate_GARCH(residuals, omega, alpha, beta):
    N = residuals.shape[0]
    sig = np.zeros(N)
    sig[0] = np.sqrt(omega / (1 - alpha - beta)) if (1 - alpha - beta) > 0 else np.sqrt(omega)
    for t in range(1, N):
        sig[t] = np.sqrt(omega + alpha * residuals[t-1]**2 + beta * sig[t-1]**2)
    return sig


def simulate_ou_garch(sigma, mu, theta, X0, N, omega, alpha, beta, dt):
    X = np.zeros(N)
    sig = np.zeros(N)
    eps = np.zeros(N)
    N = sigma.shape[0]
    
    dB = np.sqrt(dt) * np.random.normal(0, 1, N-1)

    X[0] = X0
    sig[0] = np.sqrt(omega / (1 - alpha - beta)) if (1 - alpha - beta) > 0 else np.sqrt(omega)
    for t in range(1, N):
        eps[t-1] = np.random.normal(0,np.sqrt(sig[t-1]))
        sig[t] = np.sqrt(omega + alpha * eps[t-1]**2 + beta * sig[t-1]**2)
        diffusion = sigma[t] * dB[t-1]

        X[t] = X[t-1] + theta * (mu - X[t-1]) * dt + diffusion
    return X, sig

###############################################################################
# 3. Main Script
###############################################################################

def main(csv_file='priceData.csv'):
    dt =  1 #/ (24 * 365)
    hours, prices, sim_path, mu_hat, sigma_hat, theta_hat = fit_ou_mle(dt, csv_file)
    print(f'[OU Params] mu={mu_hat:.4f}, sigma={sigma_hat:.4f}, theta={theta_hat:.4f}')
    X0 = prices[0]
    N = len(prices)
    ou_path = simulate_ou(mu_hat, theta_hat, sigma_hat, 1.0, X0, N)[0]
    residuals = prices - ou_path

    garch_model = arch_model(prices, p=1, q=1, o=1, vol='GARCH', dist='normal')
    garch_fit = garch_model.fit(update_freq=0, disp='off')
    print(garch_fit.summary())
    cond_vol = garch_fit.conditional_volatility


    omega_hat, alpha_hat, beta_hat = fit_garch(residuals)
    simulated_residuals = simulate_GARCH(residuals, omega_hat, alpha_hat, beta_hat)
    print(np.mean(simulated_residuals), np.var(simulated_residuals))
    ou_garch_path, sig_res = simulate_ou_garch(simulated_residuals, mu_hat, theta_hat, X0, N, omega_hat, alpha_hat, beta_hat, dt)

    fig, axs = plt.subplots(3, 1, figsize=(10,8), sharex=True)
    axs[0].plot(prices, label='Actual Prices')
    axs[0].plot(ou_path, label='OU Deterministic Path')
    axs[0].set_title('Actual vs OU (Constant Vol) - Deterministic Path')
    axs[0].legend()
    axs[1].plot(simulated_residuals, label='Sim OU Residuals', alpha=0.7)
    axs[1].plot(residuals, label='OU-Price Residuals', alpha=0.7)
    axs[1].set_title('Residuals and GARCH')
    axs[1].legend()
    axs[2].plot(prices, label='OU GARCH', alpha=0.7)
    axs[2].plot(ou_garch_path, label='prices', alpha=0.7) 
    axs[2].set_title('OU GARCH and Prices')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
    garch_fore = garch_fit.forecast(horizon=1)
    var_fore_t1 = garch_fore.variance.values[-1, 0]
    std_fore_t1 = np.sqrt(var_fore_t1)
    print(f'One-step-ahead GARCH forecast for next vol: {std_fore_t1:.4f}')
    next_price_det = ou_path[-1] + theta_hat*(mu_hat - ou_path[-1])*dt
    next_shock = std_fore_t1 * np.random.normal()
    next_price_stoch = next_price_det + next_shock
    print(f'Next price forecast (deterministic OU) = {next_price_det:.4f}, plus GARCH shock => {next_price_stoch:.4f}')

if __name__ == '__main__':
    main('priceData.csv')