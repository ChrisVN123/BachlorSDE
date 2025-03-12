import numpy as np
from scipy.optimize import minimize
from stock import get_data, compute_drift_and_diffusion

def simulate_double_well(S0=100, r=1, q=1, sigma=0.2, T=1.0, N=252, seed=42):
    """
    Simulates a price path using a double-well potential governed by:
    dXt = (rXt − qXt^3) dt + σ dBt.
    """
    np.random.seed(seed)
    dt = T / N
    S = np.zeros(N+1)
    S[0] = S0
    
    for i in range(1, N+1):
        Z = np.random.normal(0, 1)
        drift = (r * S[i-1] - q * S[i-1]**3) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        S[i] = S[i-1] + drift + diffusion
    
    return S

def negative_log_likelihood(params, S, dt):
    """
    Negative log-likelihood function based on the double-well model:
    dXt = (rXt − qXt^3) dt + σ dBt.
    """
    r, q, sigma = params
    if sigma <= 0:
        return np.inf
    
    X = np.diff(S)  # First-order differences in price
    
    # Drift term from double-well model
    drift = (r * S[:-1] - q * S[:-1]**3) * dt
    var_increment = sigma**2 * dt
    
    n = len(X)
    ll = -0.5 * n * np.log(2 * np.pi * var_increment) \
         - 0.5 * np.sum((X - drift)**2 / var_increment)
    
    return -ll

#-----------------------------------------------------
# 1) Get stock data
S_data, df = get_data("AAPL", 1)
print(S_data)

# 2) Initial parameter guess
initial_guess = [1.0, 1.0, 0.3]  # (r, q, sigma)
dt = 1.0 / 252

bench = compute_drift_and_diffusion(S_data, dt)
print(bench)

# 3) Optimize
bounds = [(None, None), (None, None), (1e-12, None)]  # Keep sigma > 0
result = minimize(
    fun=negative_log_likelihood,
    x0=initial_guess,
    args=(S_data, dt),
    method='L-BFGS-B',
    bounds=bounds
)

r_hat, q_hat, sigma_hat = result.x
print("Estimated r =", r_hat)
print("Estimated q =", q_hat)
print("Estimated sigma =", sigma_hat)