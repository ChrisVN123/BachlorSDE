import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import i0, i1, iv, gammaln  # iν is the modified Bessel function; we may use iv for real order


def simulateCIR(kappa, theta, sigma, dt, X0, N=750):
    """
    Simulate a Cox–Ingersoll–Ross (CIR) process via Euler–Maruyama.
    
    dX_t = kappa*(theta - X_t)*dt + sigma*sqrt(X_t)*dW_t
    
    Parameters
    ----------
    kappa, theta, sigma : floats
        CIR parameters (must be positive).
    dt : float
        Time step.
    X0 : float
        Initial value, X_0 >= 0.
    N : int
        Number of points in the time grid (total length of the simulation).
    
    Returns
    -------
    X : ndarray of shape (N,)
        The simulated path.
    t : ndarray of shape (N,)
        The time grid t[0..N-1].
    """
    t = np.linspace(0, (N-1)*dt, N)
    X = np.zeros(N)
    X[0] = X0
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=N-1)
    
    for i in range(1, N):
        # Euler-Maruyama step
        x_prev = X[i-1]
        drift = kappa*(theta - x_prev)*dt
        diffusion = sigma*np.sqrt(max(x_prev, 0.0))*dW[i-1]
        X[i] = x_prev + drift + diffusion
        # Enforce nonnegativity if you wish:
        # X[i] = max(X[i], 0.0)

    return X, t

def cir_transition_pdf(y, x, kappa, theta, sigma, dt):
    """
    Returns the transition PDF f_{X_{t+dt}|X_t}(y|x) for the CIR process,
    using the known scaled noncentral chi-square formula.
    
    If y<=0, x<0, or parameters are invalid => returns 0.
    """
    if y <= 0 or x < 0 or sigma <= 0 or kappa <= 0 or theta <= 0:
        return 0.0
    
    # Degrees of freedom
    nu = 4.0 * kappa * theta / (sigma**2)
    
    # Scale
    c = (sigma**2 * (1.0 - np.exp(-kappa*dt))) / (4.0*kappa)
    if c <= 0:
        return 0.0
    
    # Noncentrality
    lam = 4.0 * kappa * x * np.exp(-kappa*dt) / (sigma**2 * (1.0 - np.exp(-kappa*dt)))
    
    # Coefficients
    coef = 1.0 / (2.0 * c)
    exponent = np.exp(-0.5*(lam + y/c))
    # exponent factor => exp(- (lam + y/c)/2)
    
    # The power factor => (y / (lam * c))^( (nu/2)/2 - 1/2 ) = (y / (lam * c))^(nu/4 - 1/2)
    alpha = (nu/4.0) - 0.5
    power = (y / (lam * c))**alpha if (y>0 and lam>0 and c>0) else 0.0
    
    # Bessel function argument z = 2 sqrt( lam*y / c )
    z = 2.0 * np.sqrt(lam * y / c)
    # The order for iv is (nu/2 - 1)
    order = (nu/2.0) - 1.0
    bess = iv(order, z)
    
    pdf_val = coef * exponent * power * bess
    return pdf_val

def negative_log_likelihood_CIR(params, X, dt=1.0):
    """
    Negative log-likelihood for the CIR model, given data X[0..N],
    assuming constant time step dt between observations.

    params : (kappa, theta, sigma)
    X      : 1D array-like (length N+1)
    dt     : float (time step, default=1.0)

    Returns
    -------
    float
        The *negative* log-likelihood.
    """
    kappa, theta, sigma = params
    
    # Ensure positivity of parameters
    if sigma <= 0 or kappa <= 0 or theta <= 0:
        return np.inf
    
    X = np.asarray(X)
    n = len(X) - 1
    if n < 1:
        return np.inf  # no transitions
    
    loglik = 0.0
    for i in range(n):
        x_current = X[i]
        x_next = X[i+1]
        
        p = cir_transition_pdf(x_next, x_current, kappa, theta, sigma, dt)
        if p <= 0:
            return np.inf  # log(0) => -inf, so negative LL => +inf
        loglik += np.log(p)
    
    # Return NEGATIVE log-likelihood
    return -loglik
if __name__ == "__main__":
   
    #----------------------------------------------------------------
    # Replace with your actual CSV filename/path
    #----------------------------------------------------------------
    data_raw = pd.read_csv("priceData.csv", sep=";")
    df_prices = data_raw["Spot.price"]

    # Example slice: from index=12000 to 12750
    # Add 1e-10 to avoid zero if you do want to keep it strictly positive.
    prices = df_prices.values[12000:12700]
    #prices = np.abs(prices) + 1e-10
    # Drop any NaNs
    prices = prices[~np.isnan(prices)]
    prices = abs(prices)

    # Optional: create a corresponding time index
    hours = df_prices.index[12000:12700]
    hours = hours[:len(prices)]  # ensure same length if slicing changed shape

    #------------------------------------------------
    # Suppose 'df' is your data: a Pandas Series of X-values
    # Example: X_data = df["your_column_name"].dropna().values
    # If 'df' is itself just a Series, then:
    #   X_data = df.dropna().values
    #------------------------------------------------
    X_data = prices # or df.values if already cleaned

    # Check positivity (CIR requires X >= 0)
    # If any negative values are present, you'd have to handle that
    # For demonstration, let's clamp them:
    #X_data = np.clip(X_data, a_min=1e-12, a_max=None)

    dt = 1.0  # e.g. 1 hour or 1 day steps, if that's your data frequency

    # initial guess [kappa, theta, sigma]
    initial_guess = [0.7444, 0.1412, 0.0270]

    # bounds => all strictly > 0
    bnds = [(1e-12, None), (1e-12, None), (1e-12, None)]

    result = minimize(
        fun=negative_log_likelihood_CIR,
        x0=initial_guess,
        args=(X_data, dt),
        method='L-BFGS-B',
        bounds=bnds
    )

    if not result.success:
        print("Optimization failed:", result.message)

    kappa_hat, theta_hat, sigma_hat = result.x

    print("===== CIR MLE Results =====")
    print(f"kappa = {kappa_hat:.5f}")
    print(f"theta = {theta_hat:.5f}")
    print(f"sigma = {sigma_hat:.5f}")
    print("===========================")

    # Optionally, print the final negative log-likelihood
    print(f"Negative log-likelihood = {result.fun:.5f}")


    # ============================
    # Simulate the CIR process using the estimated parameters
    # ============================
    X0 = X_data[0]  # Start the simulation from the first data point
    N_sim = len(X_data)  # Simulate same length as the data
    simulated_path, t_sim = simulateCIR(kappa_hat, theta_hat, sigma_hat, dt, X0, N=N_sim)

    # ============================
    # Plot the actual data vs simulated path
    # ============================
    plt.figure(figsize=(12, 6))
    plt.plot(hours, X_data, label='Actual Data', color='blue', lw=2)
    plt.plot(hours, simulated_path, label='Simulated CIR Path (MLE)', color='red', lw=2, linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Spot Price')
    plt.title('CIR Model - Actual Data vs Simulated Path')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

