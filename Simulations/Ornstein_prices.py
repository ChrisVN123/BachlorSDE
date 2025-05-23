import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

###########################################################
# 1) Ornstein-Uhlenbeck simulation in level space
#    X_{n+1} = X_n + theta*(mu - X_n)*dt + sigma * sqrt(dt)*N(0,1)
###########################################################
def OrnsteinSimulated(mu, theta, sigma, dt, X0, N=750):
    """
    Simulate an Ornstein–Uhlenbeck process in *level* space.
    Parameters:
        mu, theta, sigma : OU parameters
        dt : time step
        X0 : initial value
        N  : number of points
    Returns:
        X : array of length N containing the simulated path
        t : array of length N containing the time grid
    """
    # Time grid
    t = np.linspace(0, (N-1)*dt, N)

    # Initialize solution array
    X = np.zeros(N)
    X[0] = X0

    # Simulate Brownian increments
    dB = np.sqrt(dt) * np.random.normal(0, 1, N-1)

    # Euler-Maruyama
    for i in range(1, N):
        drift = theta * (mu - X[i-1]) * dt
        diffusion = sigma * dB[i-1]
        X[i] = X[i-1] + drift + diffusion

    return X, t


###########################################################
# 2) Negative log-likelihood for OU in *level* space
#
#    S_{t+1} | S_t ~ Normal(
#       M = S_t*e^(-theta dt) + mu*(1 - e^(-theta dt)),
#       V = sigma^2/(2*theta)*(1 - e^(-2 theta dt)) 
#    )
#
#  => if you want log(S), adapt accordingly!
###########################################################
def negative_log_likelihood(params, S, dt):
    """
    Compute the negative log-likelihood of an Ornstein-Uhlenbeck model
    for the *levels* S[0], S[1], ..., S[N].

    The model:  S_{t+1} = S_t e^(-theta dt) + mu (1 - e^(-theta dt)) + noise
    with noise ~ Normal(0, variance) and
       variance = sigma^2/(2*theta)*(1 - exp(-2*theta*dt)).

    Parameters
    ----------
    params : [mu, sigma, theta]
    S      : 1D array-like with shape (N+1,). The observed data in level space.
    dt     : time step
    """
    mu, sigma, theta = params

    # Sigma must be strictly positive
    if sigma <= 0:
        return np.inf

    # S[:-1] = S_t, S[1:] = S_{t+1}
    S_t = S[:-1]
    S_tp1 = S[1:]

    # e^{-theta dt}
    e_term = np.exp(-theta*dt)
    e_2term = np.exp(-2*theta*dt)

    # Theoretical conditional mean for next step
    M = mu + (S_t - mu)*e_term

    # Theoretical conditional variance for next step
    V = (sigma**2*(1-e_2term))/(2*theta)

    # Log-likelihood of S_{t+1} given S_t
    n = len(S_t)
    # Summation of [ -1/2 log(2*pi*V) - (S_tp1 - M)^2/(2V) ]
    ll = (
        -0.5 * n * np.log(2.0*np.pi*V)
        - 0.5 * np.sum((S_tp1 - M)**2 / V)
    )

    # Return NEGATIVE log-likelihood
    return -ll


###########################################################
# 3) Example usage:
#    - Read data from CSV (semi-colon delimited).
#    - Extract and clean a portion of the price array.
#    - Perform MLE to find mu, sigma, theta.
#    - Simulate OU and compare.
###########################################################

if __name__ == "__main__":
    #----------------------------------------------------------------
    # Replace with your actual CSV filename/path
    #----------------------------------------------------------------
    data_raw = pd.read_csv("priceData.csv", sep=";")
    df_prices = data_raw["Electricity.Co2.Emission"]

    # Example slice: from index=12000 to 12750
    # Add 1e-10 to avoid zero if you do want to keep it strictly positive.
    prices = df_prices.values[12000:]
    #prices = np.abs(prices) + 1e-10
    # Drop any NaNs
    prices = prices[~np.isnan(prices)]

    # Optional: create a corresponding time index
    hours = df_prices.index[12000:]
    hours = hours[:len(prices)]  # ensure same length if slicing changed shape

    #----------------------------------------------------------------
    # MLE
    #----------------------------------------------------------------
    dt = 1.0  # 1 hour steps, for example

    # Initial guess: [mu, sigma, theta]
    initial_guess = [2, 2, 2]

    # Bounds: each param gets a (low, high) pair
    # e.g.  mu unconstrained, sigma > 0, theta unconstrained
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
    print("----- MLE Results -----")
    print("Estimated mu    =", mu_hat)
    print("Estimated sigma =", sigma_hat)
    print("Estimated theta =", theta_hat)
    print("-----------------------")
    print("Estimated mu    =", np.mean(prices))
    print("Estimated sd    =", np.sqrt(np.var(prices))/2)
    print("Estimated mu    =", ((np.sqrt(np.var(prices)))/np.mean(prices))/4)

    #----------------------------------------------------------------
    # 4) Simulate a new OU path using the fitted parameters
    #----------------------------------------------------------------
    # Choose an initial value near your data average or last known price
    X0 = prices[0] if len(prices)>0 else 0.4
    Nsim = len(prices)  # simulate the same length

    sim_path, sim_time = OrnsteinSimulated(
        mu=mu_hat,
        theta=theta_hat,
        sigma=sigma_hat,
        dt=dt,
        X0=X0,
        N=Nsim
    )

    #----------------------------------------------------------------
    # 5) Compare real data vs. simulated
    #----------------------------------------------------------------
    plt.figure()
    plt.plot(hours, prices, label="Observed prices")
    #plt.plot(hours, sim_path, label="Simulated OU (level space)")
    #plt.plot(hours, np.mean(prices)*np.ones(len(hours)))
    #plt.plot(hours, mu_hat*np.ones(len(hours)))
    plt.legend()
    plt.title("Ornstein–Uhlenbeck Fit Comparison")
    plt.show()

    # errors = prices-sim_path
    # plt.plot(hours, errors)
    # plt.show()