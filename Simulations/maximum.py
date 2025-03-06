from stock import get_data
import numpy as np






def gbm_mle_from_prices(prices, delta_t=1.0):
    """
    Estimate the parameters mu and sigma of a Geometric Brownian Motion (GBM):
        dX_t = mu * X_t dt + sigma * X_t dW_t
    using discrete stock price data.

    Args:
        prices (np.ndarray): 1D array of stock prices at equally spaced times.
        delta_t (float): Time step between consecutive observations. Default=1 (e.g., 1 day).

    Returns:
        (mu_hat, sigma_hat): MLE estimates for mu and sigma.
    """
    # 1. Convert price array to log-prices
    log_prices = np.log(prices)
    
    # 2. Compute log-returns (increments in log-price)
    #    r_i = log_prices[i+1] - log_prices[i]
    log_returns = np.diff(log_prices)
    
    # 3. Sample mean and variance of log-returns
    #    If (log_returns) ~ Normal((mu - sigma^2/2)*Delta, sigma^2*Delta),
    #    then for dt=Delta, we have:
    #       mean(log_returns)    = (mu - sigma^2/2) * Delta
    #       var(log_returns)     = sigma^2 * Delta
    #
    #    => sigma^2 = var(log_returns) / Delta
    #       mu      = ( mean(log_returns)/Delta ) + ( sigma^2 / 2 )
    
    r_bar = np.mean(log_returns)              # mean of increments
    r_var = np.var(log_returns, ddof=1)       # unbiased sample variance
    
    # MLE for sigma^2
    sigma_hat_sq = r_var / delta_t
    sigma_hat = np.sqrt(sigma_hat_sq)
    
    # MLE for mu
    mu_hat = (r_bar / delta_t) + (sigma_hat_sq / 2.0)
    
    return mu_hat, sigma_hat

# --------------------------------------------------------------------
# Example usage:
# Suppose you already have your stock prices in a numpy array called `prices`.
# For demonstration, let's generate some synthetic data for `prices`.
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    prices, df = get_data("AAPL")
    # Estimate mu, sigma from that data
    mu_hat, sigma_hat = gbm_mle_from_prices(prices, delta_t=1.0)
    
    print("MLE estimate for mu:", mu_hat)
    print("MLE estimate for sigma:", sigma_hat)
    
    # Quick plot
    plt.plot(prices, label="Simulated GBM Prices")
    plt.title("Synthetic GBM Price Path")
    plt.legend()
    plt.show()

