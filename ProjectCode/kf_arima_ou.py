import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm

# CRPS and variogram 
def crps_gaussian(mu, sigma2, y):
    """
    CRPS of a Normal(mu, sigma2) forecast against scalar y.
    """
    np.random.seed(42)
    sigma = np.sqrt(sigma2)
    z = (y - mu) / sigma
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    return sigma * (z * (2*cdf - 1) + 2*pdf - 1/np.sqrt(np.pi))


def empirical_variogram(x, max_lag):

    n = len(x)
    lags = np.arange(1, max_lag + 1)
    gamma = np.zeros_like(lags, dtype=float)
    for i, h in enumerate(lags):
        diffs = x[h:] - x[:-h]
        gamma[i] = 0.5 * np.mean(diffs**2)
    return lags, gamma

#OU discretization and Kalman filter functions
def discretize_ou(theta, dt):
    phi = np.exp(-theta[0]*dt)
    mu  = theta[1]
    Q   = theta[2]**2/(2*theta[0])*(1-np.exp(-2*theta[0]*dt))
    R   = theta[3]**2
    return phi, mu, Q, R

#modified kalman_filter to also return predictive means/variances
def kalman_filter(y, theta, dt):
    n = len(y)
    phi, mu, Q, R = discretize_ou(theta, dt)
    x_k = np.zeros(n); P_k = np.zeros(n)
    mu_pred = np.zeros(n); S_pred = np.zeros(n)
    residuals = np.zeros(n); L_k = np.zeros(n)

    x_k[0] = y[0]; P_k[0] = 1.0
    S_pred[0] = P_k[0] + R; mu_pred[0] = x_k[0]
    L_k[0] = -0.5*(np.log(2*np.pi*S_pred[0]))

    for i in range(1, n):
        x_p = mu +  phi * x_k[i-1] + (1-phi)*mu
        P_p = phi**2 * P_k[i-1] + Q
        mu_pred[i] = x_p; S_pred[i] = P_p + R
        K = P_p / S_pred[i]
        x_k[i] = x_p + K*(y[i] - x_p)
        P_k[i] = (1-K)*P_p
        residuals[i] = y[i] - x_p
        L_k[i] = -0.5*(np.log(2*np.pi*S_pred[i]) + residuals[i]**2/S_pred[i])

    return L_k, x_k, residuals, mu_pred, S_pred

#forecast with ARIMA as pseudo-observations
def kf_ou_forecast_with_arima(train, arima_f, dt, theta):
    phi, mu, Q, R = discretize_ou(theta, dt)
    n_train, n_f = len(train), len(arima_f)
    T = n_train + n_f
    x = np.zeros(T); P = np.zeros(T)
    mu_pred = np.zeros(T); S_pred = np.zeros(T)

    x[0] = train[0]; P[0] = 1.0
    mu_pred[0] = x[0]; S_pred[0] = P[0] + R

    for t in range(1, n_train):
        x_p = phi*x[t-1] + (1-phi)*mu; P_p = phi**2*P[t-1] + Q
        mu_pred[t] = x_p; S_pred[t] = P_p + R
        K = P_p/S_pred[t]
        x[t] = x_p + K*(train[t] - x_p); P[t] = (1-K)*P_p

    for i in range(n_f):
        idx = n_train + i
        x_p = phi*x[idx-1] + (1-phi)*mu; P_p = phi**2*P[idx-1] + Q
        mu_pred[idx] = x_p; S_pred[idx] = P_p + R
        K = P_p/S_pred[idx]
        x[idx] = x_p + K*(arima_f[i] - x_p); P[idx] = (1-K)*P_p

    return x, P, mu_pred, S_pred

# NLL for optimizer
def negative_log_likelihood(theta, y, dt):
    L_k, *_ = kalman_filter(y, theta, dt)
    return -np.sum(L_k)

# Fit parameters
def optimal_kalman(dt, guess, train, plotting=False):
    res = minimize(negative_log_likelihood, guess, args=(train, dt), bounds=[(1e-5,None)]*4)
    theta = res.x
    print(f"Estimated theta: {theta}")
    L_k, x_filt, resid, mu_pred, S_pred = kalman_filter(train, theta, dt)
    if plotting:
        plt.plot(train, label='Obs'); plt.plot(x_filt, '--',label='Filt'); plt.legend(); plt.show()
    return x_filt, theta, resid, mu_pred, S_pred

# OU-only forecast
def forecast_ou(theta, x0, P0, steps, dt):
    phi, mu, Q, R = discretize_ou(theta, dt)
    xf = np.zeros(steps); Pf = np.zeros(steps)
    xf[0] = phi*x0 + (1-phi)*mu; Pf[0] = phi**2*P0 + Q
    for i in range(1, steps): xf[i] = phi*xf[i-1] + (1-phi)*mu; Pf[i] = phi**2*Pf[i-1] + Q
    return xf, Pf

# Data loader
def get_data(s,e):
    d = pd.read_csv("priceData.csv", sep=";")
    p = np.abs(d["Spot.price"].values)
    arr = pd.Series(p[s:e]).interpolate().to_numpy()
    return arr[:-24], arr[-24:]



def main():
    dt = 1/24
    train, test = get_data(23,13894)
    x_filt, theta, resid, mu_p, S_p = optimal_kalman(dt, np.array([0.7,0.1,1,0.02]), train)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Residual histogram
    axes[0].hist(resid, bins=30, edgecolor='black')
    axes[0].axhline(0, color='black', linestyle='--')
    axes[0].set_title("In‐Sample Residuals")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")

    # 2) QQ‐plot
    stats.probplot(resid, dist="norm", plot=axes[1])
    axes[1].set_title("QQ‐Plot of Residuals")
    axes[1].get_lines()[1].set_color('red')

    # 3) ACF plot
    sm.graphics.tsa.plot_acf(resid, lags=24, ax=axes[2])
    axes[2].set_title("ACF of Residuals")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.show()

    # ===== in‐sample CRPS & variogram =====
    crps_in = [crps_gaussian(mu_p[i], S_p[i], train[i]) for i in range(1,len(train))]
    print(f"Mean in-sample CRPS: {np.mean(crps_in):.4f}")

    l_r, g_r = empirical_variogram(resid,24)
    plt.figure()
    plt.plot(l_r, g_r, 'o-')
    plt.title('Variogram Kalman Filter')
    plt.xlabel('Lag')
    plt.ylabel('Semivariance')
    plt.show()
    print(g_r)

    # ===== ARIMA + KF forecast & out‐of‐sample diagnostics =====
    ar = sm.tsa.SARIMAX(train[-96:], order=(0,1,1),
                        seasonal_order=(0,1,0,24)).fit(disp=False)
    ar_f = ar.get_forecast(steps=24).predicted_mean
    x_comb, P_comb, mu_f, S_f = kf_ou_forecast_with_arima(train, ar_f, dt, theta)

    crps_oos = [crps_gaussian(mu_f[len(train)+i], S_f[len(train)+i], test[i]) for i in range(len(test))]
    print(f"Mean out-of-sample CRPS: {np.mean(crps_oos):.4f}")

    errs = test - x_comb[len(train):]
    l_e, g_e = empirical_variogram(errs,24)
    plt.figure()
    plt.plot(l_e, g_e, 'o-')
    print(g_e)
    plt.title('Variogram of the combined model residuals')
    plt.xlabel('Lag')
    plt.ylabel('Semivariance')
    plt.show()
    print(g_e)

    # ===== final comparison plot =====
    idx = np.arange(len(train), len(train)+len(test))
    plt.figure(figsize=(10,5))
    plt.plot(idx, test, 'o-', label='True Test')
    plt.plot(idx, x_comb[len(train):], 's--', label='KF+ARIMA Forecast')
    plt.title('True vs Combined KF+ARIMA Forecast on Test Set')
    plt.xlabel('Time Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
