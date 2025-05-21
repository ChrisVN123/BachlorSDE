import numpy as np
import matplotlib.pyplot as plt


def simulated(a, b, sigma1):
    #parameters
    sigma2 = 0.1
    X0 = 5

    n = 100

    #initialize arrays
    X = np.zeros(n)
    Y = np.zeros(n)

    #initial values
    X[0] = X0
    Y[0] = X[0] + np.random.normal(0, sigma2)

    #simulate the process
    for t in range(1, n):
        e1 = np.random.normal(0, sigma1)
        X[t] = a * X[t-1] + b + e1

        e2 = np.random.normal(0, sigma2)
        Y[t] = X[t] + e2

    return X,Y

def myKalmanFilter(y, theta, R, X0, P0):
    a, b, Q = theta
    n = len(y)

    # Allocate arrays
    x_pred = np.zeros(n)  # X_hat_{t|t-1}
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)  # X_hat_{t|t}
    P_filt = np.zeros(n)
    innovations = np.zeros(n)
    S = np.zeros(n)       # Innovation variance

    # Initialization
    x_filt[0] = X0
    P_filt[0] = P0

    for t in range(1, n):
        # Predict
        x_pred[t] = a * x_filt[t-1] + b
        P_pred[t] = a**2 * P_filt[t-1] + Q

        # Innovation
        innovations[t] = y[t] - x_pred[t]
        S[t] = P_pred[t] + R

        # Kalman Gain
        K_t = P_pred[t] / S[t]

        # Update
        x_filt[t] = x_pred[t] + K_t * innovations[t]
        P_filt[t] = (1 - K_t) * P_pred[t]

    return x_pred, P_pred, innovations, S, x_filt, P_filt


# Initial values
X0 = 5
P0 = 1
theta = [0.9, 1, 1]  # a, b, Q
R = 1  # observation noise variance
X, Y = simulated(0.9,1,1)
# Apply Kalman filter to observations Y_t
x_pred, P_pred, innovations, S, x_filt, P_filt = myKalmanFilter(Y, theta, R, X0, P0)
x_pred_lower = x_pred-1.96*np.sqrt(P_pred)
x_pred_higher  =  x_pred+1.96*np.sqrt(P_pred)
# Optional: plot estimate vs true
plt.fill_between(
    range(len(x_pred)),
    x_pred_lower,
    x_pred_higher,
    color="gray",
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.plot(X, label="True $X_t$")
plt.plot(Y, label="Observed $Y_t$", linestyle='--')
plt.plot(x_pred, label="Filtered Estimate $\\hat{X}_{t|t}$", linestyle='-.')

plt.legend()
plt.title("Kalman Filtering Results")
plt.show()
