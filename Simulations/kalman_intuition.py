import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Initialize parameters
dt = 1.0  # time step
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
])  # State transition

H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])  # Measurement function

Q = 0.1 * np.eye(4)  # Process noise
R = np.array([[1.0, 0], [0, 1.0]])  # Measurement noise

x = np.array([[0], [0], [1], [0.5]])  # Initial state: [x, y, vx, vy]
P = np.eye(4)  # Initial uncertainty

# Simulate
N = 20
true_states = []
measurements = []
estimates = []

def draw_ellipse(ax, mean, cov, color='red'):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=color,
                      fc='None', lw=2, alpha=0.5)
    ax.add_patch(ellipse)

for t in range(N):
    # Simulate true state
    x = A @ x + np.random.multivariate_normal([0, 0, 0, 0], Q).reshape(-1, 1)
    z = H @ x + np.random.multivariate_normal([0, 0], R).reshape(-1, 1)

    # Predict
    x_pred = A @ x_est if t > 0 else np.zeros((4, 1))
    P_pred = A @ P @ A.T + Q

    # Update
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred

    # Store for plotting
    true_states.append(x[:2].flatten())
    measurements.append(z.flatten())
    estimates.append(x_est[:2].flatten())

# Plotting
true_states = np.array(true_states)
measurements = np.array(measurements)
estimates = np.array(estimates)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(true_states[:, 0], true_states[:, 1], label='True Path', c='black', linestyle='--')
ax.scatter(measurements[:, 0], measurements[:, 1], label='Measurements', c='blue')
ax.plot(estimates[:, 0], estimates[:, 1], label='Filtered Estimate', c='red')

# Draw uncertainty ellipses
x_est = np.zeros((4, 1))
P = np.eye(4)
for t in range(N):
    x_pred = A @ x_est
    P_pred = A @ P @ A.T + Q
    z = measurements[t].reshape(-1, 1)
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred
    draw_ellipse(ax, x_est[:2].flatten(), P[:2, :2], color='green')

ax.legend()
ax.set_title("Kalman Filter in 2D: Position Tracking with Uncertainty Ellipses")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.axis('equal')
plt.grid()
plt.show()
