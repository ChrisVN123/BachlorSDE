import numpy as np
import matplotlib.pyplot as plt

def rBM(times, sigma=1, B0=0, u=0):
    dt = np.concatenate(([times[0]], np.diff(times)))
    dB = np.random.normal(loc=u * dt, scale=sigma * np.sqrt(dt))
    B = B0 + np.cumsum(dB)
    return B

x_val = np.linspace(0,1,num=100)
y_val = rBM(x_val)
print(x_val)
plt.plot(x_val,y_val)
plt.show()
