import numpy as np
import matplotlib.pyplot as plt

mu = 0

start = 0
stop = 100
num_steps = 1000

steps = np.linspace(start, stop, num=num_steps)
motion = np.zeros(num_steps)

for i in range(1,steps.shape[0]):
    sd = np.sqrt(steps[i]-steps[i-1])
    motion[i] = motion[i-1] + np.random.normal(loc=mu, scale=sd, size=1)

plt.plot(steps, motion)
plt.show()