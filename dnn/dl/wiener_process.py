import numpy as np
import matplotlib.pyplot as plt

t = 1.0 # total time duration
n = 1000 # number of discrete timesteps
dt = t/n
t1 = np.linspace(0, t, n)

np.random.seed(42)
dW = np.sqrt(dt) * np.random.normal(size=n)
W = np.cumsum(dW)

plt.plot(t1, W, label='Wiener process')
plt.title("Simulation")
plt.xlabel('Time')
plt.ylabel('W(t)')
plt.legend()
plt.show()