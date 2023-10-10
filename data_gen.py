# generate data for the Lorenz system
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Lorenz paramters and initial conditions
sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
u0, v0, w0 = np.random.rand(3)

# Maximum time point and total number of time points
tmax, n = 3000, 150000
print(tmax/n)

def lorenz_eqn(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Integrate the Lorenz equations on the time grid t
t = np.linspace(0, tmax, n)
f = odeint(lorenz_eqn, (u0, v0, w0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig = plt.figure()
ax = fig.subplots()

# Make the line multi-coloured by plotting it in segments of length s which
# change in colour across the whole time series.
s = 10
c = np.linspace(0,1,n)

ax.plot(x[124700:124900])

# Remove all the axis clutter, leaving just the curve.

plt.show()

np.savetxt('Lorenz_Train.txt', x[1000:101000])
np.savetxt('Lorenz_Test.txt', x[124700:])

# generate data for the Rossler system
