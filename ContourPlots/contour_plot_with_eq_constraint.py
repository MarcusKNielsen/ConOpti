import matplotlib.pyplot as plt
import numpy as np

"""
Code for contour plots with equality constraints.
"""

def objective(x1, x2):
    return x1**2 + x2**2 +3*x2

from scipy.optimize import fsolve

def eq_con(x1):
    def g(x2):
        return x1**2 + x2+1 - 1
    
    # Check if x1 is a scalar (single value)
    if np.isscalar(x1):
        # Handle scalar input
        root = fsolve(g, 0)
        return root[0]
    else:
        # Handle array input
        return np.array([eq_con(element) for element in x1])


x1min, x1max = -5,5
x2min, x2max = -5,5

# Create a grid of points.
x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
z = objective(x1, x2)


# Initialize the plot.
plt.figure()

# Plot the objective function.
cs = plt.contourf(x1, x2, z, levels=10, cmap='viridis')
plt.contour(cs, colors='k')
plt.colorbar(cs)

# Plot equality constraint
x = np.linspace(x1min,x1max,1000)
plt.plot(x,eq_con(x),"r--")

# Additional settings for the plot.
plt.grid(c='k', ls='-', alpha=0.3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim([x1min, x1max])
plt.ylim([x2min, x2max])

# Show the plot.
plt.show()








