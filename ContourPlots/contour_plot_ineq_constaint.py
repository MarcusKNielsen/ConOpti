import matplotlib.pyplot as plt
import numpy as np


"""
Code for contour plots with inequality constraints.
"""

def objective(x1, x2):
    return x1 - 2*x2

x1min, x1max = -1,10
x2min, x2max = -1,10

# Create a grid of points.
x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
z = objective(x1, x2)

# implement inequality constraints here
feasible = (x2 >= 0) & (x1 >= 0) & (x1-x2+2>= 0) & (x1-5*x2+20>=0) & (x2-5*x1+15>=0)

# Define the infeasible region
infeasible_region = ~feasible

# Create a mask for the infeasible region
infeasible_mask = np.zeros_like(z)
infeasible_mask[infeasible_region] = 1  # Mark infeasible region


# Initialize the plot.
plt.figure()

# Plot the objective function.
cs = plt.contourf(x1, x2, z, levels=10, cmap='viridis')
plt.contour(cs, colors='k')
plt.colorbar(cs)

# Apply the mask for the infeasible region
plt.contourf(x1, x2, infeasible_mask, levels=[0.99, 1.01], colors='gray', alpha=0.8)


# Additional settings for the plot.
plt.grid(c='k', ls='-', alpha=0.3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.xlim([x1min, x1max])
plt.ylim([x2min, x2max])

# Show the plot.
plt.show()








