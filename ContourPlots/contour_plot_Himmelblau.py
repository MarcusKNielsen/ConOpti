import matplotlib.pyplot as plt
import numpy as np


"""
Code for contour plots with inequality constraints.
"""

def objective(x1, x2):
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

x1min, x1max = -4,4
x2min, x2max = -4,4

# Create a grid of points.
x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
z = objective(x1, x2)

# implement inequality constraints here
feasible = ((x1+2)**2-x2 >= 0) & (-4*x1+10*x2>= 0)

# Define the infeasible region
infeasible_region = ~feasible

# Create a mask for the infeasible region
infeasible_mask = np.zeros_like(z)
infeasible_mask[infeasible_region] = 1  # Mark infeasible region


# Initialize the plot.
plt.figure()

# Plot the objective function.
levels = np.linspace(0, z.max(), num=10)
cs = plt.contourf(x1, x2, z, levels=levels, cmap='viridis')
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
plt.title("Himmelblau's test problem")


# stationary points
ms = 10
# minimum
plt.plot(3.0,2.0,".",color="r",markersize=ms,label="Minimum")
plt.plot(-3.54854, -1.41941,".",color="r",markersize=ms)
plt.plot(-3.65461, 2.73772,".",color="r",markersize=ms)
plt.plot(-0.298348, 2.89562,".",color="r",markersize=ms)
# saddle
plt.plot(-1.445, 0.319,".",color="magenta",markersize=ms,label="Saddle")
plt.plot(3.234, 1.294,".",color="magenta",markersize=ms)
plt.plot(0.072, 2.880,".",color="magenta",markersize=ms)
plt.plot(-3.066, -0.083,".",color="magenta",markersize=ms)

# maximum
plt.plot(-0.486936, -0.194774,".",color="b",markersize=ms,label="Maximum")

plt.legend(loc="lower left")

# Show the plot.
plt.show()








