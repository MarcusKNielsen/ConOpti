from Himmelblau_casadi_func import SolveHimmelblau
import matplotlib.pyplot as plt
import numpy as np


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
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim([x1min, x1max])
plt.ylim([x2min, x2max])
plt.title("Himmelblau's test problem (Casadi, Ipopt)")

x0 = [-1.0, 0.0]
X = SolveHimmelblau(x0)
plt.plot(X[:,0],X[:,1],"o-",color="red",label=r"$x_0 = (-1.0,0.0)^\top$")

x0 = [1.0, 3.8]
X = SolveHimmelblau(x0)
plt.plot(X[:,0],X[:,1],"o-",color="magenta",label=r"$x_0 = (1.0,3.8)^\top$")

x0 = [-3.5, 0.5]
X = SolveHimmelblau(x0)
plt.plot(X[:,0],X[:,1],"o-",color="gold",label=r"$x_0 = (-3.5,0.5)^\top$")

x0 = [0.8, 3.8]
X = SolveHimmelblau(x0)
plt.plot(X[:,0],X[:,1],"o-",color="springgreen",label=r"$x_0 = (0.8,3.8)^\top$")

plt.legend(loc="lower right")

# Show the plot.
plt.show()








