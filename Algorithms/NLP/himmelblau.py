import numpy as np
import matplotlib.pyplot as plt

"""
Himmelblau Test Problem
"""

def f(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def df(x): # Jacobian of f
    x1, x2 = x
    J1 = 4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7)
    J2 = 2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
    return np.array([J1, J2])

def d2f(x): # Hessian of f
    x1, x2 = x
    
    H11 = 12*x1**2 + 4*x2 - 42
    H12 = 4*(x1+x2)
    H21 = 4*(x1+x2)
    H22 = 12*x2**2 + 4*x1 - 26
    
    return np.array([[H11,H12],[H21,H22]])

def g(x):
    x1, x2 = x
    
    g1 = (x1+2)**2-x2
    g2 = -4*x1+10*x2
    
    return np.array([g1,g2])

def dg(x): # Jacobian of constraints
    x1, x2 = x
    
    J11 = 2*(x1 + 2)
    J12 = -1
    J21 = -4
    J22 = 10
    
    return np.array([[J11,J12],[J21,J22]])

def d2g(x):
    x1, x2 = x
    
    H = np.zeros([2,2,2])
    H[0,0,0] = 2
    
    return H

def plotHimmelblau(X=None):
    def objective(x1,x2):
        x = np.array([x1,x2])
        return f(x)
    
    a = 4
    x1min, x1max = -a,a
    x2min, x2max = -a,a
    
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
    plt.title("Himmelblau's test problem")
    
    if X is not None:
        x1, x2 = X[:,0], X[:,1]
        plt.plot(x1[0],x2[0],"x",color="red",markersize=15,label=r"$x_0$: initial point")
        plt.plot(x1,x2,"-o",color="red")
        plt.legend(loc = "lower right")

    plt.show()

