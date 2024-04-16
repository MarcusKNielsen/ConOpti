import numpy as np
import matplotlib.pyplot as plt

# Himmelblau test problem

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

def plotHimmelblau(X):
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
    
    plt.plot(X[:,0],X[:,1],"o-",color="red")
    # Show the plot.
    plt.show()

#%% SQP algorithm

from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, InitialPointQP, plotQP


def check_optimality(xk,Jac_fk,zk,gk,Jac_gk):
    stationarity = norm(Jac_fk - Jac_gk @ zk, np.inf) < tol
    primal_fea_ineq = np.min(gk) >= 0
    dual_fea = np.min(z) >= 0
    complementarity = np.abs(np.dot(z, gk)) < tol
    return  stationarity & primal_fea_ineq & dual_fea & complementarity


tol = 10**(-6)
MaxIter = 10

# Initial point
x = np.array([1.2,3.0],dtype=float)
z = np.array([1,1],dtype=float)

k = 0 # Iteration counter
xk = np.copy(x)
zk = np.copy(z)

fk     = f(xk)
Jac_fk = df(xk)
Hes_fk = d2f(xk)

gk     = g(xk)
Jac_gk = dg(xk)
Hes_gk = d2g(xk)

converged = check_optimality(xk,Jac_fk,zk,gk,Jac_gk)
X = np.zeros([MaxIter+1,len(x)])
X[k,:] = xk
while not converged and k < MaxIter:
    
    # Solve QP sub-problem
    H = Hes_fk - z[0] * Hes_gk[:,:,0] - z[1] * Hes_gk[:,:,1]
    grad = Jac_fk
    A = np.zeros([2,0])
    b = np.zeros([])
    C = Jac_gk.T
    d = -gk
    yk = np.array([])
    x0 = np.ones(len(xk))
    y0 = np.ones(len(yk))
    z0 = np.ones(len(zk))
    s0 = np.ones(len(gk))
    x0,y0,z0,s0 = InitialPointQP(-H,grad,A,b,C,d,x0,y0,z0,s0)
    results = InteriorPointQP(H,grad,A,b,C,d,x0,y0,z0,s0,MaxIter = 100, tol = 10**(-6))
    X_subQP = results['x_array']
    dx = results['xmin']
    plotQP(H,grad,C,d,X_subQP)
    
    # Line search
    lambd = np.abs(yk)
    mu = np.abs(zk)
    
    alpha = 1
    c = fk 
    
    xk += dx
    zk  = results['lagrange_ineq']
    yk  = results['lagrange_eq']
    k += 1
    
    fk     = f(xk)
    Jac_fk = df(xk)
    Hes_fk = d2f(xk)

    gk     = g(xk)
    Jac_gk = dg(xk)
    Hes_gk = d2g(xk)
    
    converged = check_optimality(xk,Jac_fk,zk,gk,Jac_gk)
    X[k,:] = xk
    #plotHimmelblau(X[:k,:])



plotHimmelblau(X[:k,:])












