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

#%% SQP algorithm

from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, plotQP

def check_optimality(x,Jac_f,z,g,Jac_g,tol):
    
    stationarity    = norm(Jac_f.T - Jac_g.T @ z, np.inf) < tol
    primal_fea_ineq = np.min(g) >= 0
    dual_fea        = np.min(z) >= 0
    complementarity = np.abs(np.dot(z, g)) < tol
    
    return  stationarity & primal_fea_ineq & dual_fea & complementarity


#%%

tol = 10**(-6)
MaxIter = 20

# Initial point
#x = np.array([1.9,2.8], dtype=float)
x = np.array([1.25,0.7], dtype=float)
#x = np.array([3.8,3.8], dtype=float)
#x = np.array([-3.8,1.9], dtype=float)
#x = np.array([0.0,3.3], dtype=float)

z = np.array([1.0,1.0] , dtype=float)*4
y = np.array([]        , dtype=float)
s = np.ones(2) * 2

n_var  = len(x) # Number of variables
n_ineq = len(z) # Number of inequality constraints
n_eq   = len(y) # Number of equality constraints

k = 0 # Iteration counter

# Initialize arrays
X = np.zeros([MaxIter+1,n_var])
X[k,:] = x

Z = np.zeros([MaxIter+1,n_ineq])
Z[k,:] = z

#Y = np.zeros([MaxIter+1,n_eq])
#Y[k,:] = yk


# Check for optimality (only inequality right now)
converged = False

def line_search(x0,l,u,dx,f,df,g):
    
    x = x0.copy()
    alpha = 1
    i = 1
    Stop = False
    
    c = f(x) + u.T @ np.abs(np.minimum(0,g(x)))
    b = df(x) @ dx - u.T @ np.abs(np.minimum(0,g(x)))
    
    while not Stop and i < 25:
        x = x0 + alpha * dx
        phi = f(x) + u.T @ np.abs(np.minimum(0,g(x)))
        if phi <= c + 0.1 * b * alpha:
            Stop = True
        else:
            a = (phi - (c+b*alpha))/alpha**2
            alpha_min = -b/(2*a)
            alpha = np.min([0.9*alpha,np.max([alpha_min,0.1*alpha])])
    
    if i == 15:
        Exception("line search did not converge")
    
    return alpha

#%%
while not converged and k < MaxIter:
    
    # Solve sub QP problem
    
    H = d2f(x) - np.sum(z[:, np.newaxis, np.newaxis] * d2g(x), axis=0)
    q = df(x).T
    
    A = np.zeros([len(x),0])
    b = np.array([])
    
    C = dg(x).T
    d = -g(x)
    
    results = InteriorPointQP(H, q, A, b, C, d, x, y, z, s)
    
    if results['converged'] != True:
        Exception("sub QP problem did not converge!")
    
    #plotQP(H, q, C, d,results['x_array'],title=f"iter={k}")
    
    # Line search
    l = np.abs(y)
    u = np.abs(z)
    alpha = line_search(x,l,u,results['xmin'],f,df,g)
    
    # update variables
    x += alpha*results['xmin']
    y  = results['lagrange_eq']
    z  = results['lagrange_ineq']
    s += results["slack"]
    
    # Check convergence
    converged = check_optimality(x,df(x),z,g(x),dg(x),tol)
    
    # update counter
    k += 1
    
    # Update arrays
    X[k,:] = x
    
X = X[:k,:]

print(converged)
plotHimmelblau(X)



    




