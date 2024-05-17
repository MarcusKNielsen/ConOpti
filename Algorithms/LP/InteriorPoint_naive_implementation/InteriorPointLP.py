import numpy as np
from scipy.linalg import ldl, solve_triangular
from numpy.linalg import norm
from scipy.sparse.linalg import splu

"""

Primal-Dual Predictor-Corrector Interior Point Linear Programming Algorithm

    min  g' x
    s.t  A' x  = b
         C' x >= d 

x: variable we want to find
y: equality constraint lagrange multiplier
z: inequality constraint lagrange multiplier
s: slack variable

The algorithm below requires an initial guess
(x0,y0,z0,s0) where z0 > 0 and s0 > 0.
x0 should be an interior point, it must not be on the boundary.


"""


def InteriorPointLP2(g,A,b,x0,mu0,lam0,tol = 10e-6,maxiter = 100):

    # x0: Starting point
    # mu0: equality constraint Langragian multiplier
    # lam0: inequality constraint Lagragian multiplier

    # Requirement for algorithm to start
    if not x0>0 and lam0 > 0:
        raise Exception(r"Requirement is not statisfied for $x$ and $\lambda$ > 0")
    
    n,m = A.shape
    
    xk = x0
    muk = mu0
    lamk = lam0

    converged = False
    iter = 0

    while not converged and iter < maxiter:

        # Computing duality meassure
        sk = xk.T@lamk

        # Computing residuals
        rL = g - A.T@muk - lamk     # Lagranian residual    
        rA = A@xk - b               # affine residual
        Xk = xk*np.eye(n)
        LAMk = lamk*np.eye(n)

        KKT_LP = np.block([[np.zeros([n,m]), -A, -np.eye(n)],
                           [A, np.zeros([n,m]),np.zeros([n,m])],
                           [LAMk,np.zeros([n,m]),Xk]
                           ])
        
        sigmak = np.random()
        tauk = sigmak*sk
        rhs = np.block([[-rL],[-rA],[-Xk@LAMk + tauk]])

        # We solve using lu sparse
        LU_decomp = splu(KKT_LP)
        sol = LU_decomp.solve(rhs)

        delta_x = sol[:n]
        delta_mu = sol[n:-n]
        delta_lam = sol[-n:]

        # Find larges affine step alpha
        alphas = np.linspace(0,1,50)
        alphas = alphas[:, np.newaxis]
        candidates_aff = np.block([xk,lamk]) + alphas * np.block([delta_x, delta_lam])
        largest_index_aff = np.argmin(np.all(candidates_aff >= 0, axis=1))
        alpha_aff = alphas[largest_index_aff-1]

        # Updating x and lagragian multipliers
        xk = xk + alpha_aff*delta_x
        muk = muk + alpha_aff*delta_mu
        lamk = lamk + alpha_aff*delta_lam


        converged = (norm(rL,np.inf) < 0 and norm(rA,np.inf) < 0 and norm(sk,np.inf) < 0)
    


def InteriorPointLP(g,A,b,C,d,x0,y0,z0,s0,MaxIter = 100, tol = 10**(-6)):
    
    x,y,z,s = x0,y0,z0,s0
    
    def simple_stopping(r_L,r_A,r_C,mu,tol):
        r = np.block([r_L,r_A,r_C,mu])
        return norm(r,np.inf) < tol
    
    # Calculate residuals
    r_L  = g - A @ y - C @ z
    r_A  = b - A.T @ x 
    r_C  = s + d - C.T @ x
    r_sz = s*z
     
    n,mc = C.shape
    mu = (z.T @ s)/mc
    
    converged = False
    k = 0
    
    X = np.zeros([MaxIter+1,len(x)])
    X[0,:] = x
    
    while converged == False and k < MaxIter:
        
        # Compute H_bar and setup KKT system
        H_bar = C @ np.diag(z/s) @ C.T
        m = A.shape[1]
        KKT = np.block([[H_bar, -A],[-A.T, np.zeros([m,m])]])
        
        # we find ldl factorization of the KKT system
        L, D, perm = ldl(KKT)
        
        # Compute affine direction
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz/z)
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        dx_aff = res[:len(x)]
        dy_aff = res[len(x):]
        
        dz_aff = (-1)*np.diag(z/s) @ C.T @ dx_aff + np.diag(z/s) @ (r_C - r_sz/z)
        ds_aff = - r_sz/z - (s * dz_aff)/z
        
        # Find larges affine step alpha
        alphas = np.linspace(0,1,50)
        alphas = alphas[:, np.newaxis]
        candidates_aff = np.block([z,s]) + alphas * np.block([dz_aff, ds_aff])
        largest_index_aff = np.argmin(np.all(candidates_aff >= 0, axis=1))
        alpha_aff = alphas[largest_index_aff-1]
        
        # Duality gap and centering parameter
        mu_aff = ((z + alpha_aff * dz_aff).T @ (s + alpha_aff * ds_aff)) / mc
        sigma = (mu_aff/mu)**3
        
        # Affine-Centering-Correction Direction
        r_sz_bar = r_sz + ds_aff*dz_aff - sigma*mu * np.ones(len(r_sz))
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz_bar/z)
        
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        dx = res[:len(x)]
        dy = res[len(x):]
        
        
        dz = (-1)*np.diag(z/s) @ C.T @ dx + np.diag(z/s) @ (r_C - r_sz_bar/z)
        ds = -r_sz_bar/z - s * dz/z
        
        candidates = np.block([z,s]) + alphas * np.block([dz, ds])
        largest_index = np.argmin(np.all(candidates >= 0, axis=1))
        alpha = alphas[largest_index-1]
    
        
        # Update iterate
        nu = 0.9
        alpha_bar = nu*alpha 
        
        x += alpha_bar * dx
        y += alpha_bar * dy
        z += alpha_bar * dz
        s += alpha_bar * ds
        
        # Calculate residuals
        r_L  = g - A @ y - C @ z
        r_A  = b - A.T @ x
        r_C  = s + d - C.T @ x
        r_sz = s*z
        
        mu = (z.T @ s)/mc
        
        converged = simple_stopping(r_L,r_A,r_C,mu,tol)
            
        k += 1
        X[k,:] = x
        
    X = X[:(k+1),:]
    results = dict()
    results["xmin"] = x
    results["lagrange_eq"] = y
    results["lagrange_ineq"] = z
    results["converged"] = converged
    results["iter"] = k
    results["x_array"] = X
    
    return results


#%%

import matplotlib.pyplot as plt

"""

Helper function able to plot LP problems for visualization.
See test_LP_InteriorPointLP for example

"""
def plotLP(g,C,d,X=None,xlimits=None,title=None):
    
    
    def objective(x1, x2):

        linear_term = g[0] * x1 + g[1] * x2
        
        return linear_term
    
    if xlimits == None:
        # Bounds for x1 and x2
        x1min, x1max = -10, 10
        x2min, x2max = -10, 10
    else:
        x1min, x1max, x2min, x2max = xlimits
    
    # Create a grid of points.
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
    z = objective(x1, x2)
    
    # Apply each constraint over the grid
    CT = C.T
    constraint_values = np.empty((len(CT), *x1.shape))
    for i in range(len(CT)):
        constraint_values[i,:,:] = CT[i, 0] * x1 + CT[i, 1] * x2 - d[i]
    
    # Check all constraints for each point in the grid
    feasible = np.all(constraint_values >= 0, axis=0)
    
    # Define the infeasible region
    infeasible_region = ~feasible
    
    # Create a mask for the infeasible region
    infeasible_mask = np.zeros_like(z)
    infeasible_mask[infeasible_region] = 1  # Mark infeasible region
    
    # Initialize the plot.
    plt.figure()
    
    # Plot the objective function.
    cs = plt.contourf(x1, x2, z, levels=20, cmap='viridis')
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
    
    if X is not None:
        x1, x2 = X[:,0], X[:,1]
        plt.plot(x1[0],x2[0],"x",color="red",markersize=15,label=r"$x_0$: initial point")
        plt.plot(x1,x2,"-o",color="red")
        plt.title(title)
        plt.legend()
    
    # Show the plot.
    plt.show()




import numpy as np
from scipy.linalg import ldl, solve_triangular
from numpy.linalg import norm
from scipy.sparse.linalg import splu

"""

Primal-Dual Predictor-Corrector Interior Point Linear Programming Algorithm

    min  g' x
    s.t  A' x  = b
         C' x >= d 

x: variable we want to find
y: equality constraint lagrange multiplier
z: inequality constraint lagrange multiplier
s: slack variable

The algorithm below requires an initial guess
(x0,y0,z0,s0) where z0 > 0 and s0 > 0.
x0 should be an interior point, it must not be on the boundary.


"""


def InteriorPointLP2(g,A,b,x0,mu0,lam0,tol = 10e-6,maxiter = 100):

    # x0: Starting point
    # mu0: equality constraint Langragian multiplier
    # lam0: inequality constraint Lagragian multiplier

    # Requirement for algorithm to start
    if not x0>0 and lam0 > 0:
        raise Exception(r"Requirement is not statisfied for $x$ and $\lambda$ > 0")
    
    n,m = A.shape
    
    xk = x0
    muk = mu0
    lamk = lam0

    converged = False
    iter = 0

    while not converged and iter < maxiter:

        # Computing duality meassure
        sk = xk.T@lamk

        # Computing residuals
        rL = g - A.T@muk - lamk     # Lagranian residual    
        rA = A@xk - b               # affine residual
        Xk = xk*np.eye(n)
        LAMk = lamk*np.eye(n)

        KKT_LP = np.block([[np.zeros([n,m]), -A, -np.eye(n)],
                           [A, np.zeros([n,m]),np.zeros([n,m])],
                           [LAMk,np.zeros([n,m]),Xk]
                           ])
        
        sigmak = np.random()
        tauk = sigmak*sk
        rhs = np.block([[-rL],[-rA],[-Xk@LAMk + tauk]])

        # We solve using lu sparse
        LU_decomp = splu(KKT_LP)
        sol = LU_decomp.solve(rhs)

        delta_x = sol[:n]
        delta_mu = sol[n:-n]
        delta_lam = sol[-n:]

        # Find larges affine step alpha
        alphas = np.linspace(0,1,50)
        alphas = alphas[:, np.newaxis]
        candidates_aff = np.block([xk,lamk]) + alphas * np.block([delta_x, delta_lam])
        largest_index_aff = np.argmin(np.all(candidates_aff >= 0, axis=1))
        alpha_aff = alphas[largest_index_aff-1]

        # Updating x and lagragian multipliers
        xk = xk + alpha_aff*delta_x
        muk = muk + alpha_aff*delta_mu
        lamk = lamk + alpha_aff*delta_lam


        converged = (norm(rL,np.inf) < 0 and norm(rA,np.inf) < 0 and norm(sk,np.inf) < 0)
    


def InteriorPointLP(g,A,b,C,d,x0,y0,z0,s0,MaxIter = 100, tol = 10**(-6)):
    
    x,y,z,s = x0,y0,z0,s0
    
    def simple_stopping(r_L,r_A,r_C,mu,tol):
        r = np.block([r_L,r_A,r_C,mu])
        return norm(r,np.inf) < tol
    
    # Calculate residuals
    r_L  = g - A @ y - C @ z
    r_A  = b - A.T @ x 
    r_C  = s + d - C.T @ x
    r_sz = s*z
     
    n,mc = C.shape
    mu = (z.T @ s)/mc
    
    converged = False
    k = 0
    
    X = np.zeros([MaxIter+1,len(x)])
    X[0,:] = x
    
    while converged == False and k < MaxIter:
        
        # Compute H_bar and setup KKT system
        H_bar = C @ np.diag(z/s) @ C.T
        m = A.shape[1]
        KKT = np.block([[H_bar, -A],[-A.T, np.zeros([m,m])]])
        
        # we find ldl factorization of the KKT system
        L, D, perm = ldl(KKT)
        
        # Compute affine direction
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz/z)
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        dx_aff = res[:len(x)]
        dy_aff = res[len(x):]
        
        dz_aff = (-1)*np.diag(z/s) @ C.T @ dx_aff + np.diag(z/s) @ (r_C - r_sz/z)
        ds_aff = - r_sz/z - (s * dz_aff)/z
        
        # Find larges affine step alpha
        alphas = np.linspace(0,1,50)
        alphas = alphas[:, np.newaxis]
        candidates_aff = np.block([z,s]) + alphas * np.block([dz_aff, ds_aff])
        largest_index_aff = np.argmin(np.all(candidates_aff >= 0, axis=1))
        alpha_aff = alphas[largest_index_aff-1]
        
        # Duality gap and centering parameter
        mu_aff = ((z + alpha_aff * dz_aff).T @ (s + alpha_aff * ds_aff)) / mc
        sigma = (mu_aff/mu)**3
        
        # Affine-Centering-Correction Direction
        r_sz_bar = r_sz + ds_aff*dz_aff - sigma*mu * np.ones(len(r_sz))
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz_bar/z)
        
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        dx = res[:len(x)]
        dy = res[len(x):]
        
        
        dz = (-1)*np.diag(z/s) @ C.T @ dx + np.diag(z/s) @ (r_C - r_sz_bar/z)
        ds = -r_sz_bar/z - s * dz/z
        
        candidates = np.block([z,s]) + alphas * np.block([dz, ds])
        largest_index = np.argmin(np.all(candidates >= 0, axis=1))
        alpha = alphas[largest_index-1]
    
        
        # Update iterate
        nu = 0.9
        alpha_bar = nu*alpha 
        
        x += alpha_bar * dx
        y += alpha_bar * dy
        z += alpha_bar * dz
        s += alpha_bar * ds
        
        # Calculate residuals
        r_L  = g - A @ y - C @ z
        r_A  = b - A.T @ x
        r_C  = s + d - C.T @ x
        r_sz = s*z
        
        mu = (z.T @ s)/mc
        
        converged = simple_stopping(r_L,r_A,r_C,mu,tol)
            
        k += 1
        X[k,:] = x
        
    X = X[:(k+1),:]
    results = dict()
    results["xmin"] = x
    results["lagrange_eq"] = y
    results["lagrange_ineq"] = z
    results["converged"] = converged
    results["iter"] = k
    results["x_array"] = X
    
    return results


 

"""
Code for contour plots with equality constraints.
"""

from scipy.optimize import fsolve
def plotLP_2(g,A,b,X=None,xlimits=None,title=None):
 
    def objective(x1,x2):
        return g[0]*x1 + g[1]*x2

    if not xlimits:
        x1min, x1max = -1,10
        x2min, x2max = -1,10
    else:
        x1min, x1max = xlimits[0],xlimits[1]
        x2min, x2max = xlimits[2],xlimits[3]

    # Create a grid of points.
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
    z = objective(x1, x2)
    
    # Check all constraints for each point in the grid
    feasible = ((x1 >= 0) & (x2 >=0))
    
    # Define the infeasible region
    infeasible_region = ~feasible
    
    # Create a mask for the infeasible region
    infeasible_mask = np.zeros_like(z)
    infeasible_mask[infeasible_region] = 1  # Mark infeasible region
    

    # Initialize the plot.
    plt.figure()

    # Plot the objective function.
    cs = plt.contourf(x1, x2, z, levels=10, cmap='viridis')
    plt.contourf(x1, x2, infeasible_mask, levels=[0.99, 1.01], colors='gray', alpha=0.8)
    plt.contour(cs, colors='k')
    plt.colorbar(cs)

    # Plot equality constraint
    x = np.linspace(x1min,x1max,1000)

    for i in range(A.shape[0]):
        plt.plot(x,(b[i]-A[i,0]*x)/A[i,1],"r--",label= f"constraint {i}")
    
    if X is not None:
        x1, x2 = X[:,0], X[:,1]
        plt.plot(x1[0],x2[0],"x",color="red",markersize=15,label=r"$x_0$: initial point")
        plt.plot(x1,x2,"-o",color="red")
        plt.plot(x1[-1],x2[-1],"x",color="blue",markersize=15,label=r"$x_star$: solution")
        plt.title(title)
        plt.legend()

    # Additional settings for the plot.
    plt.grid(c='k', ls='-', alpha=0.3)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim([x1min, x1max])
    plt.ylim([x2min, x2max])
    plt.legend()

    # Show the plot.
    plt.show()



















