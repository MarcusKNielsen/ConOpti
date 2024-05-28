import numpy as np
from scipy.linalg import ldl, solve_triangular, solve
from numpy.linalg import norm

"""

Primal-Dual Predictor-Corrector Interior Point Quadratic Programming Algorithm

min 0.5 x' H x + g' x
subject to A' x  = b
           C' x >= d 

x: variable we want to find
y: equality constraint lagrange multiplier
z: inequality constraint lagrange multiplier
s: slack variable

The algorithm below requires an initial guess
(x0,y0,z0,s0) where z0 > 0 and s0 > 0.
x0 should be an interior point, it must not be on the boundary.

"""

 
def InteriorPointQP(H,g,A,b,C,d,x0,y0,z0,s0,MaxIter = 100, tol = 10**(-6)):
    
    x = np.copy(x0)
    y = np.copy(y0)
    z = np.copy(z0)
    s = np.copy(s0)
    
    def simple_stopping(r_L,r_A,r_C,mu,tol):
        r = np.block([r_L,r_A,r_C,mu])
        return norm(r,np.inf) < tol
    
    # Calculate residuals
    r_L  = H @ x + g - A @ y - C @ z
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
        H_bar = H + C @ np.diag(z/s) @ C.T
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
        r_L  = H @ x + g - A @ y - C @ z
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
    results["slack"] = s
    results["lagrange_eq"] = y
    results["lagrange_ineq"] = z
    results["converged"] = converged
    results["iter"] = k
    results["x_array"] = X
    
    return results

"""
Heuristic Algorithm for finding Initial Point for Interior Point QP algorithm
"""


def InitialPointQP(H,g,A,b,C,d,x,y,z,s):
     
    r_L  = H @ x + g - A @ y - C @ z
    r_A  = b - A.T @ x
    r_C  = s + d - C.T @ x
    r_sz = s*z
    
    H_bar = H + C @ np.diag(z/s) @ C.T
    m = A.shape[1]
    KKT = np.block([[H_bar, -A],[-A.T, np.zeros([m,m])]])
    
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
    
    z += dz_aff
    s += ds_aff
    
    z = np.maximum(1, np.abs(z))
    s = np.maximum(1, np.abs(s))
    
    return x,y,z,s


#%%

import matplotlib.pyplot as plt

"""

Helper function able to plot QP problems for visualization.
See InteriorPointQP_test for example

"""
def plotQP(H,g,C,d,X=None,xlimits=None,title=None):
    
    
    def objective(x1, x2):

        quadratic_term = 0.5 * (H[0, 0] * x1**2 + 2 * H[0, 1] * x1 * x2 + H[1, 1] * x2**2)
        linear_term = g[0] * x1 + g[1] * x2
        
        return quadratic_term + linear_term
    
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

def plotQP_eq(H,g,C,d,A,b,X=None,xlimits=None,title=None,idx_figure = None):
    
    
    def objective(x1, x2):

        quadratic_term = 0.5 * (H[0, 0] * x1**2 + 2 * H[0, 1] * x1 * x2 + H[1, 1] * x2**2)
        linear_term = g[0] * x1 + g[1] * x2
        
        return quadratic_term + linear_term
    
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
    plt.figure(idx_figure)
    
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
    
    x1 = np.linspace(x1min,x1max,100)
    x1 = np.linspace(x2min,x2max,100)
    
    for i in range(A.T.shape[0]):
        a1,a2 = A.T[i]
        plt.plot(x1,(b[i] - a1*x1)/a2,"--",color="red")
    
    if X is not None:
        x1, x2 = X[:,0], X[:,1]
        plt.plot(x1[0],x2[0],"x",color="red",markersize=15,label=r"$x_0$: initial point")
        plt.plot(x1,x2,"-o",color="red")
        plt.title(title)
        plt.legend()
    
    # Show the plot.
    plt.show()

#%%

"""
Version 2
"""


def InteriorPointQP_v2(H,g,A,b,C,d,x0,y0,z0,s0,MaxIter = 100, tol = 10**(-6)):
    
    x = np.copy(x0)
    y = np.copy(y0)
    z = np.copy(z0)
    s = np.copy(s0)
    
    def simple_stopping(r_L,r_A,r_C,mu,tol):
        r = np.block([r_L,r_A,r_C,mu])
        return norm(r,np.inf) < tol
    
    # Calculate residuals
    r_L  = H @ x + g - A @ y - C @ z
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
        H_bar = H + C @ np.diag(z/s) @ C.T
        m = A.shape[1]
        KKT = np.block([[H_bar, -A],[-A.T, np.zeros([m,m])]])
        
        # we find ldl factorization of the KKT system
        L, D, perm = ldl(KKT)
        
        # Compute affine direction
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz/z)
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        #res = solve_triangular(L.T,solve_triangular(D,solve_triangular(L,rhs[perm])))
        dx_aff = res[:len(x)]
        dy_aff = res[len(x):]
        
        dz_aff = (-1)*np.diag(z/s) @ C.T @ dx_aff + np.diag(z/s) @ (r_C - r_sz/z)
        ds_aff = - r_sz/z - (s * dz_aff)/z
        
        # Step length
        """ idx = np.where((dz_aff < 0) & (ds_aff < 0)) 
        if (z[idx]/dz_aff[idx]).size == 0 and (s[idx]/ds_aff[idx]).size == 0:
            alpha_aff = 1
        else:
            tau = 0.1
            alpha_aff = min(1,max(np.concatenate([-tau*z[idx]/dz_aff[idx],-tau*s[idx]/ds_aff[idx]]))) """
        
        alpha_aff = min(np.concatenate([np.array([1]),(-z/dz_aff)[(-z/dz_aff)>0],(-s/ds_aff)[(-s/ds_aff)>0]]))

        # Duality gap and centering parameter
        mu_aff = ((z + alpha_aff * dz_aff).T @ (s + alpha_aff * ds_aff)) / mc
        sigma = (mu_aff/mu)**3
        
        # Affine-Centering-Correction Direction
        r_sz_bar = r_sz + ds_aff*dz_aff - sigma*mu * np.ones(len(r_sz))
        r_L_bar = r_L - C @ np.diag(z/s) @ (r_C - r_sz_bar/z)
        
        rhs = (-1)*np.block([r_L_bar, r_A])
        rhs2 = solve_triangular(L[perm,:], rhs[perm],lower=True)
        res = solve_triangular(D @ L[perm,:].T, rhs2)[perm]
        #res = solve_triangular(L.T,solve_triangular(D,solve_triangular(L,rhs[perm])))
        dx = res[:len(x)]
        dy = res[len(x):]
        
        dz = (-1)*np.diag(z/s) @ C.T @ dx + np.diag(z/s) @ (r_C - r_sz_bar/z)
        ds = -r_sz_bar/z - s * dz/z
        
        # Step length
        """ idx = np.where((dz < 0) & (ds < 0)) 
        if (z[idx]/dz[idx]).size == 0 and (s[idx]/ds[idx]).size == 0:
            alpha = 1
        else:
            tau = 0.1
            alpha = min(1,max(np.concatenate([-tau*z[idx]/dz[idx],-tau*s[idx]/ds[idx]])))  """
        
        alpha = min(np.concatenate([np.array([1]),(-z/dz)[(-z/dz)>0],(-s/ds)[(-s/ds)>0]]))

        # Update iterate
        nu = 0.995
        alpha_bar = nu*alpha 
        
        x += alpha_bar * dx
        y += alpha_bar * dy
        z += alpha_bar * dz
        s += alpha_bar * ds
        
        # Calculate residuals
        r_L  = H @ x + g - A @ y - C @ z
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
    results["slack"] = s
    results["lagrange_eq"] = y
    results["lagrange_ineq"] = z
    results["converged"] = converged
    results["iter"] = k
    results["x_array"] = X
    
    return results