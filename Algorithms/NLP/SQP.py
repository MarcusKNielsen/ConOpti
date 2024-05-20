import numpy as np
from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, plotQP




def check_optimality(x,Jac_f,z,g,Jac_g,tol):
    
    stationarity    = norm(Jac_f.T - Jac_g.T @ z, np.inf) < tol
    primal_fea_ineq = np.min(g) >= 0
    dual_fea        = np.min(z) >= 0
    complementarity = np.abs(np.dot(z, g)) < tol
    
    return  stationarity & primal_fea_ineq & dual_fea & complementarity




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


def BFGS_update(B,xnxt,xnow,znxt,df,dg):
    
    p = xnxt - xnow
    dLnxt = df(xnxt).T - dg(xnxt).T @ znxt
    dLnow = df(xnow).T - dg(xnxt).T @ znxt
    q = dLnxt - dLnow
    
    temp = B @ p
    
    #B = B + q @ q.T/dot(p,p) - temp @ temp.T / (p.T @ temp)
    
    if p.T @ q >= 0.2 * p.T @ temp:
        theta = 1
    else:
        theta = (0.8 * p.T @ temp)/(p.T @ temp - p.T @ q)
    
    r = theta * q + (1-theta) * temp
    
    B = B + np.outer(r,r)/(p.T @ r) - np.outer(temp,temp) / (p.T @ temp)
    
    return B
    
def solveSQP(f,df,d2f,g,dg,d2g,x0,z0,y0,s0,MaxIter = 1000,tol=10**(-6)):
    
    x = x0.copy()
    z = z0.copy()
    y = y0.copy()
    s = s0.copy()
    
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
    
    while not converged and k < MaxIter:
        
        # Solve sub QP problem
        
        #H = d2f(x) - np.sum(z[:, np.newaxis, np.newaxis] * d2g(x), axis=0)
        if k == 0:
            H = np.eye(n_var)
        else:
            H = BFGS_update(H,x,X[k-1,:],z,df,dg)
        
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



