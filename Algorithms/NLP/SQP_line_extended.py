import numpy as np
from numpy.linalg import norm
from InteriorPointQP import plotQP
from InteriorPointQP import InteriorPointQP_v2 as InteriorPointQP


def check_optimality(x,Jac_f,z,g,Jac_g,y,h,Jac_h,tol):
    
    stationarity    = np.max(np.abs(Jac_f.T - Jac_h.T @ y - Jac_g.T @ z)) < tol
    primal_fea_eq   = np.max(np.abs(h)) < tol
    primal_fea_ineq = np.min(g) >= -tol
    dual_fea        = np.min(z) >= -tol
    complementarity = np.abs(np.dot(z, g)) < tol
    
    return  stationarity & primal_fea_ineq & dual_fea & complementarity


def line_search(x0,l,u,dx,f,df,g,h):
    
    x = x0.copy()
    alpha = 1
    i = 1
    Stop = False
    
    c = f(x) + l @ np.abs(h(x)) + u.T @ np.abs(np.minimum(0,g(x)))
    b = df(x) @ dx - l @ np.abs(h(x)) - u.T @ np.abs(np.minimum(0,g(x)))
    
    while not Stop and i < 25:
        x = x0 + alpha * dx
        phi = f(x) + l @ np.abs(h(x)) + u.T @ np.abs(np.minimum(0,g(x)))
        if phi <= c + 0.1 * b * alpha:
            Stop = True
        else:
            a = max((phi - (c+b*alpha))/alpha**2,10**(-14))
            alpha_min = -b/(2*a)
            alpha = np.min([0.9*alpha,np.max([alpha_min,0.1*alpha])])
    
    return alpha


def BFGS_update(B,xnxt,xnow,znxt,ynxt,df,dg,dh):
    
    p = xnxt - xnow
    dLnxt = df(xnxt).T - dg(xnxt).T @ znxt - dh(xnxt).T @ ynxt
    dLnow = df(xnow).T - dg(xnow).T @ znxt - dh(xnow).T @ ynxt
    q = dLnxt - dLnow
    
    temp = B @ p
    
    #B = B + q @ q.T/dot(p,p) - temp @ temp.T / (p.T @ temp)
    
    if p.T @ q >= 0.2 * p.T @ temp:
        theta = 1
    else:
        theta = (0.8 * p.T @ temp)/(p.T @ temp - p.T @ q)
    
    r = theta * q + (1-theta) * temp
    
    denom1 = max(p.T @ r,10**(-14))
    denom2 = max(p.T @ temp,10**(-14))
    B = B + np.outer(r,r)/denom1 - np.outer(temp,temp) / denom2
    
    return B
    
def solveSQP_Line(x0,z0,y0,s0,f,g,h,df,dg,dh,d2f=None,d2g=None,d2h=None,MaxIter = 100,tol=10**(-6)):   
    
    x = x0.copy()
    z = z0.copy()
    y = y0.copy()
    s = s0.copy()
    
    n_var  = len(x) # Number of variables
    #n_ineq = len(z) # Number of inequality constraints
    #n_eq   = len(y) # Number of equality constraints

    k = 0 # Iteration counter

    # Initialize arrays
    X = np.zeros([MaxIter+1,n_var])
    X[k,:] = x

    # Check for optimality (only inequality right now)
    converged = False
    
    while not converged and k < MaxIter:
        
        # Solve sub QP problem
        
        if d2f==None and d2g==None:
            if k == 0:
                H = np.eye(n_var)
            else:
                H = BFGS_update(H,x,X[k-1,:],z,y,df,dg,dh)
        else:
            H = d2f(x) - np.sum(z[:, np.newaxis, np.newaxis] * d2g(x), axis=0)
        
        q = df(x).T
        
        A = dh(x).T
        b = -h(x)
        
        C = dg(x).T
        d = -g(x)
        
        results = InteriorPointQP(H, q, A, b, C, d, x, y, z, s)
        
        if results['converged'] != True:
            Exception("sub QP problem did not converge!")
        
        #plotQP(H, q, C, d,results['x_array'],title=f"iter={k}")
        
        # Line search
        l = np.abs(y)
        u = np.abs(z)
        alpha = line_search(x,l,u,results['xmin'],f,df,g,h)
        
        # update variables
        x += alpha*results['xmin']
        y  = results['lagrange_eq']
        z  = results['lagrange_ineq']
        s += results["slack"]
        
        # Check convergence
        converged = check_optimality(x,df(x),z,g(x),dg(x),y,h(x),dh(x),tol)
        
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
