import numpy as np
from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, plotQP_eq
from himmelblau import plotHimmelblau

def check_optimality(x,Jac_f,z,g,Jac_g,y,h,Jac_h,tol):
    
    small_val = np.array([10**(-15)])
    h_exd = np.concatenate((h,small_val))
    g_exd = np.concatenate((g,small_val))
    
    stationarity    = np.max(np.abs(Jac_f.T - Jac_h.T @ y - Jac_g.T @ z)) < tol
    primal_fea_eq   = np.max(np.abs(h_exd)) < tol
    primal_fea_ineq = np.min(g_exd) >= -tol
    dual_fea        = np.min(z) >= -tol
    complementarity = np.abs(np.dot(z, g)) < tol
    
    return  stationarity & primal_fea_eq & primal_fea_ineq & dual_fea & complementarity

def BFGS_update_trust(B,xnxt,xnow,znxt,ynxt,df,dg_xnow,dg_xnxt,dh_xnow,dh_xnxt):
    
    p = xnxt - xnow
    dLnxt = df(xnxt).T - dg_xnxt.T @ znxt - dh_xnxt.T @ ynxt
    dLnow = df(xnow).T - dg_xnow.T @ znxt - dh_xnow.T @ ynxt
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

def radius_update(delta,rho,gamma,acceptstep,QP_failed):
    delta_max = 3
    if QP_failed == True:
        delta = min(3*delta,delta_max)
    else:
        if rho < 0.25 or gamma < 0 or acceptstep==False:
            delta = 0.5*delta
        elif rho > 0.75:
            delta = min(3*delta,delta_max)

    return delta
    

def solveSQP_Trust(x0,z0,y0,s0,f,g,h,df,dg,dh,d2f=None,d2g=None,d2h=None,MaxIter=100,tol=10**(-6), QPMaxIter = 100, QPtol = 10**(-2), LDL = True):   
    
    n_var  = len(x0) # Number of variables
    n_ineq = len(z0) # Number of inequality constraints
    n_eq   = len(y0) # Number of equality constraints
    
    def dg_exd(x,dg,n_var,n_ineq):
        C = np.zeros([n_var,n_ineq+2*n_var])
        C[:,:n_ineq] = dg(x).T
        C[:,n_ineq:n_ineq+n_var] = -np.eye(n_var)
        C[:,n_ineq+n_var:] = np.eye(n_var)
        return C

    def g_exd(x,g,n_var,n_ineq,delta):
        d = np.zeros(n_ineq+2*n_var)
        d[:n_ineq] = -g(x)
        d[n_ineq:] = -delta
        return d
    
    x = x0.copy()
    z = z0.copy()
    y = y0.copy()
    s = s0.copy()    
    
    k = 0 # accepts iteration counter
    j = 0 # while loop interation counter

    # Initialize arrays
    X = np.zeros([MaxIter+1,n_var])
    X[k,:] = x

    # Check for optimality (only inequality right now)
    converged = False
    
    # Trust region radius
    delta = 1
    
    z = np.zeros(n_ineq+2*n_var)
    z[:n_ineq] = z0.copy()
    z[n_ineq:] = 4
    
    s = np.zeros(n_ineq+2*n_var)
    s[:n_ineq] = s0.copy()
    s[n_ineq:] = 4
    
    # Initial evaluation
    q = df(x)
    A = dh(x).T
    b = -h(x)
    C = dg_exd(x,dg,n_var,n_ineq)
    d = g_exd(x,g,n_var,n_ineq,delta)
    C_old = C.copy()
    A_old = A.copy()
    
    while not converged and k < MaxIter and j < MaxIter:
        
        
        # Solve sub QP problem
        
        if d2f==None and d2g==None:
            if k == 0:
                H = np.eye(n_var)
            else:
                H = BFGS_update_trust(H,x,X[k-1,:],z,y,df,C_old.T,C.T,A_old.T,A.T)

        else:
            H = d2f(x) - np.sum(y[:, np.newaxis, np.newaxis] * d2h(x), axis=0) - np.sum(z[:n_var][:, np.newaxis, np.newaxis] * d2g(x), axis=0)
        

        results = InteriorPointQP(H, q, A, b, C, d, x, y, z, s, MaxIter=QPMaxIter, tol=QPtol, LDL = LDL)

        if results['converged'] != True:
            Exception("sub QP problem did not converge!")
        
#        plotQP_eq(H,q,C,d,A,b,X=results['x_array'],xlimits=[-10,10,-10,10])
        
        # Step quality
        dx = results['xmin']
        fnow = f(x)
        fnxt = f(x+dx)
        m = 0.5 * dx.T @ H @ dx + q @ dx
        rho = (fnow-fnxt)/(-m)
        gamma = (fnow-fnxt)/fnow

        if b.size != 0:
            hnow = h(x)
            hnxt = h(x+dx)
            if np.max(np.abs(hnow)) > 10**(-16):
                phi = max( (hnow-hnxt)/hnow )
            else:
                phi = - 10
                
            if phi >= gamma:
                gamma = phi
        
        acceptstep = min(rho,gamma)>=0
        QP_failed = results["converged"] == False
        
        delta = radius_update(delta,rho,gamma,acceptstep,QP_failed)
        
        if acceptstep == True:
            # update variables
            x += results['xmin']
            y  = results['lagrange_eq']
            z  = results['lagrange_ineq']
            s += results["slack"]
            
            # Save old C for BFGS
            C_old = C.copy()
            A_old = A.copy()
            
            # Update vectors and matrices
            q = df(x)
            A = dh(x).T
            b = -h(x)
            C = dg_exd(x,dg,n_var,n_ineq)
            d = g_exd(x,g,n_var,n_ineq,delta)
            
            # Check convergence
            converged = check_optimality(x,q,z[:n_var],g(x),C.T[:n_var,:n_var],y,h(x),A.T,tol)
            # update counter
            k += 1
            
            # Update arrays
            X[k,:] = x
            
        else:
            d = g_exd(x,g,n_var,n_ineq,delta)

#        plotHimmelblau(X=X[:(k+1),:],ext=True)
        j += 1
    
    
    X = X[:(k+1),:]
    
    results = dict()
    results["xmin"] = x
    results["slack"] = s
    results["lagrange_eq"] = y
    results["lagrange_ineq"] = z
    results["converged"] = converged
    results["iter"] = j
    results["x_array"] = X
    results["Nacceptstep"] = k
    
    
    return results


