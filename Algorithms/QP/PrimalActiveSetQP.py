import numpy as np
from numpy.linalg import solve

"""
Solving Quadratic Program with equality constraints

min 0.5 x' H x + g' x
subject to A' x = b


"""

def EqualityQPSolver(H,g,A,b):
    n,m = A.shape
    
    KKT = np.block([[H, -A], [-A.T, np.zeros([m,m])]])
    
    y = np.hstack([-g,-b])
    
    z = solve(KKT,y)
    x = z[:n]
    lambdas = z[n:]
    return x, lambdas


#%%

"""
Primal Active Set Quadratic Programming algorithm

min 0.5 x' H x + g' x
subject to A' x >= b

Note it is not implemented with equality constrains 
"""


def PrimalActiveSetQP(H,g,A,b,x0,tol,MaxIter):
    
    x = np.copy(x0)    

    # Check which Inequality constraints is active (Working set)
    is_active = (A.T@x - b == 0)
    
    converged = False
    k = 0
    
    results = {}
    
    X = np.zeros([MaxIter,len(x0)])
    
    while converged == False and k < MaxIter:
        X[k,:] = x
        gk = H @ x + g
        A_active = A[:,is_active]
        n,m = A_active.shape
        p,ls = EqualityQPSolver(H,gk,A_active,np.zeros(m))
        
        if np.all(np.abs(p)<= tol): # check that p is the zero vector
            #check that equation 16.42 is satisfied
            if np.all(A_active @ ls - (H @ x + g) <= tol):
                if np.all(ls >= 0):
                    converged = True
                else:
                    j = np.argmin(ls)
                    i = np.where(is_active==True)[0][j]
                    is_active[i] = False
    
        else:
            not_active = ~is_active
            denom = A[:,not_active].T @ p
            is_negative_idx = np.where(denom < 0)[0]
            num = b[not_active]-A[:,not_active].T @ x 
            arr = num[is_negative_idx]/denom[is_negative_idx]
            
            alpha = np.min(np.minimum(1, arr))
            
            x += alpha * p
            
            # Check for blocking constraints
            is_active = (np.abs(A.T@x - b) <= tol)
            
        k += 1
    
    results['converged'] = converged
    results['x_min'] = x
    results['num_iter'] = k
    results['x_iter'] = X[:k,:]
    
    return results




