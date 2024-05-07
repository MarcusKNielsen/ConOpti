import numpy as np
from InteriorPointQP import InteriorPointQP, plotQP

def create_modified_QP(Bk,Jac_fk,gk,Jac_gk, x0 = None):
    # inputs: Bk, Jac_fk, gk, Jac_gk

    """
        min 0.5 x' H x + g' x
        subject to A' x  = b
                   C' x >= d 
    """

    # it is convinient to define
    n_var  = Bk.shape[0]
    n_ineq = gk.shape[0]
    n_eq   = 0
    n      = n_var+n_eq+n_eq+n_ineq
#    mu     = 0

    
    """
        Setup objective function
    """

    # Setup variable x
    if x0 is not None:
        p = x0
    else:
        p = np.ones(n_var)
        
    is_feasible = (Jac_gk @ p >= -gk).all()    # Missing equality
    if not is_feasible:
        v = np.ones(n_eq)
        w = np.ones(n_eq)
        t = np.ones(n_ineq) + np.abs(Jac_gk @ p - gk) 
        mu = 10
    else:
        mu = 0
        v = np.zeros(n_eq)
        w = np.zeros(n_eq)
        t = np.zeros(n_ineq)
    
    x = np.block([p,v,w,t])
    
    # Setup Hessian
    H = np.zeros([n,n])
    H[:n_var,:n_var] = Bk
    H[n_var:,n_var:] = 0.0*np.eye(2*n_eq+n_ineq)

    # Setup gradient tranposed: gt
    gt = np.zeros(n)
    gt[:n_var] = Jac_fk
    gt[n_var:] = mu
    g = gt.T    

    """
        Setup equality constraint
    """

    # We setup A transpose: At
    At = np.zeros([n_eq,n_var+n_eq+n_eq+n_ineq])
    At[:,n_var:(n_var+n_eq)] = (-1) * np.eye(n_eq)
    At[:,(n_var+n_eq):(n_var+2*n_eq)] = np.eye(n_eq)
    #At[:,:n_var] = (-1) * gradient of equality constraints transpose evaluated at xk
    A = At.T

    b = np.zeros(n_eq)
    # b = equality constraint evaluated at xk    

    """
        Setup inequality constraints
    """

    # We setup C transpose: Ct
    Ct = np.zeros([2*n_ineq+2*n_eq,n_var+n_eq+n_eq+n_ineq])
    Ct[:n_ineq,:n_var] = Jac_gk.T
    Ct[:n_ineq,(n_var+2*n_eq):] = np.eye(n_ineq)    
    Ct[n_ineq:(n_ineq+n_eq),n_var:(n_var+n_eq)] = np.eye(n_eq)
    Ct[(n_ineq+n_eq):(n_ineq+2*n_eq),(n_var+n_eq):(n_var+2*n_eq)] = np.eye(n_eq)
    Ct[(n_ineq+2*n_eq):,(n_var+2*n_eq):] = np.eye(n_ineq)
    C = Ct.T

    # we setup the d vector
    d = np.zeros(n_ineq + n_eq + n_eq + n_ineq)
    d[:n_ineq] = (-1) * gk

    """
        Initial guess
    """

    # The only missing part to use InteriorPointQP is y,z,s
    y = np.ones(n_eq)
    z = np.ones(2*n_ineq+2*n_eq)
    s = np.ones(2*n_ineq+2*n_eq)
    
    return H,g,A,b,C,d,x,y,z,s


def solve_subQP(Bk,Jac_fk,gk,Jac_gk,plot = False, x0 = None):
    
    #Setup modified QP problem 
    H,g,A,b,C,d,x,y,z,s = create_modified_QP(Bk,Jac_fk,gk,Jac_gk, x0)


    #Solve sub QP problem by Interior Point Algorithm
    results = InteriorPointQP(H,g,A,b,C,d,x,y,z,s, MaxIter = 25, tol = 10**(-6))

    
    
    if plot == True:
        n_var = len(x0)
        n_ineq = Jac_gk.shape[1]
        X_subQP = results['x_array'][:,:n_var]
        H = H[:n_var,:n_var]
        g = g[:n_var]
        C = C[:n_var,:n_ineq]
        d = d[:n_ineq]
        # print(f"H = {H}")
        # print(f"g = {g}")
        # print(f"C = {C}")
        # print(f"d = {d}")
        #plotQP(H, g, C, d, X_subQP)

        return results, H, g, C, d, X_subQP
    
    return results