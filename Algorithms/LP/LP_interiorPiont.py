import numpy as np

# InteriorPointLP
# This class solves a linear programming problem using the interior point method.

def InteriorPointLP(A,g,b,x,mu,lam,MaxIter=1000,tol=1e-6):

    # Initialize the variables
    n = len(x)
    m = A.shape[1] 
    e = np.ones(n)
    k = 0 # iteration counter
    Xres = np.zeros(MaxIter+1,len(x))
    Xres[0,:] = x
    
    converge = False
    
    while converge == False and k < MaxIter:
        Lam = np.diag(lam)
        X = np.diag(x)
        
        # Check for convergence (stop citeria)
        if np.linalg.norm(A@x-b) < tol and np.linalg.norm(g-A.T@mu-lam) < tol and np.linalg.norm(X@Lam) < tol:
            converge = True
        else:      
            vs = np.block([[np.zeros((m,n)),-A.T,-np.eye(m)],
                    [A, np.zeros((n,n)), np.zeros((n,n))],
                    [Lam, np.zeros((n,m)), X]])
            hs = np.vstack([-(g-A.T@mu-lam),-(A@x-b),-X@Lam@e])
            
            # Solving for the affine direction
            aff = np.linalg.solve(vs,hs)
            deltaxaff = aff[:n]
            deltamuaff= aff[n:n+m]
            deltalamaff = aff[n+m:]
            
            # finding the affine og alpha and beta
            d1 = -x/deltaxaff
            h1 = d1 >= 0
            alphaaff = max(h1)
            
            d2 = -lam/deltalamaff
            h2 = d2 >= 0
            betaaff = max(h2)
            
            # finding s affine
            saff = ((x+alphaaff*deltaxaff).T @ (lam+betaaff*deltalamaff))/n
            
            # finding sigma 
            s = (x.T@lam)/n
            sigma = (saff/s)**3
            
            # Solve for the search direction
            deltaXaff = np.diag(deltaxaff)
            deltaLamaff = np.diag(deltalamaff)
            hs1 = np.vstack([-(g-A.T@mu-lam),-(A@x-b),-X@Lam@e-deltaXaff@deltaLamaff@e+sigma*s*e])
            
            aff1 = np.linalg.solve(vs,hs1)
            deltax = aff1[:n]
            deltamu= aff1[n:n+m]
            deltalam = aff1[n+m:]
            
            # Finding alpa and beta
            d11 = -x/deltax
            h11 = d11 >= 0
            alpha = max(h11)
            
            d22 = -lam/deltalam
            h22 = d22 >= 0
            beta = max(h22)
            
            # Update x, mu and lam 
            nabla = 0.995
            
            x = x + nabla*alpha*deltax
            mu = mu + nabla*beta*deltamu
            lam = lam + nabla*beta*deltalam
            
            k = k + 1
            Xres[k,:] = x
    
    Xres = Xres[:(k+1),:]
    results = dict()
    results['xmin'] = x
    results['lam (lagrange_ineq)'] = lam
    results['mu (lagrange_eq)'] = mu
    results['X_results'] = Xres
    results['iterations'] = k
    results['Converged'] = converge
      
    return results
    
    

    
    
    
