import numpy as np

# InteriorPointLP
# This class solves a linear programming problem using the interior point method.

def InteriorPointLP(A,g,b,x,mu,lam,MaxIter=1000,tol=1e-6):

    # Initialize the variables
    m,n = A.shape 
    e = np.ones(n)
    k = 0 # iteration counter
    Xres = np.zeros([MaxIter+1,len(x)])
    Xres[0,:] = x

    converge = False
    
    while converge == False and k < MaxIter:
        Lam = np.diag(lam)
        X = np.diag(x)
        
        # Check for convergence (stop citeria)
        if np.linalg.norm(A@x-b) < tol and np.linalg.norm(g-A.T@mu-lam) < tol and np.linalg.norm(X@Lam) < tol:
            converge = True
        else:      
            vs = np.block([[np.zeros((n,n)),-A.T,-np.eye(n)],
                    [A, np.zeros((m,m)), np.zeros((m,n))],
                    [Lam, np.zeros((n,m)), X]])
            hs = np.concatenate([-(g-A.T@mu-lam),-(A@x-b),-X@Lam@e])
             
            # Solving for the affine direction
            aff = np.linalg.solve(vs,hs)
            deltaxaff = aff[:n]
            deltamuaff= aff[n:n+m]
            deltalamaff = aff[n+m:]
            
            # finding the affine og alpha and beta
            d1 = -x/deltaxaff
            if d1[d1 >= 0].size != 0:
                alphaaff = min(1,min(d1[d1 >= 0]))
            else:
                alphaaff = 1
            #h1 = d1 >= 0
            #alphaaff = max(h1)
            
            d2 = -lam/deltalamaff 
            #h2 = d2 >= 0

            if d2[d2 >= 0].size != 0:
                betaaff = min(1,min(d2[d2 >= 0]))
            else:
                betaaff = 1
            #betaaff = max(h2)
            
            # finding s affine
            saff = ((x+alphaaff*deltaxaff).T @ (lam+betaaff*deltalamaff))/n
            
            # finding sigma 
            s = (x.T@lam)/n
            sigma = (saff/s)**3
            
            # Solve for the search direction
            deltaXaff = np.diag(deltaxaff)
            deltaLamaff = np.diag(deltalamaff)
            hs1 = np.concatenate([-(g-A.T@mu-lam),-(A@x-b),-X@Lam@e-deltaXaff@deltaLamaff@e+sigma*s*e])
            
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
    
    


def InteriorPointLP_simplified(A,g,b,x,mu,lam,MaxIter=1000,tol=1e-6):
    
    """
    This version only takes an affine step which has been normalized
    """
    
    # Initialize the variables
    m,n = A.shape 
    e = np.ones(n)
    k = 0 # iteration counter
    Xres = np.zeros([MaxIter+1,len(x)])
    Xres[0,:] = x

    converge = False
    
    while converge == False and k < MaxIter:
        Lam = np.diag(lam)
        X = np.diag(x)
        
        # Check for convergence (stop citeria)
        if np.max(np.abs(A@x-b)) < tol and np.max(np.abs(g-A.T@mu-lam)) < tol and np.linalg.norm(X@Lam) and (x >= -tol).all() and (lam >= -tol).all():
            converge = True
        else:      
            vs = np.block([[np.zeros((n,n)),-A.T,-np.eye(n)],
                    [A, np.zeros((m,m)), np.zeros((m,n))],
                    [Lam, np.zeros((n,m)), X]])
            hs = np.concatenate([(A.T@mu+lam-g),-(A@x-b),-x*lam])
             
            # Solving for the affine direction
            aff = np.linalg.solve(vs,hs)
            aff = aff/np.linalg.norm(aff)
            deltaxaff = aff[:n]
            deltamuaff= aff[n:n+m]
            deltalamaff = aff[n+m:]            
            
            # Update x, mu and lam 
            nabla = 1
            
            nabla = min(1,np.linalg.norm(A@x-b))
            
            if k > (1-0.0025)*MaxIter:
                print(nabla)
            
            x = x + nabla*deltaxaff
            mu = mu + nabla*deltamuaff
            lam = lam + nabla*deltalamaff
            
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
    
    


def InteriorPointLP_simplified_mari(A,g,b,x,mu,lam,MaxIter=10000,tol=1e-9):
    
    # Initialize the variables
    m,n = A.shape 
    e = np.ones(n)
    k = 0 # iteration counter
    Xres = np.zeros([MaxIter+1,len(x)])
    Xres[k,:] = x
    eta = 0.99

    # Computing residuals
    rL = g-A.T@mu-lam
    rA = A@x-b
    rC = x*lam

    # Duality gap 
    s = np.sum(rC)/n
    
    converge = (np.max(np.abs(rL)) < tol and np.max(np.abs(rA)) < tol and np.abs(s) < tol)

    while converge == False and k < MaxIter:
        # Updating iterations
        k = k + 1

        #### Form an factorize hessian ####
        xdivlambda = x/lam
        H = A@np.diag(xdivlambda)@A.T
        L = np.linalg.cholesky(H)

        #### Affine step ####
        temp = (x*rL + rC)/ lam
        rhs = -rA + A@temp 

        dmu     = np.linalg.solve(L.T,np.linalg.solve(L,rhs))
        dx      = xdivlambda * (A.T@dmu) - temp 
        dlambda = -(rC+lam*dx)/x 

        # Step length
        idx = np.where(dx < 0)
        if (x[idx]/dx[idx]).size == 0:
            alpha = 1
        else:
            alpha = min(np.concatenate([np.array([1]),-x[idx]/dx[idx]])) 

        idx = np.where(dlambda <0)
        if (lam[idx]/dlambda[idx]).size == 0:
            beta = 1
        else:
            beta = min(np.concatenate([np.array([1]),-lam[idx]/dlambda[idx]]))

        #### Center parameters ####
        xAff        = x + alpha * dx
        lambdaAff   = lam + beta * dlambda
        sAff        = np.sum(xAff*lambdaAff)/n

        sigma   = (sAff/s)**3
        tau     = sigma * s

        #### Center corrector step ####
        rC = rC + dx*dlambda - tau  

        temp    = (x*rL+rC)/lam
        rhs     = -rA + A@temp

        dmu     = np.linalg.solve(L.T,np.linalg.solve(L,rhs))
        dx      = xdivlambda * (A.T@dmu) - temp 
        dlambda = -(rC+lam*dx)/x

        # step length
        idx = np.where(dx < 0)
        if (x[idx]/dx[idx]).size == 0:
            alpha = 1
        else:
            alpha = min(np.concatenate([np.array([1]),-x[idx]/dx[idx]])) 

        idx = np.where(dlambda <0)
        if (lam[idx]/dlambda[idx]).size == 0:
            beta = 1
        else:
            beta = min(np.concatenate([np.array([1]),-lam[idx]/dlambda[idx]]))

        ##### Take a step ####
        x   = x + (eta*alpha)*dx
        mu  = mu + (eta*beta)*dmu
        lam = lam + (eta*beta)*dlambda 

        # Updating output
        Xres[k,:] = x

        ### residuals and convergence check #### 
        # Computing residuals
        rL = g-A.T@mu-lam
        rA = A@x-b
        rC = x*lam

        # Duality gap 
        s = np.sum(rC)/n
        
        converge = (np.max(np.abs(rL)) < tol and np.max(np.abs(rA)) < tol and np.abs(s) < tol)
 
 
    Xres = Xres[:(k+1),:]
    results = dict()
    results['xmin'] = x
    results['lam (lagrange_ineq)'] = lam
    results['mu (lagrange_eq)'] = mu
    results['X_results'] = Xres
    results['iterations'] = k
    results['Converged'] = converge
      
    return results
    
    
