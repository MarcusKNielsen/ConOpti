import numpy as np

def InteriorPointLP(A,g,b,x,mu,lam,MaxIter=10000,tol=1e-9):
    
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

    # Saving of residuals
    rL_save = np.array([np.max(np.abs(rL))])
    rA_save = np.array([np.max(np.abs(rA))])
    rC_save = np.array([np.max(np.abs(rC))])

    # Duality gap 
    s = np.sum(rC)/n 
    
    converge = (np.max(np.abs(rL)) < tol and np.max(np.abs(rA)) < tol and np.abs(s) < tol)

    while converge == False and k < MaxIter:
        # Updating iterations
        k += 1

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

        # Saving
        rL_save = np.append(rL_save,np.max(np.abs(rL)))
        rA_save = np.append(rA_save,np.max(np.abs(rA)))
        rC_save = np.append(rC_save,np.max(np.abs(rC)))

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
    results['rL'] = rL_save
    results['rA'] = rA_save
    results['rC'] = rC_save
       
    return results
    
    
