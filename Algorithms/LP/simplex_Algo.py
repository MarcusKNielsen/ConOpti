import numpy as np

def phase1_simplex(A,b):

    tolerance = 1e-10

    m,n = A.shape

    A_bar = np.vstack([
        np.hstack([A, np.ones((m,1)), -np.eye(m), np.zeros((m,m))]),
        np.hstack([-A, np.ones((m,1)), np.zeros((m,m)), -np.eye(m)])
    ])

    b_bar = np.hstack([b, -b])

    g_bar = np.hstack([np.zeros(n), 1, np.zeros(2 * m)])

    t0 = np.max(np.abs(b))

    x0 = np.hstack([np.zeros(n), t0, t0 - b, t0 + b])

    res = phase2_simplex(A_bar, b_bar, x0, g_bar,0)
    
    if res["X all"].size != 0:
        print("No solution found")
        
    x = res["X all"][-1] 
    xstar = x[:n]
    tstar = x[n]
    sstar = x[n + 1:]

    if np.all(sstar < tolerance) and tstar < tolerance:
        print("phase1 done")
        return xstar,res["iter"]
    else:
        print('No feasible point found')
        return []
  

def phase2_simplex(A0,b0,x0,g0,iter0):

    x = x0.copy()
    A = A0.copy()
    b = b0.copy()
    g = g0.copy()

    # Initalizations for the while loop
    max_iter = 10000
    converged = False
    iter = 0 
    
    tolerance = 1e-15

    N_set = np.where(np.abs(x)<tolerance)[0]
    B_set = np.where(np.abs(x)>tolerance)[0]

    B = A[:,B_set]
    N = A[:,N_set]

    # Initializing X output
    X = np.zeros([max_iter+1,len(x)])
    X[0,:] = x 

    while not converged and iter < max_iter:
        
        mu = np.linalg.solve(B.T, g[B_set])
        lam_N = g[N_set] - N.T @ mu 

        # Checking if all lambda values are larger than 0
        if (lam_N >= 0).all() and lam_N.size > 0:

                # Optimal solution sound 
                print("Optimal solution found.")
                converged = True
                # Returning output
                result = dict()
                result["lambda N"] = lam_N
                result["xB"] = x[B_set]
                result["xN"] = x[N_set]
                result["mu"] = mu
                result["iter"] = iter+iter0
                result["X all"] = X[:iter+1]

                return result
        else: 
                
                # Not sure with the index yet
                s = np.argmin(lam_N)
                i_s = N_set[s]
                a_i_s = A[:,i_s]
                
                # Solve for h
                h = np.linalg.solve(B, a_i_s) 

                # The set of the indices which is to be added to the non basic set (The active set) 
                h_idx = np.where(h>0)[0]

                J = np.argmin(x[B_set[h_idx]]/h[h_idx])

                # If there is no constraint to arrive at, the problem will continue to 
                # inifinity and the problem is unbounded
                if J.size == 0:
                      print("Unbounded problem, no solution")
                      converged = True
                      return False
                else:   
                      # Select an element from J, corresponding to the index to be 
                      # removed from the active set 
                      j = h_idx[J] # It justs equals because numpy already chose one

                      # Computing alpha (The scale size of the step length)
                      alpha = x[B_set[j]]/h[j] 
                      # Updating xB and xN 
                      x[B_set] = x[B_set] - alpha*h # Moving 
                      x[B_set[j]] = 0
                      x[N_set[s]] = alpha
                      
                      # Changing the non basic set by removing the is index
                      z = B_set[j]
                      B_set[j] = N_set[s] 
                      N_set[s] = z 

                      # Updating the two sets
                      B = A[:,B_set]
                      N = A[:,N_set]

        iter += 1
        X[iter,:] = x
    
        if iter == max_iter:
             return False
 

def run_simplex(A,b,g):
    
    print("phase1 starts")
    xstar,iter = phase1_simplex(A, b)
    print("xstar:",xstar)
    print("phase1 done")

    # Solving the problem
    print("phase2 starts")

    result = phase2_simplex(A,b,xstar,g,iter)
    print("phase2 done")

    return result

if __name__ == "__main__":

    A = np.array([
        [1, 1, 1,0],
        [2, 1/2,0,1]
    ]) 
    b = np.array([5,8])
    g = np.array([-4,-2,0,0]) 

    result = run_simplex(A,b,g)

    lamN = result["lambda N"] 
    xB = result["xB"] 
    xN = result["xN"] 
    iter = result["iter"]
    X = result["X all"] 
    mu = result["mu"] 
    print("Iterations",iter)
    print("Lambda N",result["lambda N"])
    print("sol",X[-1])
    print("mu",mu) 

