
import numpy as np
from scipy.linalg import lu, solve, ldl, solve_triangular, qr
import random
from numpy.linalg import solve
from scipy.optimize import linprog

def feasibleInitialPoint(A, b):
    n, m = A.shape
    n1 = n
    A_bar = np.vstack([
        np.hstack([A.T, np.ones((m, 1)), -np.eye(m), np.zeros((m, m))]),
        np.hstack([-A.T, np.ones((m, 1)), np.zeros((m, m)), -np.eye(m)])
    ]).T
    b_bar = np.hstack([b, -b])
    g_bar = np.vstack([np.zeros((n, 1)), 1, np.zeros((2 * m, 1))])

    t0 = np.max(np.abs(b))

    x0 = np.hstack([np.zeros(n), t0, t0 - b, t0 + b])

    res = linprog(g_bar, A_eq=A_bar.T, b_eq=b_bar)
    x = res.x
    xstar = x[:n1]
    tstar = x[n1]
    sstar = x[n1 + 1:]

    if np.all(sstar == 0) and tstar == 0:
        return xstar
    else:
        print('No feasible point found')
        return []

def simplex(A,b,g,x):

    # Initalizations for the while loop
    max_iter = 100
    converged = False
    iter = 0
    tolerance = 1e-7

    # All the sets
    all_sets = np.arange(A.shape[1])
    
    # The index of the activate and non active set
    N_set = np.where(A@x-b < tolerance)[0]
    B_set = np.setdiff1d(all_sets, N_set)

    if B_set.size == 0:
          raise Exception("x is not a feasible point")

    # The active and the non active sets
    B = A[B_set,:]
    N = A[N_set,:]
    gN = g[N_set]
    gB = g[B_set]

    xN = solve(A[N_set,:],b[N_set])
    xB = solve(A[B_set,:],b[B_set])
    
    while not converged and iter < max_iter:

        # The active and the non active sets
        B = A[B_set]
        N = A[N_set]
        gN = g[N_set]
        gB = g[B_set]
        
        # Solving with LU 
        P,L,U = lu(B)
        #Y  = solve_triangular(P@L,gB,lower=True)
        mu = np.linalg.solve(B, gB)#solve_triangular(U,Y) 

        # Compute lambda_N
        lam_N = gN-N.T@mu

        # Checking if all lambda values are larger than 0
        if (lam_N >= 0).all():
                # Optimal solution sound 
                print("Optimal solution found.")
                converged = True
                # Returning lambda and x
                return [lam_N, xB, xN]
        else: 
                # Not sure with the index yet
                s = np.argmin(lam_N)
                i_s = N_set[s]
                a_i_s = N[i_s]
                
                # Solve for h
                Y  = solve_triangular(P@L,a_i_s,lower=True)
                h = np.linalg.solve(B, a_i_s) #solve_triangular(U,Y) 

                # The set of the indices which is to be added to the non basic set (The active set) 
                J = np.argmin(np.array([(xBi/hi) for xBi,hi in zip(xB,h) if hi>0]))
                
                # If there is no constraint to arrive at, the problem will continue to 
                # inifinity and the problem is unbounded
                if J.size == 0:
                      print("Unbounded problem, no solution")
                      converged = True
                      return False
                else:   
                      # Select an element from J, corresponding to the index to be 
                      # removed from the active set 
                      j = J # It justs equals because numpy already chose one

                      # Computing alpha (The scale size of the step length)
                      alpha = xB[j]/h[j]
                      # Updating xB and xN 
                      xB = xB - alpha*h # Moving 
                      xB[j] = 0 
                      xN[s] = alpha
                    
                      # Finding the index from the non active set 
                      i_j = B_set[j]
                      
                      # Changing the basic set by removing ij the index
                      B_set = np.setdiff1d(B_set, i_j)
                      # Adding the new index
                      B_set = np.append(B_set,i_s)
                      
                      # Changing the non basic set by removing the is index
                      N_set = np.setdiff1d(N_set, i_s)
                      # Adding the new index ij
                      N_set = np.append(N_set,i_j) 

        iter += 1
 
        
def PrimalActiveSetLPSolver(g, A, b, x):
    """
    This function is an implementation of the primal active-set algorithm for
    linear programs in standard form:

    min F(x) = g'x
    subject to: A'x = b
                x >= 0

    Parameters:
    g (numpy.ndarray): The cost vector.
    A (numpy.ndarray): The constraint matrix.
    b (numpy.ndarray): The right-hand side vector.
    x (numpy.ndarray): Initial feasible solution guess.

    Returns:
    numpy.ndarray: Optimal solution x or NaN if no solution is found.
    """
    n = len(x)
    max_it = n * 10
    tolerance = 1.0e-15

    # Find initial basic and non-basic sets
    N_i = np.where(np.abs(x) == 0)[0]
    B_i = np.where(np.abs(x) != 0)[0]

    B = A.T[B_i]
    N = A.T[N_i]

    converged = False
    iter = 0

    while True:
        mu = np.linalg.solve(B.T, g[B_i])
        lambda_N = g[N_i] - N.T @ mu

        # Check if optimal solution
        if not np.any(lambda_N < 0):
            print("Solver converged to a solution")
            converged = True
            break

        iter += 1

        min_lambda_N_i = np.argmin(lambda_N)
        h = np.linalg.solve(B, N[min_lambda_N_i])

        h_pos_i = np.where(h > 0)[0]
        if h_pos_i.size == 0:
            print("Unbounded problem, no solution")
            break

        x_B_i = x[B_i]
        alpha, j = min((x_B_i[h_pos_i] / h[h_pos_i], idx) for idx, _ in enumerate(h_pos_i))

        # Take step
        x[B_i] -= alpha * h
        x[B_i[j]] = 0
        x[N_i[min_lambda_N_i]] = alpha

        # Switch columns between basic and non-basic set
        temp = B_i[j]
        B_i[j] = N_i[min_lambda_N_i]
        N_i[min_lambda_N_i] = temp

        B = A[B_i]
        N = A[N_i]

        if iter == max_it:
            print("No convergence within maximum number of iterations")
            break

    if not converged:
        return np.nan

    return x

#A = np.array([[2,-1,1/2],[1,1,1]])
A = np.array([
    [2, 1],
    [-1, 1],
    [1/2, 1]
]) 

b = np.array([8,2,6])
g = np.array([1,3])

#x = feasibleInitialPoint(A, b)
x = np.array([2.666666667,4.666666667])

lam_N, xB, xN = simplex(A,b,g,x)
#x = PrimalActiveSetLPSolver(g,A,b,x)
print(x)



