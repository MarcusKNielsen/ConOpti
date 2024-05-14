
import numpy as np
from scipy.linalg import lu, solve, ldl, solve_triangular, qr
import random


def simplex(A,b,g,x):

    # Initalizations for the while loop
    max_iter = 10
    converged = False
    iter = 0

    # All the sets
    all_sets = np.arange(A.shape[0])
    
    # The index of the activate and non active set
    N_set = np.where(A.T@x - b == 0)[0]
    B_set = np.setdiff1d(all_sets, N_set)

    if N_set.size ==0:
          raise Exception("x is not a f point")

    # The active and the non active sets
    B = A[B_set].T
    N = A[N_set].T
    gN = g[N_set]
    gB = g[B_set]

    # Start value for xB
    xB = np.linalg.solve(B,b)# np.linalg.inv(B)@b
    # Start value for xN
    xN = np.zeros(len(N))
    
    while not converged and iter < max_iter:

        # The active and the non active sets
        B = A[B_set].T
        N = A[N_set].T
        gN = g[N_set]
        gB = g[B_set]
        
        # Solving with LU 
        P,L,U = lu(B)
        Y  = solve_triangular(P@L,gB,lower=True)
        mu = solve_triangular(U,Y) 

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
                h = solve_triangular(U,Y) 

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


A = np.array([[1,1,-5],[-1,5,1],[1,0,0],[0,1,0],[0,0,1]])
b = np.array([2,20,15])
g = np.array([1,-2,0,0,0])
x = np.array([2.5,4.5,0,0,0])

lam_N, xB, xN = simplex(A,b,g,x)
print(lam_N,xB,xN)






