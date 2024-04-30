
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags, eye, kron 
from scipy.linalg import lu, solve, ldl, solve_triangular, qr, lu_factor, lu_solve
from scipy.sparse.linalg import splu
from numpy.linalg import inv
import math

def construct_matrix(rows,cols,procent15):

    A = np.zeros([rows,cols])

    # Generated random indicies for non zero elements
    nonzero_indicies = np.random.choice(rows*cols,procent15,replace=False)
    for index in nonzero_indicies:     
        row = index // cols    
        col = index % cols 

        # Assigning standard normal distributed values to non zero indicies
        A[row, col] = np.random.normal(0,1)

    return A

def constructQP(n,alpha):

    beta        = 0.8
    m           = int(beta*n) 
    x_star      = np.random.normal(0,1,size=n)
    lamb_star   = np.random.normal(0,1,size=m)
    procent15   = int(n*m*0.15) # This number of values have to be non zero
    
    A = construct_matrix(n,m,procent15)
    M = construct_matrix(n,n,procent15)
    H = M@M.T + alpha*np.eye(n)

    KKT_mat = np.block([[H, -A], [-A.T, np.zeros((m, m))]])

    rhs = KKT_mat@np.block([x_star,lamb_star])

    g = -1*rhs[:len(x_star)]
    b = -1*rhs[len(x_star):]

    return H, g, A, b, x_star,lamb_star 

def construct_KKT(H, g, A, b):

    n, m = A.shape

    KKT_mat = np.block([[H, -A], [-A.T, np.zeros((m, m))]])
    rhs = np.block([-g, -b])

    return KKT_mat, rhs, n

def EqualityQPSolverLUdense(H, g, A, b):

    KKT_mat, rhs, n = construct_KKT(H, g, A, b)

    lu, piv = lu_factor(KKT_mat)
    sol = lu_solve((lu,piv),rhs)

    x = sol[:n]
    lam = sol[n:]

    return x,lam

def EqualityQPSolverLUsparse(H, g, A, b):
    
    KKT_mat, rhs, n = construct_KKT(H, g, A, b)
    
    LU_decomp = splu(KKT_mat)
    sol = LU_decomp.solve(rhs)

    x = sol[:n]
    lam = sol[n:]

    return x,lam 

def EqualityQPSolverLDLdense(H, g, A, b):

    KKT_mat, rhs, n = construct_KKT(H, g, A, b)

    L,D,perm = ldl(KKT_mat)
    rhs_permuted = rhs[perm]
    Y = solve_triangular(L,rhs_permuted,lower=True)
    sol = solve_triangular(D@L.T,Y)    

    x = sol[:n]
    lam = sol[n:]

    return x,lam

def EqualityQPSolverNULLSPACE(H, g, A, b):

    n,m = A.shape
    Q,R = qr(A)
    Q1 = Q[:,:m]
    Q2 = Q[:,m:]
    R = R[:m,:]
    xY = solve_triangular(R.T,b,lower=True) 
    xZ = solve(Q2.T@H@Q2,-Q2.T@(H@Q1@xY+g))
    x = Q1@xY+Q2@xZ
    lam = solve_triangular(R,Q1.T@(H@x+g))

    return x,lam

def EqualityQPSolverRANGESPACE(H, g, A, b):
    # Only good for small systems because the use of inverse of H

    v = solve(H,g)
    lam = solve(A.T@np.linalg.inv(H)@A,b+A.T@v)
    x = solve(H,A@lam-g)

    return x,lam

def EqualityQPSolver(H,g,A,b,solver):

    if solver == "LUdense":
        x,lam = EqualityQPSolverLUdense(H, g, A, b)
    elif solver == "LDLdense":
        x,lam = EqualityQPSolverLDLdense(H, g, A, b)
    elif solver == "Nullspace":
        x,lam = EqualityQPSolverNULLSPACE(H, g, A, b)
    elif solver == "Rangespace":
        x,lam = EqualityQPSolverRANGESPACE(H, g, A, b)
    elif solver == "LUsparse":
        x,lam = EqualityQPSolverLUsparse(H, g, A, b)
    elif solver == "LDLsparse":
        False

    return x,lam

def sensitivity(H,A):

    n, m = A.shape

    res_g = np.block([[np.array(np.identity(n))],[np.array(np.zeros((m,n)))]])
    res_b = np.block([[np.array(np.zeros((n,m)))],[np.array(np.identity(m))]])

    Mat = -inv(np.block([[H, -A], [-A.T, np.zeros((m, m))]]))

    resg = Mat@res_g 
    resb = Mat@res_b 

    return resg,resb














