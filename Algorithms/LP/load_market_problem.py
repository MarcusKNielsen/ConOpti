import scipy.io
import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from LP_interiorPiont import InteriorPointLP
from Casadi_solve_problem import casadi_solve

def load_problem(directory):
    # Loading data
    mat = scipy.io.loadmat(directory)

    # Extracting data
    C = mat["C"].ravel()
    Pd_max = mat["Pd_max"].ravel()
    Pg_max = mat["Pg_max"].ravel()
    U = mat["U"].ravel()

    # Setting the A matrix
    # The first line as: [1,1,1,...,1,-1,-1,....,-1]
    A_firstline = np.concatenate([np.ones(len(U)),-1*np.ones(len(C))])

    # Slack matrix under the right side of the matrix, just an identity matrix swifting -1,1,-1,1
    slack_mat = np.eye((len(U)+len(C))*2)
    slack_mat[::2] *= -1
    # Setting zeros on top
    A_right_matrix = np.vstack([np.zeros(len(U)*2+len(C)*2),slack_mat])

    # Left side of matrix has a weird structure need loop
    A_left_matrix_temp = np.zeros([A_right_matrix.shape[1],A_firstline.shape[0]])
    for i in range(A_left_matrix_temp.shape[1]):
        A_left_matrix_temp[i*2,i*2-i] = 1
        A_left_matrix_temp[i*2+1,i*2-i] = 1

    # Stacking with the first line 
    A_left_matrix = np.vstack([A_firstline,A_left_matrix_temp])

    # Setting everything together
    A = np.hstack([A_left_matrix,A_right_matrix])

    # g matrix
    g = np.concatenate([-1*U,C,np.zeros(2*(len(U)+len(C)))]) # Changing sign because max to min
    # b matrix
    b_temp = np.zeros(A.shape[0]-2)

    Upper_bounds = np.hstack([Pd_max,Pg_max])

    b_temp[::2] += Upper_bounds
    b = np.hstack([np.zeros(2),b_temp])

    return A,g,b,U,C,Pd_max,Pg_max 