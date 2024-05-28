# import numpy as np
# import matplotlib.pyplot as plt
# from himmelblau import *
# from SQP_trust import solveSQP_Trust
# from SQP_line_extended import solveSQP_Line

# # Initial point
# x1,x2 = 1   , 2
# #x1,x2 = 1.3 , 3.7
# #x1,x2 = 0.7 , 2.9
# #x1,x2 = 3.7 , 3.7

# x = np.array([x1,x2],  dtype=float)

# y = np.array([] , dtype=float)

# z = np.array([1.0,1.0] , dtype=float)*4
# s = np.array([1.0,1.0] , dtype=float)*2

# #results = solveSQP_Line(x,z,y,s,f,g,h,df,dg,dh,MaxIter = 100, tol=10**(-2), LDL = False)
# results = solveSQP_Trust(x,z,y,s,f,g,df,dg,MaxIter = 20, tol=10**(-4))

# # unpack results
# converged = results["converged"]
# iterations = results["iter"]
# #accepted_steps = results["Nacceptstep"]
# X = results["x_array"]

# # check convergence and number of iterations
# print(f"converged = {converged}")
# print(f"iterations = {iterations}")
# #print(f"accepted steps = {accepted_steps}")
# # visualization of results
# #plotHimmelblau(X=X)

#%%

import numpy as np
import matplotlib.pyplot as plt
from himmelblau import *
from SQP_trust_extended import solveSQP_Trust
from SQP_line_extended import solveSQP_Line

# Initial point
#x1,x2 = 2   , 2
#x1,x2 = -1 , 0
#x1,x2 = 0.7 , 2.9
#x1,x2 = 3.8 , 3.0

# Initial point eq
x1,x2 = 3.8 , 1.6
x = np.array([x1,x2],  dtype=float)

y = np.array([1.0,1.0] , dtype=float)*4
#y = np.array([])

z = np.array([1.0,1.0] , dtype=float)*1
s = np.array([1.0,1.0] , dtype=float)*2

# Line search BFGS
results = solveSQP_Line(x,z,y,s,f,g,h_ext,df,dg,dh_ext,MaxIter = 100, tol=10**(-2), LDL = False)

# Line search Hessian
#results = solveSQP_Line(x,z,y,s,f,g,h,df,dg,dh,d2f,d2g,d2h,MaxIter = 100, tol=10**(-4), LDL = True)

# Trust region BFGS
#results = solveSQP_Trust(x,z,y,s,f,g,h_ext,df,dg,dh_ext, MaxIter = 5, tol=10**(-2), QPMaxIter = 100, QPtol = 10**(-2), LDL = False)

# unpack results
converged = results["converged"]
iterations = results["iter"]
#accepted_steps = results["Nacceptstep"]
X = results["x_array"]

# check convergence and number of iterations
print(f"converged = {converged}")
print(f"iterations = {iterations}")
#print(f"accepted steps = {accepted_steps}")
# visualization of results
plotHimmelblau(X=X,ext=True)
