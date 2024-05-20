import numpy as np
import matplotlib.pyplot as plt
from himmelblau import *
from SQP_trust import solveSQP_Trust
from SQP_line import solveSQP_Line

# Initial point
x1 = 0.0
x2 = 0.3
x = np.array([x1,x2],  dtype=float)

z = np.array([1.0,1.0] , dtype=float)*4
y = np.array([]        , dtype=float)
s = np.array([1.0,1.0] , dtype=float)*2

results = solveSQP_Line(x,z,y,s,f,g,df,dg,MaxIter = 20, tol=10**(-4))
#results = solveSQP_Trust(x,z,y,s,f,g,df,dg,MaxIter = 20, tol=10**(-4))

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
plotHimmelblau(X=X)

