import numpy as np
import matplotlib.pyplot as plt
from himmelblau import *
from SQP import solveSQP

tol = 10**(-6)
MaxIter = 20

# Initial point
x = np.array([1.0,3.3],  dtype=float)
#x = np.array([1.25,0.7],  dtype=float)
#x = np.array([3.8,3.8],  dtype=float)
#x = np.array([-3.8,1.9], dtype=float)
#x = np.array([0.0,3.3],  dtype=float)

z = np.array([1.0,1.0] , dtype=float)*4
y = np.array([]        , dtype=float)
s = np.ones(2) * 2

results = solveSQP(f,df,d2f,g,dg,d2g,x,z,y,s,MaxIter = 100,tol=10**(-6))

X = results["x_array"]

plotHimmelblau(X=X)

