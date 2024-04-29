"""
Implementation of Primal-Dual Predictor-Corrector Interior-Point Algorithm
"""
import numpy as np
from scipy.linalg import lu, solve, ldl, solve_triangular,qr,inv
from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, plotQP

# Initial point
#x = np.array([1.2,1.5]) 
x = np.array([3.0,1.0])
#x = np.array([0.0,0.0])
x = np.array([2.0,2.0])
#x = np.array([0.1,0.1])
#x = np.array([4.0,1.0])

# objective function
H = np.array([[20.88, 15.6 ],[15.6 , 17.48]])
g = np.array([-43.068, -21.908])

# equality constraints
A = np.zeros([len(x),0])
b = np.array([])
y = np.array([])             # equality lagrange multiplier

# Inequality constraints
C = np.array([[ 8.4, -4. ],[-1. , 10. ]])
d = np.array([-15.94,  -8.2 ])
s = np.ones(len(d))          # slack variables
z = np.ones(len(d))          # inequality lagrange multiplier

def f(x1,x2):
    x = np.array([x1,x2])
    return 0.5 * x.T @ H @ x + g.T @ x


MaxIter = 100
tol = 10**(-6)

#%% Interior-Point Algorithm

res = InteriorPointQP(H,g,A,b,C,d,x,y,z,s,MaxIter, tol)
X = res['x_array']
x0 = X[0,:]
xmin = res['xmin']

plotQP(H,g,C,d,X)


