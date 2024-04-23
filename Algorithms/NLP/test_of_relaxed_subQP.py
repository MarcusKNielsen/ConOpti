"""
Implementation of Primal-Dual Predictor-Corrector Interior-Point Algorithm
"""
import numpy as np
from scipy.linalg import lu, solve, ldl, solve_triangular,qr,inv
from numpy.linalg import norm
from InteriorPointQP import InteriorPointQP, plotQP


# Initial point
x1 = 0.1
x2 = 0.1
x = np.array([x1,x2,2,2],dtype=np.float64)

# objective function
H = np.array([[20.88, 15.6 ],[15.6 , 17.48]])
g = np.array([-43.068, -21.908])

# equality constraints
A = np.zeros([len(x),0])
b = np.array([])
y = np.array([])             # equality lagrange multiplier

# Inequality constraints
C = np.array([[ 8.4, -4. ],[-1. , 10. ]])
d = np.array([-15.94,-8.2,0.0,0.0 ])
s = np.ones(len(d))          # slack variables
z = np.ones(len(d))          # inequality lagrange multiplier

#%%
mu = 2
num_ineq = len(g)
g = np.block([g,mu*np.ones(num_ineq)])

num_x = 2
H = np.block([[H,np.zeros([num_x,num_ineq])],[np.zeros([num_ineq,num_x]),0.1*np.eye(num_ineq)]])

I = np.eye(num_ineq)
nul = np.zeros([num_ineq,num_ineq])

C = np.block([[C,nul],[I,I]])



#%% Interior-Point Algorithm

MaxIter = 100
tol = 10**(-6)

res = InteriorPointQP(H,g,A,b,C,d,x,y,z,s,MaxIter, tol)
X = res['x_array']
x0 = X[0,:]
xmin = res['xmin']

plotQP(H,g,C,d,X)


