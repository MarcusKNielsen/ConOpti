import numpy as np
from InteriorPointQP import InteriorPointQP, plotQP



# objective function
from QP_Examples import QP_example

example = 1
H,g,C,d,x = QP_example(example)

# equality constraints
A = np.zeros([len(x),0])
b = np.array([])
y = np.array([])             # equality lagrange multiplier

# Inequality constraints
s = np.ones(len(d))          # slack variables
z = np.ones(len(d))          # inequality lagrange multiplier


MaxIter = 100
tol = 10**(-6)

#%% Interior-Point Algorithm

res = InteriorPointQP(H,g,A,b,C,d,x,y,z,s,MaxIter, tol)
X = res['x_array']
x0 = X[0,:]
xmin = res['xmin']

plotQP(H,g,C,d,X,title=f"InteriorPointQP: Example {example}")


