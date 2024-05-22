import numpy as np
from InteriorPointQP import plotQP_eq
from InteriorPointQP import InteriorPointQP

x1 = -2
x2 = 2

x = np.array([x1,x2],dtype=float)

H = np.eye(2)
g = np.ones(2)

# inequality constraints
C = np.array([[-1, 0],
              [ 1, 0],
              [ 0,-1],
              [ 0, 1]],dtype=float).T

d = np.ones(4)*(-4)
z = np.ones(4)*2
s = np.ones(4)*2

# equality constraints
A = np.array([[ 1,-1],
              [-2, 1]],dtype=float).T

b = np.array([1.0,1.0])
y = np.array([4.0,4.0])

MaxIter = 3
tol = 10**(-6)

#%% Interior-Point Algorithm

res = InteriorPointQP(H,g,A,b,C,d,x,y,z,s,MaxIter, tol, LDL = False)
X = res['x_array']
x0 = X[0,:]
xmin = res['xmin']

xlimits = [-5,5,-5,5]
plotQP_eq(H,g,C,d,A,b,X=X,xlimits=xlimits)



