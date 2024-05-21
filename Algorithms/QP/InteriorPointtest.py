import numpy as np
from InteriorPointQP import plotQP_eq
from InteriorPointQP import InteriorPointQP as IP_marcus
from InteriorPointQP2 import InteriorPointQP as IP_mariana

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

MaxIter = 100
tol = 10**(-2)

#%% Interior-Point Algorithm

res = IP_mariana(H,g,A,b,C,d,x,y,z,s,MaxIter, tol)
X = res['X_results']
x0 = X[0,:]
xmin = res['xmin']
print(res["Converged"])

xlimits = [-5,5,-5,5]
print(res["iterations"])
plotQP_eq(H,g,C,d,A,b,X=X,xlimits=xlimits)



