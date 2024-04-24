"""
Test Interior Point Linear Program Algorithm
"""

import numpy as np
from InteriorPointLP import InteriorPointLP, plotLP



# objective function
from LP_Examples import LP_example

example = 3
g,C,d,x,xlimits = LP_example(example,PrintFormat=True)

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

res = InteriorPointLP(g,A,b,C,d,x,y,z,s,MaxIter, tol)
X = res['x_array']
x0 = X[0,:]
xmin = res['xmin']

plotLP(g,C,d,X,title=f"InteriorPointLP: Example {example}",xlimits=xlimits)


