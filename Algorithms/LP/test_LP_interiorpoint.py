import scipy.io
import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from LP_interiorPiont import InteriorPointLP
from InteriorPoint_naive_implementation.InteriorPointLP import plotLP_2
from InteriorPoint_naive_implementation.LP_Examples import LP_example

n = 2 
m = 1 
np.random.seed(1) 
A = np.random.normal(0,1,(m,n)) 
x = np.zeros(n) 
x[:m] = abs(np.random.uniform(0,1,m)) 

lamb = np.zeros(n) 
lamb[m:] = abs(np.random.uniform(0,1,n-m)) 

mu = np.random.uniform(0,1,m) 

g = A.T@mu + lamb
b = A@x

x1 = np.array([20,40])

example = 4
g,A,b,x,xlimits = LP_example(example,PrintFormat=True)

m,n = A.shape
lamb1 = np.ones(n)
mu1 = np.zeros(m) 

results = InteriorPointLP(A,g,b,x1,mu1,lamb1,MaxIter=1000,tol=1e-6)
print(results['xmin'], results['iterations'], results['Converged'])

plotLP_2(g,A,b,X = results['X_results'],title=f"InteriorPointLP: Example",xlimits=[-50,50,-50,50])

