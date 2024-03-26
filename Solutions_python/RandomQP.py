import scipy as sp
import numpy as np

def randomQP(n,alpha,density):
    m = 10*n
    A = sp.sparse.random(n,m,density)
    bl = -np.random.rand(m)
    bu = np.random.rand(m)
    M = sp.sparse.random(n,n,density)
    H = M@M+alpha*np.eye(n,n)
    g = np.random.rand(n)
    l = np.ones(n)
    u = np.ones(n)

    return H,g,bl,A,bu,l,u


n=10 
alpha=0.1 
density=0.15 
H,g,bl,A,bu,l,u = randomQP(n,alpha,density)
debug = True



