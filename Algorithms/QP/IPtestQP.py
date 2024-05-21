import numpy as np
from InteriorPointQP2 import InteriorPointQP
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\QP\QP_Test.mat")

H_load = data["H"].squeeze().astype(float)
g = data["g"].squeeze().astype(float)
C_load = data["C"].squeeze().astype(float)
dl_load = data["dl"].squeeze().astype(float)
du_load = data["du"].squeeze().astype(float)
l_load = data["l"].squeeze().astype(float)
u_load = data["u"].squeeze().astype(float)

n = np.size(u_load)

# Inequality constraints
d = np.hstack([dl_load, -du_load, l_load, -u_load]).squeeze()
C = np.vstack([C_load.T, -C_load.T, np.eye(n), -np.eye(n)]).T
A = np.zeros([200, 0])
b = np.array([])
y = np.array([])
s = np.ones(len(d))  # slack variables
z = np.ones(len(d))  # inequality lagrange multiplier

def f(x1, x2):
    x = np.array([x1, x2])
    return 0.5 * x.T @ H_load @ x + g.T @ x

MaxIter = 10000
tol = 1e-2

# %% Interior-Point Algoritm
x0 = np.linalg.solve(H_load, -g)
res = InteriorPointQP(H_load, g, A, b, C, d, x0, y, z, s, MaxIter, tol)

X = res['x_array']
x0 = X[0, :]
xmin = res["xmin"]
print(res["converged"])
print(res["iter"])
print(res["min"])


