import numpy as np
from InteriorPointQP import InteriorPointQP
from scipy.io import loadmat
import matplotlib.pyplot as plt
from casadi import *

# directory = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\QP\QP_Test.mat"
directory = r'/home/max/Documents/DTU/ConOpti/ConOpti/Algorithms/QP/QP_Test.mat'
data = loadmat(directory)


H_load = data["H"].squeeze().astype(float)
g = data["g"].squeeze().astype(float)
C_load = data["C"].squeeze().astype(float)
dl_load = data["dl"].squeeze().astype(float)
du_load = data["du"].squeeze().astype(float)
l_load = data["l"].squeeze().astype(float)
u_load = data["u"].squeeze().astype(float)

n = np.size(u_load)

# Solve with casadi
# Define the decision variables
x = SX.sym("x",n)
f = 1/2*dot(x,H_load@x) + dot(g,x)
c = C_load.T@x

# Define the problem
nlp = {'x': x, 'f': f, 'g': c}

# Create the solver
solver = nlpsol('solver', 'ipopt', nlp)

# Solve the problem
r = solver(lbg=dl_load, ubg=du_load, lbx=l_load, ubx=u_load)
x_opt = r['x']

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

MaxIter = 100
tol = 1e-4

# %% Interior-Point Algoritm
x0 = np.linalg.solve(H_load, -g)
res = InteriorPointQP(H_load, g, A, b, C, d, x0, y, z, s, MaxIter, tol)

X = res['x_array']
x0 = X[0, :]
xmin = res["xmin"]
print(res["converged"])
print(res["iter"])

print("difference", np.max(np.abs(xmin-x_opt)))

plt.figure(1)
plt.plot(x_opt)
plt.figure(2)
plt.plot(xmin)
plt.show()


