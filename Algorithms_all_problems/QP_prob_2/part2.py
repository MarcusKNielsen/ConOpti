from scipy.io import loadmat
from casadi import *
import numpy as np
import time
from casadi import SX, vertcat, nlpsol
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from scipy.io import savemat


from src.algorithms.PrimalActiveSetQP import *
from src.algorithms.InteriorPointQP import *

# %% Load and define the data

data = loadmat("QP_Test.mat")

H_load = data["H"].squeeze().astype(float)
g_load = data["g"].squeeze().astype(float)
C_load = data["C"].squeeze().astype(float)
dl_load = data["dl"].squeeze().astype(float)
du_load = data["du"].squeeze().astype(float)
l_load = data["l"].squeeze().astype(float)
u_load = data["u"].squeeze().astype(float)

n = np.size(u_load)


# %% Solving the solution using the library function IPOPT with Casadi

print("Solving the solution using the library function IPOPT with Casadi")

# Define the decision variables
x = SX.sym("x", n)
f = 1 / 2 * dot(x, H_load @ x) + dot(g_load, x)
c = C_load.T @ x

# Define the problem
nlp = {"x": x, "f": f, "g": c}

# Create the solver
solver = nlpsol("solver", "ipopt", nlp)

# Solve the problem
start = time.time()
r = solver(lbg=dl_load, ubg=du_load, lbx=l_load, ubx=u_load)
end = time.time()
print("Time to solve: ", end - start)

x_opt = r["x"]
print("x_opt: ", x_opt)

savemat("x_opt.mat", {"x_opt": x_opt})

# %% Solving the solution using the Primal Active-Set method

print("Solving the solution using the Primal Active-Set method")

# Convert the QP problem to the LP problem
A_bar, b_bar = LPStandardForm(C_load, dl_load, C_load, du_load, l_load, u_load)

# Find a feasible initial point
x_bar = feasibleInitialPoint(A_bar, b_bar)
x0 = x_bar[n : 2 * n] - x_bar[:n]

# Define the problem
A = np.vstack([C_load.T, -C_load.T, np.eye(n), -np.eye(n)]).T  # inequality constraints
b = np.hstack([dl_load, -du_load, l_load, -u_load])  # inequality constraints
g = g_load.squeeze()  # objective function
m = A.shape[1]

MaxIter = 1000
tol = 1e-6

# Solve the problem
start = time.time()
res = PrimalActiveSetQP(H_load, g, A, b, x0, tol, MaxIter)
end = time.time()
print("Time to solve: ", end - start)

x_opt = res["x_min"]
print("x_opt: ", x_opt)

# Save the solution
savemat("x_optPAS.mat", {"x_opt": x_opt})

# %% Solving the solution using the Primal-Dual Predictor-Corrector Interior-Point method

print(
    "Solving the solution using the Primal-Dual Predictor-Corrector Interior-Point method"
)

# Define the problem
d = np.hstack([dl_load, -du_load, l_load, -u_load]).squeeze()  # inequality constraints
C = np.vstack([C_load.T, -C_load.T, np.eye(n), -np.eye(n)]).T  # inequality constraints
A = np.zeros([200, 0])  # equality constraints
b = np.array([])  # equality constraints
y = np.array([])  # equality lagrange multiplier
s = np.ones(len(d))  # slack variables
z = np.ones(len(d))  # inequality lagrange multiplier

# Find initial point
x0 = np.linalg.solve(H_load, -g)
# Find intial values for the dual variables
x, y, z, s = InitialPointQP(H_load, g, A, b, C, d, x0, y, z, s)


# Define the objective function
def f(x1, x2):
    x = np.array([x1, x2])
    return 0.5 * x.T @ H_load @ x + g.T @ x


MaxIter = 1000
tol = 10 ** (-10)

# Solve the problem
start = time.time()
res = InteriorPointQP(H_load, g, A, b, C, d, x, y, z, s, MaxIter, tol)
end = time.time()
print("Time to solve: ", end - start)

# Solution
X = res["x_array"]
x0 = X[0, :]
xmin = res["xmin"]

print("x_opt: ", xmin)

# Save the solution
savemat("x_optIP.mat", {"x_opt": xmin})

# Residual plot

plt.subplot(131)
plt.plot(res["R_L"])
plt.yscale("log")
plt.title(r"$\|r_L\|_{\infty}$", fontsize=20)
plt.xlabel("Iterations", fontsize=14)
plt.yticks(fontsize=11)
plt.grid()
plt.gcf().set_size_inches(11, 6)

plt.subplot(132)
plt.plot(res["R_C"], color="red")
plt.yscale("log")
plt.title(r"$\|r_C\|_{\infty}$", fontsize=20)
plt.xlabel("Iterations", fontsize=14)
plt.yticks(fontsize=11)
plt.grid()

plt.subplot(133)
plt.plot(res["R_sz"], color="green")
plt.yscale("log")
plt.title(r"$\|r_{S\Lambda}\|_{\infty}$", fontsize=20)
plt.xlabel("Iterations", fontsize=14)
plt.yticks(fontsize=11)
plt.grid()

plt.savefig("residualsQP.pdf")
plt.show()
