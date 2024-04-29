import numpy as np
from QP_Examples import QP_example
from InteriorPointQP import plotQP

# objective function
example = 1
H,g,C,d,x = QP_example(example)

# equality constraints
A = np.zeros([len(x),0])
b = np.array([])
y = np.array([])             # equality lagrange multiplier

# Inequality constraints
s = np.ones(len(d))          # slack variables
z = np.ones(len(d))          # inequality lagrange multiplier


#%%

# inputs

Bk = H
Jac_fk = g
gk = -d
Jac_gk = C

from solve_subQP import solve_subQP

results, H, g, C, d, X_subQP = solve_subQP(Bk,Jac_fk,gk,Jac_gk, plot=True, x0 = x)
plotQP(H, g, C, d, X_subQP, title=f"subQP: Example {example}")
X = results["x_array"]
xmin = results["xmin"][:len(x)]
print(xmin)

