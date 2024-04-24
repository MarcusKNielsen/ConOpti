import numpy as np
from QP_Examples import QP_example
from InteriorPointQP import plotQP
from solve_subQP import create_modified_QP
from scipy.optimize import minimize

# objective function
H,g,C,d,x = QP_example(1)

n_var = len(x)

H,g,A,b,C,d,x,y,z,s = create_modified_QP(H,g,d,C, x0 = x)


# Objective function
def objective(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(g, x)

# Constraint functions

Ct = C.T

def constraint1(x):
    return np.dot(Ct[0], x) - d[0]

def constraint2(x):
    return np.dot(Ct[1], x) - d[1]

def constraint3(x):
    return np.dot(Ct[2], x) - d[2]

def constraint4(x):
    return np.dot(Ct[3], x) - d[3]

# Define constraints in a format suitable for scipy.optimize
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3},
               {'type': 'ineq', 'fun': constraint4}]

# Call minimize function from scipy.optimize
result = minimize(objective, x, method='trust-constr', constraints=constraints, options={'verbose': 0})

# Print results
print("Optimal x:", result.x)

X = np.array([x,result.x])

H = H[:n_var,:n_var]
g = g[:n_var]
C = C[:n_var,:n_var]
d = -d[:n_var]

plotQP(H,g,C,d,X,title="Scipy Optimize")