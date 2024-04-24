import numpy as np
from scipy.optimize import minimize
from InteriorPointQP import plotQP

# Define the problem parameters

from QP_Examples import QP_example

H,g,C,d,x = QP_example(1)


# Objective function
def objective(x):
    return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(g, x)

# Constraint functions

Ct = C.T

def constraint1(x):
    return np.dot(Ct[0], x) - d[0]

def constraint2(x):
    return np.dot(Ct[1], x) - d[1]

# Define constraints in a format suitable for scipy.optimize
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2}]

# Call minimize function from scipy.optimize
result = minimize(objective, x, method='trust-constr', constraints=constraints, options={'verbose': 0})

# Print results
print("Optimal x:", result.x)
xmin = result.x
X = np.array([x,result.x])
plotQP(H,g,C,d,X,title="Scipy Optimize")
