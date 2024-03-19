from casadi import *

# Declare variables
x = SX.sym("x", 2)

# Form the NLP
f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2  # Objective
g1 = (x[0])**2 - x[1]  # Original constraint
g2 = 4*x[0] + 10*x[1]  # New constraint: 4x1 + 10x2 >= 0


# Combine constraints into a vector
g = vertcat(g1, g2)

nlp = {'x': x, 'f': f, 'g': g}

# Pick an NLP solver
MySolver = "ipopt"
# Solver options
opts = {}

# Allocate a solver
solver = nlpsol("solver", MySolver, nlp, opts)

# Define the initial guess
x0 = [1, 1]  # Example initial guess

# Solve the NLP with inequality constraints
lbg = [0, 0]  # Lower bounds for g1 and g2
sol = solver(x0 = x0, lbg=lbg)

# Print solution
print("-----")
print("objective at solution = ", sol["f"])
print("primal solution = ", sol["x"])
print("dual solution (x) = ", sol["lam_x"])
print("dual solution (g) = ", sol["lam_g"])

