import casadi as ca
import numpy as np

"""
Solver for inequality constrained QP
"""

def solveQP(H, g, C, d):
    
    # Convert to casadi objects
    H = ca.DM(H) 
    g = ca.DM(g)
    C = ca.DM(C)
    d = ca.DM(d)
    
    # Number of variables
    n = H.shape[1]

    # Defining the optimization variable
    x = ca.MX.sym('x', n)

    # Objective function
    obj = 0.5 * ca.mtimes([x.T, H, x]) + ca.mtimes(g.T, x)

    # Constraints
    ineq_constraints = ca.mtimes(C.T, x) - d

    # Setting up the QP
    qp = {'x': x, 'f': obj, 'g': ca.vertcat(ineq_constraints)}
    opts = {'print_time': 0, 'printLevel': 'none'}

    # Creating a QP solver
    solver = ca.qpsol('solver', 'qpoases', qp, opts)

    # Number of constaints
    num_ineq_constr = d.numel()  # Number of inequality constraints

    # Constraints bounds (0 for equality, positive for inequality)
    lbg = [-ca.inf] * num_ineq_constr
    ubg = [0] * num_ineq_constr

    # Solving the QP
    sol = solver(lbg=lbg, ubg=ubg)

    return np.array(sol['x']).reshape(-1), np.array(sol['lam_x']).reshape(-1)

if __name__ == "__main__":
    
    #Example 1
    
    import numpy as np
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-1, -1])
    C = np.array([[1],[-1]])
    d = np.array([-0.5])
    
    
    # Solve the QP
    x, lam_x = solveQP(H, g, C, d)
    print("Solution of the QP problem:", x, lam_x)



