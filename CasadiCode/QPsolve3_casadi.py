import casadi as ca

def solveQP(H, g, A, b, C, d):
    
    # Convert to casadi objects
    H = ca.DM(H) 
    g = ca.DM(g)
    A = ca.DM(A)
    b = ca.DM(b)
    C = ca.DM(C)
    d = ca.DM(d)
    
    # Number of variables
    n = H.shape[1]

    # Defining the optimization variable
    x = ca.MX.sym('x', n)

    # Objective function
    obj = 0.5 * ca.mtimes([x.T, H, x]) + ca.mtimes(g.T, x)

    # Constraints
    eq_constraints = ca.mtimes(A, x) - b
    ineq_constraints = ca.mtimes(C, x) - d

    # Setting up the QP
    qp = {'x': x, 'f': obj, 'g': ca.vertcat(eq_constraints, ineq_constraints)}
    opts = {'print_time': 0}  # remove solver output

    # Creating a QP solver
    solver = ca.qpsol('solver', 'qpoases', qp, opts)

    # Number of constaints
    num_eq_constr = b.numel()  # Number of equality constraints
    num_ineq_constr = d.numel()  # Number of inequality constraints

    # Constraints bounds (0 for equality, positive for inequality)
    lbg = [0] * num_eq_constr + [-ca.inf] * num_ineq_constr
    ubg = [0] * num_eq_constr + [0] * num_ineq_constr

    # Solving the QP
    sol = solver(lbg=lbg, ubg=ubg)

    return np.array(sol['x'])

if __name__ == "__main__":
    # Example
    
    import numpy as np
    H = np.array([[2, 0], [0, 2]])
    g = np.array([-1, -1])
    A = np.array([[1, 1]])
    b = np.array([1])
    C = np.array([[1, -1]])
    d = np.array([-0.5])
    
    
    # Solve the QP
    x = solveQP(H, g, A, b, C, d)
    print("Solution of the QP problem:", x)
