import numpy as np
from numpy.linalg import solve
from scipy.optimize import linprog

"""
Finding a feasible initial point for the 
Primal Active Set Quadratic Programming algorithm

min 0.5 x' H x + g' x
subject to A' x >= b

Note it is not implemented with equality constrains 

"""


def LPStandardForm(Al, bl, Au, bu, l, u):
    """
    Converts QP problem to the LP problem into standard form

    """

    nAl, mAl = Al.shape
    nAu, mAu = Au.shape
    nl = len(l)
    nu = len(u)

    A_bar = np.vstack(
        (
            np.hstack((-Al.T, Al.T, -np.eye(mAl), np.zeros((mAl, mAu + nl + nu)))),
            np.hstack(
                (
                    -Au.T,
                    Au.T,
                    np.zeros((mAu, mAl)),
                    np.eye(mAu),
                    np.zeros((mAu, nl + nu)),
                )
            ),
            np.hstack(
                (
                    -np.eye(nl),
                    np.eye(nl),
                    np.zeros((nl, mAl + mAu)),
                    -np.eye(nl),
                    np.zeros((nl, nu)),
                )
            ),
            np.hstack(
                (-np.eye(nu), np.eye(nu), np.zeros((nu, mAl + mAu + nl)), np.eye(nu))
            ),
        )
    ).T

    b_bar = np.concatenate((bl, bu, l, u))

    return A_bar, b_bar


def feasibleInitialPoint(A, b):
    """
    Finds a feasible initial point for the QP problem

    """

    n, m = A.shape
    n1 = n
    A_bar = np.vstack(
        [
            np.hstack([A.T, np.ones((m, 1)), -np.eye(m), np.zeros((m, m))]),
            np.hstack([-A.T, np.ones((m, 1)), np.zeros((m, m)), -np.eye(m)]),
        ]
    ).T
    b_bar = np.hstack([b, -b])
    g_bar = np.vstack([np.zeros((n, 1)), 1, np.zeros((2 * m, 1))])

    t0 = np.max(np.abs(b))

    x0 = np.hstack([np.zeros(n), t0, t0 - b, t0 + b])

    res = linprog(g_bar, A_eq=A_bar.T, b_eq=b_bar)
    x = res.x
    xstar = x[:n1]
    tstar = x[n1]
    sstar = x[n1 + 1 :]

    if np.all(sstar == 0) and tstar == 0:
        return xstar
    else:
        print("No feasible point found")
        return []


# %%
"""
Solving Quadratic Program without equality constraints

min 0.5 x' H x + g' x
subject to A' x ≤ b

"""


def EqualityQPSolver(H, g, A, b, xk):
    """
    Solves the equality constrained QP problem

    """
    n, m = A.shape
    # Set up the KKT system
    KKT = np.block([[H, -A], [-A.T, np.zeros([m, m])]])
    y = -np.block([g, b])
    # Solves the KKT system
    z = solve(KKT, y)
    x = z[:n]  # Solution
    lambdas = z[n:]  # Lagrange multipliers
    return x, lambdas


def computeAlpha(xk, wk, A, b, p):
    """
    Computes the distance to the nearest inactive set in the search direction

    """
    S = np.where(wk == False)[0]  # The set of inactive constraints
    denom = A[:, S].T @ p  # Compute the denominators
    keep_idx = denom < 0  # Remove constraints with negative denominators
    S = S[keep_idx]  # Update S
    denom = denom[keep_idx]  # Update denom

    # Compute alpha
    alphas = (-A[:, S].T @ xk + b[S]) / denom

    # Find the minimum alpha and the corresponding constraint
    idx = np.argmin(alphas)
    alpha = alphas[idx]
    con_idx = S[idx]

    return alpha, con_idx


def PrimalActiveSetQP(H, g, A, b, x0, tol, MaxIter):
    """
    Solves the QP problem with inequality constraints using the active set method

    min 0.5 x' H x + g' x
    s.t. A' x ≤ b

    """
    x = x0.copy()

    # Initializations
    is_active = np.zeros(A.shape[1], dtype=bool)
    converged = False
    k = 0
    results = {}
    X = np.zeros([MaxIter, len(x0)])

    while not converged and k < MaxIter:

        X[k, :] = x
        # Set up the equality problem
        gk = H @ x + g
        A_active = A[:, is_active]

        n, m = A_active.shape

        # Solve the equality constrained QP for search direction p
        p, ls = EqualityQPSolver(H, gk, A_active, np.zeros(m), x)

        if np.linalg.norm(p) <= tol:  # check that p is the zero vector
            if np.all(ls >= 0):
                # Check optimal solution is found
                converged = True
            else:
                # Find the constraint that is most violated
                j = np.argmin(ls)
                i = np.where(is_active == True)[0][j]
                is_active[i] = False

        else:
            # Compute alpha
            alpha, con_idx = computeAlpha(x, is_active, A, b, p)
            if alpha < 1:
                # Update x and activate constraint
                x += alpha * p
                is_active[con_idx] = True
            else:
                # Update x
                x += p
        k += 1

    # Save results
    results["converged"] = converged
    results["x_min"] = x
    results["num_iter"] = k
    results["x_iter"] = X[:k, :]
    results["lambda"] = ls

    return results
