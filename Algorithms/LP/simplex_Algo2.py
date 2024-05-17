import numpy as np

def PrimalActiveSetLPSolver(g, A, b, x):
    """
    This function is an implementation of the primal active-set algorithm for
    linear programs in standard form:

    min F(x) = g'x
    subject to: A'x = b
                x >= 0

    Parameters:
    g (numpy.ndarray): The cost vector.
    A (numpy.ndarray): The constraint matrix.
    b (numpy.ndarray): The right-hand side vector.
    x (numpy.ndarray): Initial feasible solution guess.

    Returns:
    numpy.ndarray: Optimal solution x or NaN if no solution is found.
    """
    n = len(x)
    max_it = n * 10
    tolerance = 1.0e-15

    # Find initial basic and non-basic sets
    B_i = np.where(np.abs(x) >= tolerance)[0]
    N_i = np.where(np.abs(x) < tolerance)[0]

    B = A[:, B_i]
    N = A[:, N_i]

    converged = False
    iter = 0

    while True:
        mu = np.linalg.solve(B.T, g[B_i])
        lambda_N = g[N_i] - N.T @ mu

        # Check if optimal solution
        if not np.any(lambda_N < 0):
            print("Solver converged to a solution")
            converged = True
            break

        iter += 1

        min_lambda_N_i = np.argmin(lambda_N)
        h = np.linalg.solve(B, N[:, min_lambda_N_i])

        h_pos_i = np.where(h > 0)[0]
        if h_pos_i.size == 0:
            print("Unbounded problem, no solution")
            break

        x_B_i = x[B_i]
        alpha, j = min((x_B_i[h_pos_i] / h[h_pos_i], idx) for idx, _ in enumerate(h_pos_i))

        # Take step
        x[B_i] -= alpha * h
        x[B_i[j]] = 0
        x[N_i[min_lambda_N_i]] = alpha

        # Switch columns between basic and non-basic set
        temp = B_i[j]
        B_i[j] = N_i[min_lambda_N_i]
        N_i[min_lambda_N_i] = temp

        B = A[:, B_i]
        N = A[:, N_i]

        if iter == max_it:
            print("No convergence within maximum number of iterations")
            break

    if not converged:
        return np.nan

    return x

# Example usage:
# Define A, b, g, x based on your specific problem.
# optimal_x = PrimalActiveSetLPSolver(g, A, b, x)

A = np.array([
    [1,1,1,0],
    [2,0.5,0,1]
]) 

A_a = 

b = np.array([5,8]).T
g = np.array([-4,-2,0,0]).T

#x = feasibleInitialPoint(A, b)
x = np.array([0,1,0,0])

x = PrimalActiveSetLPSolver(g,A,b,x)
print(x)