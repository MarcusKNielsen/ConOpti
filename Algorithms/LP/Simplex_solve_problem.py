from load_market_problem import load_problem
from simplex_Algo import run_simplex,phase1_simplex
import matplotlib.pylab as plt
from scipy.optimize import linprog
import numpy as np
from Casadi_solve_problem import casadi_solve
from plot_curve_LP import plot_demand_supply_curve

directory = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat"
A,g,b,U,C = load_problem(directory)

res = linprog(g, A_eq=A, b_eq=b)
sol,solx = casadi_solve()
xstar,iter = phase1_simplex(A, b)

print("casadi solution of objective function",-1*sol["cost"])
print("linprog solution of objective function",-1*res.fun)
print("difference in objective function solvers:",-1*sol["cost"]+res.fun)
print("difference in x arrays",np.max(np.abs(np.array(solx)-np.array(res.x))))
print("iter",iter)
print("cost:",-1*g@xstar)

#plot_demand_supply_curve(U,C,xstar,1)
plot_demand_supply_curve(U,C,res.x,1)
plot_demand_supply_curve(U,C,np.array(solx),1)
plt.show()

