from load_market_problem import load_problem
from simplex_Algo import run_simplex,phase1_simplex
import matplotlib.pylab as plt
from scipy.optimize import linprog
import numpy as np
from Casadi_solve_problem import casadi_solve
from plot_curve_LP import plot_demand_supply_curve
from LP_interiorPiont import InteriorPointLP, InteriorPointLP_simplified,InteriorPointLP_simplified_mari

directory = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat"
#directory = r"/home/max/Documents/DTU/ConOpti/ConOpti/Algorithms/LP/LP_Test.mat"

A,g,b,U,C,Pd_max,Pg_max = load_problem(directory)

res = linprog(g, A_eq=A, b_eq=b)
sol,solx = casadi_solve(C,Pd_max,Pg_max,U)
#xstar,iter = phase1_simplex(A, b) 

# Interior point solve  
m,n = A.shape
lam = np.ones(n) 
mu = np.zeros(m)
x = np.ones(n)*2.1
#result = InteriorPointLP(A,g,b,x,mu,lam,MaxIter=1000,tol=1e-6) 
result2 = InteriorPointLP_simplified_mari(A,g,b,x,mu,lam,MaxIter=10000,tol=1e-6)

print("cost casadi",-1*sol["cost"])
print("cost linprog",-1*res.fun)
print("difference in objective function solvers:",-1*sol["cost"]+res.fun)
print("difference in x arrays",np.max(np.abs(np.array(solx)-np.array(res.x))))
print("iter",iter)
#print("cost simplex:",-1*g@xstar)
#print("cost interior:",-1*g@result["xmin"])
print("cost interior simply:",-1*g@result2["xmin"])

plot_demand_supply_curve(U,C,result2["xmin"],3)
#plot_demand_supply_curve(U,C,xstar,2)
plot_demand_supply_curve(U,C,res.x,4)
plot_demand_supply_curve(U,C,np.array(solx),1)
plt.show()

