from load_market_problem import load_problem
from simplex_Algo import run_simplex
import matplotlib.pylab as plt
from scipy.optimize import linprog
import numpy as np
from Casadi_solve_problem import casadi_solve
from plot_curve_LP import plot_demand_supply_curve
from LP_interiorPiont import InteriorPointLP
from time import perf_counter

directory = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat"
A,g,b,U,C,Pd_max,Pg_max = load_problem(directory)

t = perf_counter()
res = linprog(g, A_eq=A, b_eq=b)
print("time linprog:", perf_counter()-t)
t = perf_counter()
sol,solx = casadi_solve(C,Pd_max,Pg_max,U)
print("time casadi:", perf_counter()-t)

t = perf_counter()
res_simplex = run_simplex(A,b,g)
print("time simplex", perf_counter()-t)
x_simplex = res_simplex["X all"][-1]

# Interior point solve   
m,n = A.shape
lam = np.ones(n) 
mu = np.zeros(m)
x = np.ones(n)*2.1
t = perf_counter()
res_interior = InteriorPointLP(A,g,b,x,mu,lam,MaxIter=10000,tol=1e-6)
print("time intertior", perf_counter()-t)

print("cost casadi",-1*sol["cost"])
print("cost linprog",-1*res.fun)
print("cost simplex:",-1*g@x_simplex)
print("cost interior simply:",-1*g@res_interior["xmin"])
print("difference in objective function solvers:",-1*sol["cost"]+res.fun)
print("difference in x arrays",np.max(np.abs(np.array(solx)-np.array(res.x))))

plot_demand_supply_curve(U,C,res.x,1,title = "using linprog")
plot_demand_supply_curve(U,C,res_interior["xmin"],2,title = "using interior-point")
plot_demand_supply_curve(U,C,np.array(solx),3,title = "using casadi")
plot_demand_supply_curve(U,C,x_simplex,4,title = "using active-set") 

fig, ax = plt.subplots(3, 1)
ax[0].semilogy(np.arange(res_interior["iterations"]),res_interior["rL"][1:],"b",label=r"$\Vert r_L\Vert_\infty$")
ax[0].set_xlabel("Iterations",fontsize=14)
ax[0].set_ylabel(r"$\log(\Vert r_L\Vert_\infty)$",fontsize=14)
ax[0].legend(fontsize=14)
ax[1].semilogy(np.arange(res_interior["iterations"]),res_interior["rA"][1:],"g",label=r"$\Vert r_A\Vert_\infty$")
ax[1].set_xlabel("Iterations",fontsize=14)
ax[1].set_ylabel(r"$\log(\Vert r_A\Vert_\infty)$",fontsize=14)
ax[1].legend(fontsize=14)
ax[2].semilogy(np.arange(res_interior["iterations"]),res_interior["rC"][1:],"r",label=r"$\Vert r_C\Vert_\infty$")
ax[2].set_xlabel("Iterations",fontsize=14)
ax[2].set_ylabel(r"$\log(\Vert r_C\Vert_\infty)$",fontsize=14)
ax[2].legend(fontsize=14)
fig.suptitle("KKT residuals as functions of iterations")
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()

