import scipy.io
import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from LP_interiorPiont import InteriorPointLP,InteriorPointLP_simplified
from Casadi_solve_problem import casadi_solve
from load_market_problem import load_problem
 
#directory = r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat"
directory = r"/home/max/Documents/DTU/ConOpti/ConOpti/Algorithms/LP/LP_Test.mat"
A,g,b,U,C,Pd_max,Pg_max = load_problem(directory)

m,n = A.shape
lam = np.ones(n) 
mu = np.zeros(m)
x = np.ones(n)*2.1 

#result = InteriorPointLP(A,g_IP,b_IP,x,mu,lam,MaxIter=1000,tol=1e-6)
result = InteriorPointLP_simplified(A,g,b,x,mu,lam,MaxIter=10000,tol=1e-6)

print("iter",result["iterations"])
demand_sol = result['xmin'][:len(U)]
supply_sol = result['xmin'][len(U):len(U)+len(C)]

print("cost:",-1*g@result["xmin"])
# Prepare data for plotting
sorted_demand_indices = np.argsort(U)  # Sort demand by price in ascending order
sorted_supply_indices = np.argsort(-C)  # Sort supply by price in descending order

cumulative_demand = np.cumsum(demand_sol[sorted_demand_indices])
cumulative_supply = np.cumsum(supply_sol[sorted_supply_indices])

# Plotting
plt.figure(figsize=(10, 6))
plt.step(cumulative_demand, U[sorted_demand_indices],where="post", label="Cumulative Demand")
plt.step(cumulative_supply, C[sorted_supply_indices], where="post", label="Cumulative Supply")
#plt.plot(cumulative_demand, U[sorted_demand_indices], "o-", label="Cumulative Demand")
#plt.plot(cumulative_supply, C[sorted_supply_indices],"o-", label="Cumulative Supply")
plt.xlabel("Energy Quantity (MWh)")
plt.ylabel("Price ($/MWh)")
plt.title("Supply-Demand Curve")
plt.legend()
plt.grid(True)
plt.show()
