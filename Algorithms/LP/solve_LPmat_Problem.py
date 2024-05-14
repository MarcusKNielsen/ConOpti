import scipy.io
import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from LP_interiorPiont import InteriorPointLP

mat = scipy.io.loadmat(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat")

C = mat["C"].ravel()
Pd_max = mat["Pd_max"].ravel()
Pg_max = mat["Pg_max"].ravel()
U = mat["U"].ravel()

# Sparsity of the LP linear term
A = Sparsity.dense(1+len(U)+len(C), len(U)+len(C))

# Create solver
solver = conic('solver', 'qpoases', {'a':A})
g = DM(np.concatenate([-1*U,C]))
a = DM(np.vstack([np.concatenate([np.ones(len(U)),-1*np.ones(len(C))]),np.eye(len(U)+len(C))]))

lba = DM(np.block([0,np.zeros(len(U)+len(C))]))
uba = DM(np.block([0,Pd_max,Pg_max]))

# Solving the problem
sol = solver(g=g, a=a, lba=lba, uba=uba) 
print("cost = ",-1*sol["cost"]) 

A_first_line = np.concatenate([np.ones(len(U)),-1*np.ones(len(C))])
slack_mat = np.eye((len(U)+len(C))*2)
slack_mat[::2] *= -1
A_right_matrix = np.vstack([np.zeros(len(U)*2+len(C)*2),slack_mat])


A_left_matrix = np.zeros([A_right_matrix.shape[1],A_first_line.shape[0]])

for i in range(A_left_matrix.shape[1]):
    A_left_matrix[i*2,i*2-i] = 1
    A_left_matrix[i*2+1,i*2-i] = 1

A = A_left_matrix


g_IP = np.concatenate([-1*U,C])
b_IP = np.zeros(1)
n,m = A_IP.shape
mu = np.zeros()

result = InteriorPointLP(A_IP,g_IP,b_IP,x,mu,lam,MaxIter=1000,tol=1e-6)

demand_sol = sol["x"][:len(U)]
supply_sol = sol["x"][len(U):]

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
#plt.show()


