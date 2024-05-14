import scipy.io
import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from LP_interiorPiont import InteriorPointLP

# Loading data
mat = scipy.io.loadmat(r"C:\Users\maria\OneDrive - Danmarks Tekniske Universitet\Kandidat\1_semester\Constrained optimization\ConOpti\Algorithms\LP\LP_Test.mat")

# Extracting data
C = mat["C"].ravel()
Pd_max = mat["Pd_max"].ravel()
Pg_max = mat["Pg_max"].ravel()
U = mat["U"].ravel()

# Setting the A matrix
# The first line as: [1,1,1,...,1,-1,-1,....,-1]
A_firstline = np.concatenate([np.ones(len(U)),-1*np.ones(len(C))])

# Slack matrix under the right side of the matrix, just an identity matrix swifting -1,1,-1,1
slack_mat = np.eye((len(U)+len(C))*2)
slack_mat[::2] *= -1
# Setting zeros on top
A_right_matrix = np.vstack([np.zeros(len(U)*2+len(C)*2),slack_mat])

# Left side of matrix has a weird structure need loop
A_left_matrix_temp = np.zeros([A_right_matrix.shape[1],A_firstline.shape[0]])
for i in range(A_left_matrix_temp.shape[1]):
    A_left_matrix_temp[i*2,i*2-i] = 1
    A_left_matrix_temp[i*2+1,i*2-i] = 1

# Stacking with the first line 
A_left_matrix = np.vstack([A_firstline,A_left_matrix_temp])

# Setting everything together
A = np.hstack([A_left_matrix,A_right_matrix])

# g matrix
g_IP = np.concatenate([-1*U,C,np.zeros(2*(len(U)+len(C)))]) # Changing sign because max to min
# b matrix
b_IP_temp = np.zeros(A.shape[0]-2)

Upper_bounds = np.hstack([Pd_max,Pg_max])

b_IP_temp[::2] += Upper_bounds
b_IP = np.hstack([np.zeros(2),b_IP_temp])

m,n = A.shape
lam = np.ones(n)
mu = np.zeros(m)
x = np.ones(n)*2.1

result = InteriorPointLP(A,g_IP,b_IP,x,mu,lam,MaxIter=1000,tol=1e-6)
print(result["iterations"])
demand_sol = result['xmin'][:len(U)]
supply_sol = result['xmin'][len(U):len(U)+len(C)]

print("cost:",-1*g_IP@result["xmin"])
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
