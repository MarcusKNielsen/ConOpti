import pandas as pd
from casadi import *
import numpy as np
import matplotlib.pyplot as plt

def casadi_solve(C,Pd_max,Pg_max,U):

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

    return sol,sol["x"]

if __name__ == "__main__":
    
    sol,x = casadi_solve()

    print("cost = ",-1*sol["cost"]) 

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
    plt.show()


