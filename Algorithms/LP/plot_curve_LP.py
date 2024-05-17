
import numpy as np 
import matplotlib.pyplot as plt

def plot_demand_supply_curve(U,C,xstar,idx):
    
    demand_sol = xstar[:len(U)]
    supply_sol = xstar[len(U):len(U)+len(C)]

    # Prepare data for plotting
    sorted_demand_indices = np.argsort(U)  # Sort demand by price in ascending order
    sorted_supply_indices = np.argsort(-C)  # Sort supply by price in descending order

    cumulative_demand = np.cumsum(demand_sol[sorted_demand_indices])
    cumulative_supply = np.cumsum(supply_sol[sorted_supply_indices])

    # Plotting
    plt.figure(idx)
    plt.step(cumulative_demand, U[sorted_demand_indices],where="post", label="Cumulative Demand")
    plt.step(cumulative_supply, C[sorted_supply_indices], where="post", label="Cumulative Supply")
    plt.xlabel("Energy Quantity (MWh)")
    plt.ylabel("Price ($/MWh)")
    plt.title("Supply-Demand Curve")
    plt.legend()
    plt.grid(True)


