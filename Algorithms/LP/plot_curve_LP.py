
import numpy as np 
import matplotlib.pyplot as plt

def plot_demand_supply_curve(Ud,Cg,xstar,idx,title = " "):
    
    Pd = xstar[:len(Ud)]
    Pg = xstar[len(Ud):len(Ud)+len(Cg)]
 
    # Prepare data for plotting
    sorted_demand_indices = np.argsort(Ud)[::-1] # descending 
    sorted_supply_indices = np.argsort(Cg)       # increasing

    cum_Pd = np.cumsum(Pd[sorted_demand_indices])
    cum_Pg = np.cumsum(Pg[sorted_supply_indices])

    # Plotting 
    plt.figure(idx) 
    plt.step(cum_Pd, Ud[sorted_demand_indices],where="post", label="Demand curve")
    plt.step(cum_Pg, Cg[sorted_supply_indices], where="post", label="Supply curve")
    plt.plot([0, max(cum_Pd)], [16, 16],"--r",label="Market clearing price")
    plt.xlabel("Energy Quantity (MWh)")
    plt.ylabel("Price ($/MWh)")
    plt.title(f"Supply-Demand Curve {title}")
    plt.legend()
    plt.grid(True) 


