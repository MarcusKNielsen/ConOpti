import numpy as np
import matplotlib.pyplot as plt

def plotLP(g,A,b,X=None,xlimits=None,title=None):
 
    def objective(x1,x2):
        return g[0]*x1 + g[1]*x2

    if not xlimits:
        x1min, x1max = -1,10
        x2min, x2max = -1,10
    else:
        x1min, x1max = xlimits[0],xlimits[1]
        x2min, x2max = xlimits[2],xlimits[3]

    # Create a grid of points.
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 400), np.linspace(x2min, x2max, 400))
    z = objective(x1, x2)
    
    # Check all constraints for each point in the grid
    feasible = ((x1 >= 0) & (x2 >=0))
    
    # Define the infeasible region
    infeasible_region = ~feasible
    
    # Create a mask for the infeasible region
    infeasible_mask = np.zeros_like(z)
    infeasible_mask[infeasible_region] = 1  # Mark infeasible region
    

    # Initialize the plot.
    plt.figure()

    # Plot the objective function.
    cs = plt.contourf(x1, x2, z, levels=10, cmap='viridis')
    plt.contourf(x1, x2, infeasible_mask, levels=[0.99, 1.01], colors='gray', alpha=0.8)
    plt.contour(cs, colors='k')
    plt.colorbar(cs)

    # Plot equality constraint
    x = np.linspace(x1min,x1max,1000)

    for i in range(A.shape[0]):
        plt.plot(x,(b[i]-A[i,0]*x)/A[i,1],"r--",label= f"constraint {i}")
    
    if X is not None:
        x1, x2 = X[:,0], X[:,1]
        plt.plot(x1[0],x2[0],"x",color="red",markersize=15,label=r"$x_0$: initial point")
        plt.plot(x1,x2,"-o",color="red")
        plt.plot(x1[-1],x2[-1],"x",color="blue",markersize=15,label=r"$x_{\text{star}}$: solution")
        plt.title(title)
        plt.legend()

    # Additional settings for the plot.
    plt.grid(c='k', ls='-', alpha=0.3)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.xlim([x1min, x1max])
    plt.ylim([x2min, x2max])
    plt.legend()

    # Show the plot.
    plt.show()