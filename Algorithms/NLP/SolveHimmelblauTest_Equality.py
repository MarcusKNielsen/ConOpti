import numpy as np
import matplotlib.pyplot as plt
from himmelblau import *
from SQP_line import solveSQP_Line

solver = "SQP_line_BFGS"

title = r"Himmelblau's Test Problem Equality (SQP, Line, BFGS)"


plotHimmelblau(title=title,ext=True)


y = np.array([1.0,1.0] , dtype=float)*4
z = np.array([1.0,1.0] , dtype=float)*1
s = np.array([1.0,1.0] , dtype=float)*2


xs = np.array([[0.7, 0.7],[3.7 , 3.7]],dtype=float)
colors = ["red","gold","magenta","springgreen"]
labels = [r"$x_0 = (0.7,0.7)^\top$", r"$x_0 = (3.7 , 3.7)^\top$"]

MaxIter = 100
tol = 10**(-2)

for i in range(len(xs)):
    print(f"-- Test {i} --")
    x = xs[i]
    
    print(f"x = {x}")
    
    label = labels[i]
    color = colors[i]
    
    results = solveSQP_Line(x,z,y,s,f,g,h_ext,df,dg,dh_ext,MaxIter = MaxIter, tol=tol, LDL = False)
    
    X = results["x_array"]
    converged = results["converged"]
    iterations = results["iter"]
    print(f"converged = {converged}")
    print(f"iterations = {iterations}")
    plt.plot(X[:,0],X[:,1],"o-",color=color)
    x1, x2 = X[:,0], X[:,1]
    plt.plot(x1[0],x2[0],"x",color=color,markersize=15,label=label)
    print("\n")
    

plt.legend(loc="lower right")

# Show the plot.
plt.show()
