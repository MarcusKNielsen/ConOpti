import numpy as np
import matplotlib.pyplot as plt
from himmelblau import *
from SQP_trust_extended import solveSQP_Trust
from SQP_line_extended import solveSQP_Line

#solver = "SQP_line_BFGS"
#solver = "SQP_line_Hess"
#solver = "SQP_trust_BFGS"
solver = "SQP_trust_Hess"

if solver == "SQP_line_BFGS":
    title = r"Himmelblau's Test Problem (SQP, Line, BFGS)"
elif solver == "SQP_line_Hess":
    title = r"Himmelblau's Test Problem (SQP, Line, $\nabla_x^2 \mathcal{L}$)"
elif solver == "SQP_trust_BFGS":
    title = r"Himmelblau's Test Problem (SQP, trust, BFGS)"
elif solver == "SQP_trust_Hess":
    title = r"Himmelblau's Test Problem (SQP, trust, $\nabla_x^2 \mathcal{L}$)"


plotHimmelblau(title=title)


y = np.array([])
z = np.array([1.0,1.0] , dtype=float)*1
s = np.array([1.0,1.0] , dtype=float)*2


xs = np.array([[-1.0, 0.0],[1.0, 3.8],[-3.5, 0.5],[0.8, 3.8]])
colors = ["red","magenta","gold","springgreen"]
labels = [r"$x_0 = (-1.0,0.0)^\top$", r"$x_0 = (1.0,3.8)^\top$", r"$x_0 = (-3.5,0.5)^\top$",r"$x_0 = (0.8,3.8)^\top$"]

MaxIter = 100
tol = 10**(-2)

for i in range(4):
    print(f"-- Test {i} --")
    x = xs[i]
    
    print(f"x = {x}")
    
    label = labels[i]
    color = colors[i]
    
    if solver == "SQP_line_BFGS":
        results = solveSQP_Line(x,z,y,s,f,g,h,df,dg,dh,MaxIter = MaxIter, tol=tol, LDL = True)
    elif solver == "SQP_line_Hess":
        results = solveSQP_Line(x,z,y,s,f,g,h,df,dg,dh,d2f,d2g,d2h,MaxIter = MaxIter, tol=tol, LDL = True)
    elif solver == "SQP_trust_BFGS":
        results = solveSQP_Trust(x,z,y,s,f,g,h,df,dg,dh,MaxIter = MaxIter, tol=tol, LDL = True)
    elif solver == "SQP_trust_Hess":
        results = solveSQP_Trust(x,z,y,s,f,g,h,df,dg,dh,d2f,d2g,d2h,MaxIter = MaxIter, tol=tol, LDL = True)
    
    
    X = results["x_array"]
    converged = results["converged"]
    iterations = results["iter"]
    print(f"converged = {converged}")
    print(f"iterations = {iterations}")
    print(f"x* = {X[-1]}")
    print(f"f(x*) = {f(X[-1])}")
    plt.plot(X[:,0],X[:,1],"o-",color=color)
    x1, x2 = X[:,0], X[:,1]
    plt.plot(x1[0],x2[0],"x",color=color,markersize=15,label=label)
    print("\n")
    

plt.legend(loc="lower right")

# Show the plot.
plt.show()