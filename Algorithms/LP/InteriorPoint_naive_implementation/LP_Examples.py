import numpy as np

def LP_example(example,PrintFormat=False):
    
    
    if PrintFormat==True:
        print("\n")
        print("All examples are on the form:")
        print(""" min  g' x \n s.t. C' x >= d \n""")
        
    
    
    if example == 1:
        
        # LP problem
        g = np.array([-43.068, -21.908])
        C = np.array([[8.4, -4.0, -1.66], [-1.0, 10.0, -0.5]])
        d = np.array([-15.94, -8.2, -10.0])
        
        # Initial guess
        x1 = 1.0
        x2 = 10.0
        x0 = np.array([x1,x2],dtype=np.float64)
        
        # limites for plotting
        xlimits = [-5,10,-2.5,20]
        
    elif example == 2:
        
        # LP problem
        g = np.array([-98.848, 0])
        C = np.array([[-3.6, -1.0, 22.0 ],[-4.,  10.0, -1.0 ]])
        d = np.array([ -0.84, -39.2, -160.0 ])
        
        # Initial guess
        x1 = -5.0
        x2 = 0.0
        x0 = np.array([x1,x2],dtype=np.float64)
        
        # limites for plotting
        xlimits = [-10,5,-10,10]
        
    elif example == 3:
        
        # QP problem
        g = np.array([-98.848, -36.704])
        C = np.array([[-3.6, -1. ],[-4.,  10. ]])
        d = np.array([ -0.84, -39.2 ])
        
        # Initial guess
        x1 = -5.0
        x2 = 0.0
        x0 = np.array([x1,x2],dtype=np.float64)
        
        # limites for plotting
        xlimits = [-10,5,-10,10]

    if example == 4:
        g = np.array([0.23838225, 0.21255372])
        C = np.array([[1.62434536, -0.61175641]])
        d = np.array([20])
        x1 = 2
        x2 = 3
        x0 = np.array([x1,x2],dtype=np.float64)

        # limites for plotting
        xlimits = [-10,5,-10,10]
        

    if example == 5:
        g = np.array([-4,-2,0,0])
        A = np.array([[1,1,1,0],[2,0.5,0,1]])
        x1 = 2
        x2 = 3
        x0 = np.array([x1,x2],dtype=np.float64)
        xlimits = [-10,5,-10,10]

    return g,C,d,x0,xlimits


