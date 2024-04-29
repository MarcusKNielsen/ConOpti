import numpy as np

def QP_example(example,PrintFormat=False):
    
    
    
    if PrintFormat==True:
        print("\n \n")
        print("""min 0.5 x' H x + g' x \n subject to C' x >= d \n \n you are given H, g, C, d \n \n""")
        
    
    
    if example == 1:
        
        # QP problem
        H = np.array([[20.88, 15.6], [15.6, 17.48]])
        g = np.array([-43.068, -21.908])
        C = np.array([[8.4, -4.0], [-1.0, 10.0]])
        d = np.array([-15.94, -8.2])
        
        # Initial guess
        x1 = 7.5
        x2 = 7.5
        x0 = np.array([x1,x2],dtype=np.float64)
        
        print(f"Example {example}")
        print("Optimal x: [ 1.94587573, -0.0416497 ] (by scipy optimize)")
        print("Optimal x: [ 1.94587573, -0.0416497 ] (by interior point)")
        print("\n \n")
        
        
    elif example == 2:
        
        # QP problem
        H = np.array([[1., 0.],[0., 1.]])
        g = np.array([-98.848, -36.704])
        C = np.array([[-3.6, -1. ],[-4.,  10. ]])
        d = np.array([ -0.84, -39.2 ])
        
        # Initial guess
        x1 = -5.0
        x2 = 0.0
        x0 = np.array([x1,x2],dtype=np.float64)
        
        print(f"Example {example}")
        print("Optimal x: [ 4.12998068, -3.50699043] (by scipy optimize)")
        print("Optimal x: [ 4.12999999, -3.507     ] (by interior point)")
        print("\n \n")
        
    elif example == 3:
        
        # QP problem
        H = 40*np.array([[1., 0.],[0., 1.]])
        g = np.array([-98.848, -36.704])
        C = np.array([[-3.6, -1. ],[-4.,  10. ]])
        d = np.array([ -0.84, -39.2 ])
        
        # Initial guess
        x1 = -5.0
        x2 = 0.0
        x0 = np.array([x1,x2],dtype=np.float64)
        
        print(f"Example {example}")
        print("Optimal x: [ 1.01345857, -0.70211272] (by scipy optimize)")
        print("Optimal x: [ 1.01345856, -0.70211271] (by interior point)")
        print("\n \n")
        
    return H,g,C,d,x0


