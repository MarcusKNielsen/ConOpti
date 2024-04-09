from casadi import *

def SolveHimmelblau(x0):
    class MyCallback(Callback):
        def __init__(self, name, x_size, opts={}):
            Callback.__init__(self)
            self.x_size = x_size  # Expected size of x
            self.construct(name, opts)
            self.iterations = []  # List to store iterations
            
        def get_n_in(self): return nlpsol_n_out()
        def get_n_out(self): return 1
        def get_name_in(self, i): return nlpsol_out(i)
        def get_name_out(self, i): return "ret"
        
        def get_sparsity_in(self, i):
            if nlpsol_out(i) == 'x':
                return Sparsity.dense(self.x_size, 1)
            return Sparsity(0,0)
        
        def init(self):
            print("Initialize callback")
            
        def eval(self, arg):
            x = arg[0]  # Current solution
            self.iterations.append(x.full().flatten())  # Store current solution as a numpy array
            return [0]
    
    # Usage example
    x_size = 2  # Assuming 'x' is a 2-dimensional vector
    mycallback = MyCallback('mycallback', x_size)
    
    # Declare variables
    x = SX.sym("x", 2)
    
    # Form the NLP
    f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2  # Objective
    g1 = (x[0]+2)**2 - x[1]  # Original constraint
    g2 = -4*x[0] + 10*x[1]  # New constraint: 4x1 + 10x2 >= 0
    
    
    # Combine constraints into a vector
    g = vertcat(g1, g2)
    
    nlp = {'x': x, 'f': f, 'g': g}
    
    ''
    # Solver options
    opts = {"iteration_callback": mycallback, 'ipopt':{'tol': 1e-6}}
    
    # Setup the solver (assuming 'nlp' and other variables are defined as before)
    solver = nlpsol("solver", "ipopt", nlp, opts)
    
    # Solve the NLP with inequality constraints
    lbg = [0, 0]  # Lower bounds for g1 and g2
    
    sol = solver(x0 = x0, lbg=lbg)
    
    # Print solution
    print("-----")
    print("objective at solution = ", sol["f"])
    print("primal solution = ", sol["x"])
    print("dual solution (x) = ", sol["lam_x"])
    print("dual solution (g) = ", sol["lam_g"])

    X = np.array(mycallback.iterations)
    return X