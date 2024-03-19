from casadi import *

x = SX.sym("x",2)

f = 0.5*x[0]**2 + x[1]**2 - x[0]*x[1] - 2*x[0] - 6*x[1]

g1 = 2 - x[0] - x[1]
g2 = 2 + x[0] - 2*x[1]
g3 = 3 - 2*x[0] - x[1]

# Combine constraints into a vector
g = vertcat(g1, g2, g3)

qp = {'x': x, 'f': f, 'g': g}
solver = qpsol('S', 'qpoases', qp)

r = solver(lbg=[0,0,0])
x_opt = r['x']
print('x_opt: ', x_opt)

