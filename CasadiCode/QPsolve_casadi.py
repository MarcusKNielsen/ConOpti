from casadi import *

x = SX.sym("x",2)

f = 3*x[0]**2 + 2*x[1]**2 + x[0]*x[1] + 3*x[0] + 2*x[1] + 4

g1 = x[0]
g2 = x[1]
g3 = x[0] + x[1] - 3

# Combine constraints into a vector
g = vertcat(g1, g2, g3)

qp = {'x': x, 'f': f, 'g': g}
solver = qpsol('S', 'qpoases', qp)

r = solver(lbg=[0,0,0])
x_opt = r['x']
print('x_opt: ', x_opt)

