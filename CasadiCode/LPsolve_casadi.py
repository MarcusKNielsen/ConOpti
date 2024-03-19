from casadi import *

# minimize    3x + 4y
# subject to  x + 2y <= 14
#            3x -  y >= 0
#             x -  y <= 2


# Sparsity of the LP linear term
A = Sparsity.dense(3, 2)

# Create solver
solver = conic('solver', 'qpoases', {'a':A})
#solver = conic('solver', 'clp', {'a':A}) # Use clp

g = DM([3,4])
a = DM([[1, 2],[3, -1], [1, -1]])
lba = DM([-inf, 0, -inf])
uba = DM([14, inf, 2])

sol = solver(g=g, a=a, lba=lba, uba=uba)
print(sol)