import numpy as np
from numpy import flatnonzero as find
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp
from solvers.ccg_benders_decomposition import BendersDecomposition
from numpy import zeros

# Import the second stage problem
f = open("./modelCSV/hs", "r+")
hs = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/Ws", "r+")
Ws = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/Ts", "r+")
Ts = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/qs", "r+")
qs = np.loadtxt(f, delimiter=',')
f.close()

# Import the first stage optimization problem
f = open("./modelCSV/lbX", "r+")
lbX = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/ubX", "r+")
ubX = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/vtypeX", "r+")
vtypeX = np.loadtxt(f, delimiter=',', dtype='str')
f.close()

f = open("./modelCSV/objX", "r+")
c = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/Ax", "r+")
Ax = np.loadtxt(f, delimiter=',')
f.close()

f = open("./modelCSV/senseX", "r+")
senseX = np.loadtxt(f, delimiter=',', dtype='str')
f.close()

f = open("./modelCSV/rhsX", "r+")
rhsX = np.loadtxt(f, delimiter=',')
f.close()

Aeq = Ax[find(senseX == "="), :]
beq = rhsX[find(senseX == "=")]

A_negative = Ax[find(senseX == "<"), :]
b_negative = rhsX[find(senseX == "<")]
# A_positive = Ax[find(senseX == ">"), :]
# b_positive = rhsX[find(senseX == ">")]

# (xx, obj, success) = lp(c, Aeq=Aeq, beq=beq, A=A_negative, b=b_negative, xmin=lbX, xmax=ubX, vtypes=vtypeX.tolist())
# f = open("result_first_stage.txt", "w+")
# np.savetxt(f, xx, '%.18g', delimiter=',')
# f.close()
#
# f = open("result_first_stage.txt", "r+")
# xx_base = np.loadtxt(f, delimiter=',')
# f.close()
# solve the second stage problem
# (yy, obj_second_stage, success_seconstage) = lp(qs, Aeq=Ws, beq=hs - Ts.dot(xx), xmin=zeros((qs.shape[0])))

ps = [1]
Ws = [Ws]
qs = [qs.reshape((qs.shape[0], 1))]
Ts = [Ts]
hs = [hs.reshape((hs.shape[0], 1))]

benders_decomposition = BendersDecomposition()
sol = benders_decomposition.main(c=c.reshape((c.shape[0], 1)), lb=lbX.reshape((lbX.shape[0], 1)),
                                 ub=ubX.reshape((ubX.shape[0], 1)), Aeq=Aeq, beq=beq.reshape((beq.shape[0], 1)),
                                 A=A_negative, b=b_negative.reshape((b_negative.shape[0], 1)),
                                 vtype=vtypeX.tolist(), ps=ps, qs=qs, hs=hs,
                                 Ts=Ts, Ws=Ws)

print(sol)
