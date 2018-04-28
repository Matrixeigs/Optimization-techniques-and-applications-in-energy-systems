"""
Mixed-integer programming using the CPLEX
"""
import cplex  # import the cplex solver package
import time
from numpy import ones
from cplex.exceptions import CplexError
import sys


def mixed_integer_linear_programming(c, Aeq=None, beq=None, A=None, b=None, xmin=None, xmax=None, vtypes=None,
                                     opt=None):
    t0 = time.time()
    if type(c) == list:
        nx = len(c)
    else:
        nx = c.shape[0]  # number of decision variables

    if A is not None:
        if A.shape[0] != None:
            nineq = A.shape[0]  # number of equality constraints
        else:
            nineq = 0
    else:
        nineq = 0

    if Aeq is not None:
        if Aeq.shape[0] != None:
            neq = Aeq.shape[0]  # number of inequality constraints
        else:
            neq = 0
    else:
        neq = 0
    # Fulfilling the missing informations
    if beq is None or len(beq) == 0: beq = -cplex.infinity * ones(neq)
    if b is None or len(b) == 0: b = cplex.infinity * ones(nineq)
    if xmin is None or len(xmin) == 0: xmin = -cplex.infinity * ones(nx)
    if xmax is None or len(xmax) == 0: xmax = cplex.infinity * ones(nx)

    # modelling based on the high level gurobi api
    try:
        prob = cplex.Cplex()
        prob.objective.set_sense(prob.objective.sense.minimize)
        # Declear the variables
        varnames = ["x" + str(j) for j in range(nx)]
        var_types = [prob.variables.type.continuous] * nx
        for i in range(nx):
            if vtypes[i] == "b" or vtypes[i] == "B":
                var_types[i] = prob.variables.type.binary
            elif vtypes[i] == "d" or vtypes[i] == "D":
                var_types[i] = prob.variables.type.integer
        prob.variables.add()
        prob.variables.add(obj=c.tolist(), lb=xmin, ub=xmax, types=var_types, names=varnames)
        # Populate by non-zero to accelerate the formulation
        rhs = beq.tolist() + b.tolist()
        sense = ['E'] * neq + ["L"] * nineq
        rows = []
        cols = []
        vals = []
        if neq != 0:
            for i in range(neq):
                for j in range(nx):
                    if Aeq[i, j] != 0:
                        rows.append(i)
                        cols.append(j)
                        vals.append(float(Aeq[i, j]))

        if nineq != 0:
            for i in range(nineq):
                for j in range(nx):
                    if A[i, j] != 0:
                        rows.append(i + neq)
                        cols.append(j)
                        vals.append(float(A[i, j]))

        prob.linear_constraints.add(rhs=rhs,
                                    senses=sense)
        prob.linear_constraints.set_coefficients(zip(rows, cols, vals))

        print(time.time() - t0)
        prob.solve()
        obj = prob.solution.get_objective_value()
        x = prob.solution.get_values()

        success = 1

    except CplexError:
        print(CplexError)
        x = 0
        obj = 0
        success = 0

    except AttributeError:
        print('Encountered an attribute error')
        x = 0
        obj = 0
        success = 0

    elapse_time = time.time() - t0
    print(elapse_time)
    return x, obj, success


if __name__ == "__main__":
    # A test problem from Gurobi
    #  maximize
    #        x +   y + 2 z
    #  subject to
    #        x + 2 y + 3 z <= 4
    #        x +   y       >= 1
    #  x, y, z binary

    from numpy import array
    from scipy.sparse import csr_matrix

    c = array([1, 1, 2])
    A = csr_matrix(array([[1, 2, 3],
                          [-1, -1, 0]]))  # A sparse matrix
    b = array([4, -1])
    vtypes = []
    vtypes.append('b')
    vtypes.append('b')
    vtypes.append('b')

    solution = mixed_integer_linear_programming(c, A=A, b=b, vtypes=vtypes)
    print(solution)
