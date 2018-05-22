"""
Mixed-integer programming using the CPLEX
"""
from docplex.mp.model import Model
import cplex
from numpy import ones
from cplex.exceptions import CplexError


def linear_programming(c, Aeq=None, beq=None, A=None, b=None, xmin=None, xmax=None, vtypes=None,
                       opt=None):
    # t0 = time.time()
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

    if vtypes == None:
        vtypes = ["c"] * nx
    # Fulfilling the missing informations
    if beq is None or len(beq) == 0: beq = -cplex.infinity * ones(neq)
    if b is None or len(b) == 0: b = cplex.infinity * ones(nineq)
    if xmin is None or len(xmin) == 0: xmin = -cplex.infinity * ones(nx)
    if xmax is None or len(xmax) == 0: xmax = cplex.infinity * ones(nx)
    # Convert the data format
    try:
        c = c[:, 0]
        c = c.tolist()
    except IndexError:
        c = c.tolist()
    except:
        pass

    try:
        b = b[:, 0]
        b = b.tolist()
    except IndexError:
        b = b.tolist()
    except:
        pass

    try:
        beq = beq[:, 0]
        beq = beq.tolist()
    except IndexError:
        beq = beq.tolist()
    except:
        pass

    try:
        xmin = xmin[:, 0]
        xmin = xmin.tolist()
    except IndexError:
        xmin = xmin.tolist()
    except:
        pass

    try:
        xmax = xmax[:, 0]
        xmax = xmax.tolist()
    except IndexError:
        xmax = xmax.tolist()
    except:
        pass

    if neq == 0: beq = []
    if nineq == 0: b = []
    # modelling based on the high level cplex api
    try:
        model = Model()
        x = model.continuous_var_dict(nx, lb=xmin, ub=xmax, name=None)

        if neq != 0:
            for i in range(neq):
                expr = 0
                for j in range(nx):
                    if Aeq[i, j] != 0:
                        expr += Aeq[i, j] * x[j]
                model.add_constraint(expr == beq[i])
                # gurobi_model.addConstr(beq[i] == quicksum(Aeq[i, j] * x[j] for j in range(nx)))
                # gurobi_model.addConstr(x.prod(Aeq[i, :]) == beq[i])
                # print(i)

            # gurobi_model.addConstrs()
        # Inequal constraints
        if nineq != 0:
            for i in range(nineq):
                expr = 0
                for j in range(nx):
                    if A[i, j] != 0:
                        expr += A[i, j] * x[j]
                model.add_constraint(expr <= b[i])
        # Set the objective function
        obj = 0
        for i in range(nx):
            if c[i] != 0:
                obj+=c[i]*x[i]

        model.minimize(obj)

        solution = model.solve()
        # if solution.is_dual_feasible():
        #     print(solution.get_dual_values())
        # else:
        #     print("Dual problem is infeasible")
        #
        # if solution.is_primal_feasible():
        #     print(solution.get_values())
        # else:
        #     print("Primal problem is infeasible")
        x = [solution.get_value(x[i]) for i in range(nx)]
        obj = solution.get_objective_value()
        success = 1

    except CplexError:
        x = 0
        obj = 0
        success = 0
        print(CplexError)

    except AttributeError:
        print('Encountered an attribute error')
        x = 0
        obj = 0
        success = 0

    # elapse_time = time.time() - t0
    # print(elapse_time)
    return x, obj, success


if __name__ == "__main__":
    # A test problem from Gurobi
    #  maximize
    #        x +   y + 2 z
    #  subject to
    #        x + 2 y + 3 z <= 4
    #        x +   y       >= 1
    #  x, y, z binary

    from numpy import array, zeros

    c = array([4, 5, 6])

    A = array([[-1, -1, 0],
               [1, -1, 0],
               [-7, -12, 0]])

    b = array([-11, 5, -35])

    Aeq = array([[-1, -1, 1], ])
    beq = array([0])

    lb = zeros((3, 1))
    #
    vtypes = ["c"] * 3
    solution = linear_programming(c, A=A, b=b, Aeq=Aeq, beq=beq, xmin=lb, vtypes=vtypes)

    # c = array([1, 1])
    # A = array([[1, 1]])
    # b = array([11])
    # lb = array([6, 6])
    # solution = linear_programming(c, A=A, b=b, xmin=lb)

    print(solution)
