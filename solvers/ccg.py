"""
Solvers for two-stage robust optimization problems
References:
    [1]Solving two-stage robust optimization problems using a column-and-constraint generation method
    The standard format is equation (1)

It should be noted that, the second stage problem is solved using dual problems

"""
from numpy import inf, array, concatenate, ones, zeros, eye, diag, hstack
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp


class TwoStageRobustOptimization():
    """
    column-and-constraint generation method for two-stage robust optimization
    one kind of primal-dual cuts methods
    """

    def __init__(self):
        self.name = " Robust optimization"

    def main(self, c, Aeq=None, beq=None, A=None, b=None, lb=None, ub=None, vtypes=None, d=None, G=None,
             E=None, M=None, h=None, u_mean=None, u_delta=None, budget=None):
        """
        :param c: First stage decision objective function
        :param Aeq: Equal constraints of the first stage decision making
        :param beq: Equal constraints of the first stage decision making
        :param A: Inequal constraints of the first stage decision making
        :param b: Inequal constraints of the first stage decision making
        :param lb: Lower boundary of the first stage decision making
        :param ub: Upper boundary of the first stage decision making
        :param vtypes: Variables types of the first stage decision making
        :param d: Objective function of the second stage decision making
        :param G: Inequality constraint with the second stage variables
        :param E: Inequality constraint with the first stage variables
        :param M: Inequality constraint with the uncertainty variables
        :param h: Inequality constants in the second stage decision making
        :param u_mean: mean value of the uncertainty variables
        :param u_delta: boundary information of the uncertainty variables
        :param budget: budget constraints on the uncertainty variables
        :return:
        """

        if Aeq is not None:
            neq = Aeq.shape[0]
        else:
            neq = 0
        if A is not None:
            nineq = A.shape[0]
        else:
            nineq = 0

        bigM = 10 ** 8
        # Modify the first stage decision variable
        ny = c.shape[0]
        c_first_stage = concatenate([c, array([[1]])], axis=0)  # Column wise append
        lb_first_stage = concatenate([lb, array([[-bigM]])], axis=0)
        ub_first_stage = concatenate([ub, array([[bigM]])], axis=0)
        Aeq_first_stage = hstack([Aeq, zeros((neq, 1))])
        beq_first_stage = beq
        A_first_stage = hstack([A, zeros((nineq, 1))])
        b_first_stage = b
        vtypes.append("c")
        # Solve the first stage optimization problem
        (yy, obj_first_stage, success_first_stage) = lp(c_first_stage,
                                                        Aeq=Aeq_first_stage,
                                                        beq=beq_first_stage,
                                                        A=A_first_stage,
                                                        b=b_first_stage,
                                                        xmin=lb_first_stage,
                                                        xmax=ub_first_stage,
                                                        vtypes=vtypes, objsense="min")

        # Modify the second stage decision making problems
        # U = u_mean - u_delta + 2*I*u_delta
        y = array(yy[0:ny]).reshape((ny, 1))
        nx = d.shape[0]
        nu = u_mean.shape[0]
        nlam = G.shape[0]  # The number of auxilary variables
        lb_second_stage = zeros((nlam + nu, 1))
        lb_second_stage = concatenate([lb_second_stage, -bigM * ones((nu, 1))], axis=0)
        ub_second_stage = concatenate([bigM * ones((nlam, 1)), ones((nu, 1)), bigM * ones((nu, 1))], axis=0)
        vtypes_second_stage = ["c"] * nlam + ["b"] * nu + ["c"] * nu
        c_second_stage = zeros((nlam + nu + nu, 1))
        c_second_stage[0:nlam] = h - E.dot(y) - M.dot(u_mean - u_delta)
        c_second_stage[nlam + nu:] = -2 * u_delta
        # The constraint set
        # Equal constraints
        Aeq_second_stage = zeros((nx, nlam + nu + nu))
        beq_second_stage = d
        Aeq_second_stage[:, 0:nlam] = G.transpose()
        # The McCormick envelopes
        # 1)
        A_second_stage = zeros((nu, nlam + nu + nu))
        b_second_stage = zeros((nu, 1))
        A_second_stage[:, nlam:nlam + nu] = -bigM * eye(nu)
        A_second_stage[:, nlam + nu:] = -eye(nu)
        # 2)
        A_second_stage_temp = zeros((nu, nlam + nu + nu))
        b_second_stage_temp = bigM * ones((nu, 1))
        A_second_stage_temp[:, 0:nlam] = M.transpose()
        A_second_stage_temp[:, nlam:nlam + nu] = bigM * eye(nu)
        A_second_stage_temp[:, nlam + nu:] = -eye(nu)
        A_second_stage = concatenate([A_second_stage, A_second_stage_temp], axis=0)
        b_second_stage = concatenate([b_second_stage, b_second_stage_temp], axis=0)
        # 3)
        A_second_stage_temp = zeros((nu, nlam + nu + nu))
        b_second_stage_temp = bigM * ones((nu, 1))
        A_second_stage_temp[:, 0:nlam] = -M.transpose()
        A_second_stage_temp[:, nlam:nlam + nu] = bigM * eye(nu)
        A_second_stage_temp[:, nlam + nu:] = eye(nu)
        A_second_stage = concatenate([A_second_stage, A_second_stage_temp], axis=0)
        b_second_stage = concatenate([b_second_stage, b_second_stage_temp], axis=0)
        # 4)
        A_second_stage_temp = zeros((nu, nlam + nu + nu))
        b_second_stage_temp = zeros((nu, 1))
        A_second_stage_temp[:, nlam:nlam + nu] = bigM * eye(nu)
        A_second_stage_temp[:, nlam + nu:] = eye(nu)
        A_second_stage = concatenate([A_second_stage, A_second_stage_temp], axis=0)
        b_second_stage = concatenate([b_second_stage, b_second_stage_temp], axis=0)
        # 5) budget constraints
        A_second_stage_temp = zeros((1, nlam + nu + nu))
        A_second_stage_temp[0, nlam:nlam + nu] = ones((1, nu))
        b_second_stage_temp = budget
        A_second_stage = concatenate([A_second_stage, A_second_stage_temp], axis=0)
        b_second_stage = concatenate([b_second_stage, b_second_stage_temp], axis=0)

        (x, obj_second_stage, success_second_stage) = lp(c_second_stage,
                                                         Aeq=Aeq_second_stage,
                                                         beq=beq_second_stage,
                                                         A=A_second_stage,
                                                         b=b_second_stage,
                                                         xmin=lb_second_stage,
                                                         xmax=ub_second_stage,
                                                         vtypes=vtypes_second_stage,
                                                         objsense="max")

        LB = obj_first_stage
        UB = ((c_first_stage[0:ny].transpose()).dot(y) + obj_second_stage)[0][0]
        Gap = abs((UB - LB) / LB)
        k = 0
        kmax = 1000

        while Gap > 10**-2:
            # Solve the first stage problem
            (yy, obj_first_stage, success_first_stage) = lp(c_first_stage,
                                                            Aeq=Aeq_first_stage,
                                                            beq=beq_first_stage,
                                                            A=A_first_stage,
                                                            b=b_first_stage,
                                                            xmin=lb_first_stage,
                                                            xmax=ub_first_stage, vtypes=vtypes, objsense="min")
            y = array(yy[0:ny]).reshape((ny, 1))
            # Update the second stage problem
            c_second_stage[0:nlam] = h - E.dot(y) - M.dot(u_mean - u_delta)
            # solve the second stage problem
            (x, obj_second_stage, success_second_stage) = lp(c_second_stage,
                                                             Aeq=Aeq_second_stage,
                                                             beq=beq_second_stage,
                                                             A=A_second_stage,
                                                             b=b_second_stage,
                                                             xmin=lb_second_stage,
                                                             xmax=ub_second_stage,
                                                             vtypes=vtypes_second_stage,
                                                             objsense="max")
            # Update gap
            LB = obj_first_stage
            UB = ((c_first_stage[0:ny].transpose()).dot(y) + obj_second_stage)[0][0]
            Gap = abs((UB - LB) / LB)
            print("The upper boundary is {0}".format(UB))
            print("The lower boundary is {0}".format(LB))
            print("The gap is {0}".format(Gap))

            if float(Gap) < 10**-2:
                break
            # Obtain cuts
            x = array(x).reshape((len(x), 1))
            Iu = x[nlam:nlam + nu]
            u = u_mean - u_delta + 2 * diag(Iu) * u_delta
            # Add cuts to the first stage optimization problem
            nx_temp = c_first_stage.shape[0]  # Previous decision variables
            c_first_stage = concatenate([c_first_stage, zeros((nx, 1))], axis=0)  # Column wise append
            lb_first_stage = concatenate([lb_first_stage, -bigM * ones((nx, 1))], axis=0)
            ub_first_stage = concatenate([ub_first_stage, bigM * ones((nx, 1))], axis=0)
            vtypes_temp = ["c"] * nx
            vtypes += ["c"] * nx
            neq = Aeq_first_stage.shape[0]
            Aeq_first_stage = hstack([Aeq_first_stage, zeros((neq, nx))])
            nineq = A_first_stage.shape[0]
            A_first_stage = hstack([A_first_stage, zeros((nineq, nx))])

            # Add primal cuts
            A_first_stage_temp = zeros((h.shape[0], nx_temp + nx))
            A_first_stage_temp[:, 0:ny] = -E
            A_first_stage_temp[:, nx_temp:] = -G
            b_first_stage_temp = M.dot(u) - h
            A_first_stage = concatenate([A_first_stage, A_first_stage_temp])
            b_first_stage = concatenate([b_first_stage, b_first_stage_temp])

            A_first_stage_temp = zeros((1, nx_temp + nx))
            b_first_stage_temp = zeros((1, 1))
            A_first_stage_temp[0, nx_temp:] = d.transpose()
            A_first_stage_temp[0, ny] = -1
            A_first_stage = concatenate([A_first_stage, A_first_stage_temp])
            b_first_stage = concatenate([b_first_stage, b_first_stage_temp])

            k += 1
            if k > kmax:
                break

        return y


if __name__ == "__main__":
    robust_optimization = TwoStageRobustOptimization()
