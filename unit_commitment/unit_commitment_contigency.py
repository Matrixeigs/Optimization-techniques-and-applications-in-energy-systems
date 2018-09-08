"""
Unit Commitment Problem Considering the Failures of Transmission Lines

@date: 13 June 2018
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array, eye
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
from scipy.sparse import csr_matrix as sparse

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp

from unit_commitment.data_format.data_format_contigency import ALPHA, BETA, IG, PG, RD, RU, THETA, PL, NG


class UnitCommitmentContingency():
    """"""

    def __init__(self):
        self.name = "Unit commitment considering the contingencies of transmission lines"

    def problem_formulation(self, case, delta=0.03):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """

        MIN_UP = -2
        MIN_DOWN = -3

        # Modify the bus, gen and branch matrix
        bus[:, BUS_I] = bus[:, BUS_I] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1

        ng = shape(case['gen'])[0]  # number of schedule injections
        nl = shape(case['branch'])[0]  ## number of branches
        nb = shape(case['bus'])[0]  ## number of branches
        self.ng = ng
        self.nb = nb
        self.nl = nl

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices
        ## connection matrix
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                    (nb, ng))

        u0 = [0] * ng  # The initial generation status
        for i in range(ng):
            u0[i] = int(gencost[i, -1] > 0)
        # Formulate a mixed integer quadratic programming problem
        # 1) Announce the variables
        # [vt,wt,ut,Pt]:start-up,shut-down,status,generation level, up-reserve, down-reserve
        # 1.1) boundary information
        T = case["Load_profile"].shape[0]
        nx = NG * T * ng + nb * T + nl * T
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx
        self.T = T

        for i in range(T):
            for j in range(ng):
                # lower boundary
                lb[ALPHA * ng * T + i * ng + j] = 0
                lb[BETA * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RU * ng * T + i * ng + j] = 0
                lb[RD * ng * T + i * ng + j] = 0

                # upper boundary
                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RU * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RD * ng * T + i * ng + j] = gen[j, PMAX]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "D"

        for i in range(T):
            for j in range(nb):
                lb[THETA * ng * T + i * nb + j] = -360
                ub[THETA * ng * T + i * nb + j] = 360
                if bus[j, BUS_TYPE] == REF:
                    lb[THETA * ng * T + i * nb + j] = 0
                    ub[THETA * ng * T + i * nb + j] = 0

        for i in range(T):
            for j in range(nl):
                lb[THETA * ng * T + T * nb + i * nl + j] = -branch[j, RATE_A]
                ub[THETA * ng * T + T * nb + i * nl + j] = branch[j, RATE_A]

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[BETA * ng * T + i * ng + j] = 0
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]
                c[RU * ng * T + i * ng + j] = 0
                c[RD * ng * T + i * ng + j] = 0

                q[PG * ng * T + i * ng + j] = gencost[j, 4]

        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = zeros((T * nb, nx))
        beq = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            Aeq[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()

            Aeq[i * nb:(i + 1) * nb,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -(Cft.transpose()).todense()

            beq[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]

        # 2.2) Status transformation of each unit
        Aeq_temp = zeros((T * ng, nx))
        beq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aeq_temp[i * ng + j, ALPHA * ng * T + i * ng + j] = -1
                Aeq_temp[i * ng + j, BETA * ng * T + i * ng + j] = 1
                Aeq_temp[i * ng + j, IG * ng * T + i * ng + j] = 1
                if i != 0:
                    Aeq_temp[i * ng + j, IG * ng * T + (i - 1) * ng + j] = -1
                else:
                    beq_temp[i * T + j] = 0

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.3) Transmission line flows
        Aeq_temp = zeros((T * nl, nx))
        beq_temp = zeros((T * nl, 1))
        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            Aeq_temp[i * nl:(i + 1) * nl,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -eye(nl)
            Aeq_temp[i * nl:(i + 1) * nl, THETA * ng * T + i * nb:THETA * ng * T + (i + 1) * nb] = X.dot(Cft.todense())

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)
        # 2.4) Power range limitation
        Aineq = zeros((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aineq[i * ng + j, BETA * ng * T + i * ng + j] = 1
                bineq[i * ng + j] = 1

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PMIN]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PMAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
            DOWN_LIMIT[i] = T - int(gencost[i, MIN_DOWN])
        # 2.5.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Down limit
        Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_DOWN]), T):
                for k in range(j - int(gencost[i, MIN_DOWN]), j):
                    Aineq_temp[
                        sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), BETA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Ramp constraints:
        # 2.6.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.7)  Up and down reserve for the forecasting errors
        # Up reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RU * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # Down reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RD * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
                 "q": q,
                 "lb": lb,
                 "ub": ub,
                 "A": Aineq,
                 "b": bineq,
                 "Aeq": Aeq,
                 "beq": beq,
                 "vtypes": vtypes}
        return model

    def problem_solving(self, model):
        """

        :param model: Formulated mathematical models
        :return:
        """
        (xx, obj, success) = miqp(model["c"], model["q"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx, obj

    def result_check(self, sol):
        """

        :param sol: The solution of mathematical
        :return:
        """
        T = self.T
        ng = self.ng
        nl = self.nl
        nb = self.nb
        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        rug = zeros((ng, T))
        rdg = zeros((ng, T))
        theta = zeros((nb, T))
        pf = zeros((nl, T))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                rug[j, i] = sol[RU * ng * T + i * ng + j]
                rdg[j, i] = sol[RD * ng * T + i * ng + j]

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[THETA * ng * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[THETA * ng * T + T * nb + i * nl + j]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RUG": rug,
                    "RDG": rdg, }

        return solution


if __name__ == "__main__":
    # Import the test cases
    from unit_commitment.test_cases.case6 import case6

    unit_commitment_contingency = UnitCommitmentContingency()
    profile = array(
        [1.75, 1.65, 1.58, 1.54, 1.55, 1.60, 1.73, 1.77, 1.86, 2.07, 2.29, 2.36, 2.42, 2.44, 2.49, 2.56, 2.56, 2.47,
         2.46, 2.37, 2.37, 2.33, 1.96, 1.96])
    case_base = case6()
    case_base["Load_profile"] = profile

    model = unit_commitment_contingency.problem_formulation(case_base)
    (sol, obj) = unit_commitment_contingency.problem_solving(model)
    sol = unit_commitment_contingency.result_check(sol)
