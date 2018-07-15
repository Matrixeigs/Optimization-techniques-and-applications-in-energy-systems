"""
Two Stage Robust Optimization for Unit Commitment Problem
The test case is the IEEE-6ww system.

@date: 13 June 2018
@author: Tianyang Zhao
@e-mail: zhaoty@ntu.edu.sg
"""

from numpy import zeros, shape, ones, diag, concatenate, r_, arange, array
import matplotlib.pyplot as plt
from solvers.mixed_integer_quadratic_programming import mixed_integer_quadratic_programming as miqp
import scipy.linalg as linalg
from scipy.sparse import csr_matrix as sparse

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN, RAMP_AGC
from pypower.idx_cost import STARTUP

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as lp


class TwoStageUnitCommitmentRobust():
    """"""

    def __init__(self):
        self.name = "Two stage robust optimization"

    def problem_formulation(self, case, delta=0.03):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]

        ON = 0
        OFF = 1
        IG = 2
        PG = 3
        RUG = 4
        RDG = 5
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

        u0 = [0] * ng  # The initial generation status
        for i in range(ng):
            u0[i] = int(gencost[i, -1] > 0)
        # Formulate a mixed integer quadratic programming problem
        # 1) Announce the variables
        # [vt,wt,ut,Pt]:start-up,shut-down,status,generation level, up-reserve, down-reserve
        # 1.1) boundary information
        T = case["Load_profile"].shape[0]
        nx = (RDG + 1) * T * ng
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))
        vtypes = ["c"] * nx

        for i in range(T):
            for j in range(ng):
                # lower boundary
                lb[ON * ng * T + i * ng + j] = 0
                lb[OFF * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RUG * ng * T + i * ng + j] = 0
                lb[RDG * ng * T + i * ng + j] = 0
                # upper boundary
                ub[ON * ng * T + i * ng + j] = 1
                ub[OFF * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RUG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RDG * ng * T + i * ng + j] = gen[j, PMAX]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "D"

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ON * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[OFF * ng * T + i * ng + j] = 0
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]
                c[RUG * ng * T + i * ng + j] = 0
                c[RDG * ng * T + i * ng + j] = 0

                q[PG * ng * T + i * ng + j] = gencost[j, 4]

        # 2) Constraint set
        # 2.1) Power balance equation
        Aeq = zeros((T, nx))
        beq = zeros((T, 1))
        for i in range(T):
            # For the hydro units
            for j in range(ng):
                Aeq[i, PG * ng * T + i * ng + j] = 1

            beq[i] = profile[i] * sum(bus[:, PD])

        # # 2.2) Status transformation of each unit
        Aeq_temp = zeros((T * ng, nx))
        beq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aeq_temp[i * ng + j, ON * ng * T + i * ng + j] = -1
                Aeq_temp[i * ng + j, OFF * ng * T + i * ng + j] = 1
                Aeq_temp[i * ng + j, IG * ng * T + i * ng + j] = 1
                if i != 0:
                    Aeq_temp[i * ng + j, IG * ng * T + (i - 1) * ng + j] = -1
                else:
                    beq_temp[i * T + j] = 0

        Aeq = concatenate((Aeq, Aeq_temp), axis=0)
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.3) Power range limitation
        Aineq = zeros((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq[i * ng + j, ON * ng * T + i * ng + j] = 1
                Aineq[i * ng + j, OFF * ng * T + i * ng + j] = 1
                bineq[i * ng + j] = 1

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PMIN]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_temp[i * ng + j, RDG * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PMAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RUG * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.4) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
            DOWN_LIMIT[i] = T - int(gencost[i, MIN_DOWN])
        # 2.4.1) Up limit
        Aineq_temp = zeros((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_UP]), T):
                for k in range(j - int(gencost[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ON * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.4.2) Down limit
        Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gencost[i, MIN_DOWN]), T):
                for k in range(j - int(gencost[i, MIN_DOWN]), j):
                    Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), OFF * ng * T + k * ng + i] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Ramp constraints:
        # 2.5.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, ON * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, OFF * ng * T + (j + 1) * ng + i] = gen[i, RAMP_AGC] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_AGC]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.6) Line flow limitation
        b = 1 / branch[:, BR_X]  ## series susceptance

        ## build connection matrix Cft = Cf - Ct for line and from - to buses
        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices
        ## connection matrix
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))

        ## build Bf such that Bf * Va is the vector of real branch powers injected
        ## at each branch's "from" bus
        Bf = sparse((r_[b, -b], (i, r_[f, t])), shape=(nl, nb))  ## = spdiags(b, 0, nl, nl) * Cft

        ## build Bbus
        Bbus = Cft.T * Bf
        # The distribution factor
        Distribution_factor = sparse(linalg.solve(Bbus.toarray().transpose(), Bf.toarray().transpose()).transpose())
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                    (nb, ng))  # Sparse index generation method is different from the way of matlab

        Aineq_temp = zeros((T * nl, nx))
        bineq_temp = zeros((T * nl, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = -(
                    Distribution_factor * Cg).todense()
            bineq_temp[i * nl:(i + 1) * nl, :] = (
                    branch[:, RATE_A] - Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = zeros((T * nl, nx))
        bineq_temp = zeros((T * nl, 1))
        for i in range(T):
            Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = (
                    Distribution_factor * Cg).todense()
            bineq_temp[i * nl:(i + 1) * nl, :] = (
                    branch[:, RATE_A] + Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.7)  Up and down reserve for the forecasting errors
        # Up reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RUG * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # Down reserve limitation
        Aineq_temp = zeros((T, nx))
        bineq_temp = zeros((T, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i, RDG * ng * T + i * ng + j] = -1
            bineq_temp[i] -= delta * sum(bus[:, PD])
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
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
        :return:
        """
        (xx, obj, success) = lp(model["c"], Aeq=model["Aeq"], beq=model["beq"],
                                A=model["A"],
                                b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx

    def result_check(self):
        """

        :return:
        """


if __name__ == "__main__":
    # Import the test cases
    from unit_commitment.test_cases.case6 import case6

    two_stage_unit_commitment_robust = TwoStageUnitCommitmentRobust()
    profile = array(
        [0.64, 0.60, 0.58, 0.56, 0.56, 0.58, 0.64, 0.76, 0.87, 0.95, 0.99, 1.00, 0.99, 1.00, 1.00, 0.97, 0.96, 0.96,
         0.93, 0.92, 0.92, 0.93, 0.87, 0.72])
    case_base = case6()
    case_base["Load_profile"] = profile

    model = two_stage_unit_commitment_robust.problem_formulation(case_base)
    sol = two_stage_unit_commitment_robust.problem_solving(model)
