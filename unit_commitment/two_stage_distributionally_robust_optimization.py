"""
Two-stage unit commitment using distributionally robust approach
@author: Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
@date: 14 June 2019
"""

from numpy import zeros, shape, ones, concatenate, r_, arange, array, eye, tile
import scipy.linalg as linalg
from scipy.sparse import csc_matrix as sparse
from scipy.sparse import lil_matrix, vstack, hstack
from scipy import inf

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN, RAMP_AGC
from pypower.idx_cost import STARTUP

from solvers.mixed_integer_quadratic_solver_cplex_sparse import mixed_integer_quadratic_programming as miqp

from unit_commitment.data_format.data_format_jointed_energy_reserve import ALPHA, BETA, IG, PG, RDG, RUG


class TwoStageUnitCommitmentRobust():
    """"""

    def __init__(self):
        self.name = "Two stage robust optimization"

    def problem_formulation(self, case, beta=0.03):
        """
        Input check for the unit commitment problem
        :param cases:
        :return:
        """
        baseMVA, bus, gen, branch, gencost, profile = case["baseMVA"], case["bus"], case["gen"], case["branch"], case[
            "gencost"], case["Load_profile"]

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
        self.T = T

        for i in range(T):
            for j in range(ng):
                # lower boundary
                lb[ALPHA * ng * T + i * ng + j] = 0
                lb[BETA * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RUG * ng * T + i * ng + j] = 0
                lb[RDG * ng * T + i * ng + j] = 0
                # upper boundary
                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RUG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RDG * ng * T + i * ng + j] = gen[j, PMAX]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "D"
                vtypes[ALPHA * ng * T + i * ng + j] = "D"
                vtypes[BETA * ng * T + i * ng + j] = "D"

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[BETA * ng * T + i * ng + j] = 0
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
                Aeq_temp[i * ng + j, ALPHA * ng * T + i * ng + j] = -1
                Aeq_temp[i * ng + j, BETA * ng * T + i * ng + j] = 1
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
                Aineq[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aineq[i * ng + j, BETA * ng * T + i * ng + j] = 1
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
        # UP_LIMIT = [0] * ng
        # DOWN_LIMIT = [0] * ng
        # for i in range(ng):
        #     UP_LIMIT[i] = T - int(gencost[i, MIN_UP])
        #     DOWN_LIMIT[i] = T - int(gencost[i, MIN_DOWN])
        # # 2.4.1) Up limit
        # Aineq_temp = zeros((sum(UP_LIMIT), nx))
        # bineq_temp = zeros((sum(UP_LIMIT), 1))
        # for i in range(ng):
        #     for j in range(int(gencost[i, MIN_UP]), T):
        #         for k in range(j - int(gencost[i, MIN_UP]), j):
        #             Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
        #         Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)
        # # 2.4.2) Down limit
        # Aineq_temp = zeros((sum(DOWN_LIMIT), nx))
        # bineq_temp = ones((sum(DOWN_LIMIT), 1))
        # for i in range(ng):
        #     for j in range(int(gencost[i, MIN_DOWN]), T):
        #         for k in range(j - int(gencost[i, MIN_DOWN]), j):
        #             Aineq_temp[
        #                 sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), BETA * ng * T + k * ng + i] = 1
        #         Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gencost[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Ramp constraints:
        # 2.5.1) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = gen[i, PMAX] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, PMAX]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = gen[i, PMAX] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, PMAX]
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

        # Aineq_temp = zeros((T * nl, nx))
        # bineq_temp = zeros((T * nl, 1))
        # for i in range(T):
        #     Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = -(
        #             Distribution_factor * Cg).todense()
        #     bineq_temp[i * nl:(i + 1) * nl, :] = (
        #             branch[:, RATE_A] - Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)
        #
        self.Distribution_factor = Distribution_factor
        self.Pd = bus[:, PD]
        self.profile = profile
        self.Cg = Cg
        # Aineq_temp = zeros((T * nl, nx))
        # bineq_temp = zeros((T * nl, 1))
        # for i in range(T):
        #     Aineq_temp[i * nl:(i + 1) * nl, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = (
        #             Distribution_factor * Cg).todense()
        #     bineq_temp[i * nl:(i + 1) * nl, :] = (
        #             branch[:, RATE_A] + Distribution_factor * bus[:, PD] * profile[i]).reshape((nl, 1))
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.7)  Up and down reserve for the forecasting errors
        # Up reserve limitation
        # Aineq_temp = zeros((T, nx))
        # bineq_temp = zeros((T, 1))
        # for i in range(T):
        #     for j in range(ng):
        #         Aineq_temp[i, RUG * ng * T + i * ng + j] = -1
        #     bineq_temp[i] -= beta * sum(bus[:, PD])
        #
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)
        # # Down reserve limitation
        # Aineq_temp = zeros((T, nx))
        # bineq_temp = zeros((T, 1))
        # for i in range(T):
        #     for j in range(ng):
        #         Aineq_temp[i, RDG * ng * T + i * ng + j] = -1
        #     bineq_temp[i] -= beta * sum(bus[:, PD])
        # Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        # bineq = concatenate((bineq, bineq_temp), axis=0)

        model = {"c": c,
                 "q": q,
                 "lb": lb,
                 "ub": ub,
                 "A": sparse(Aineq).tocoo(),
                 "b": bineq,
                 "Aeq": sparse(Aeq).tocoo(),
                 "beq": beq,
                 "vtypes": vtypes}
        self.problem_solving(model)
        # The second stage model
        nx = len(model["lb"])
        ny = int(ng * T + nb * T)
        nz = nb * T
        nu = T
        pd_mean = zeros((nb * T, 1))
        for i in range(T): pd_mean[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]
        # [pg,pd_shed]
        # 1) Pg+Ru-pg>=0
        # Tx+Wy>=h+Hz
        Tx = zeros((ng * T, nx))
        Wy = zeros((ng * T, ny))
        h0 = zeros((ng * T, 1))
        Hz = zeros((ng * T, nz))
        for i in range(T):
            for j in range(ng):
                Tx[i * ng + j, PG * ng * T + i * ng + j] = 1
                Tx[i * ng + j, RUG * ng * T + i * ng + j] = 1
                Wy[i * ng + j, i * ng + j] = -1
        # 2) -Pg+Ru+pg>=0
        Tx_temp = zeros((ng * T, nx))
        Wy_temp = zeros((ng * T, ny))
        h0_temp = zeros((ng * T, 1))
        Hz_temp = zeros((ng * T, nz))
        for i in range(T):
            for j in range(ng):
                Tx_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Tx_temp[i * ng + j, RDG * ng * T + i * ng + j] = 1
                Wy_temp[i * ng + j, i * ng + j] = 1
        Tx = concatenate([Tx, Tx_temp])
        Wy = concatenate([Wy, Wy_temp])
        h0 = concatenate([h0, h0_temp])
        Hz = concatenate([Hz, Hz_temp])
        # 3) -pd_shed>=-pd
        Tx_temp = zeros((nb * T, nx))
        Wy_temp = zeros((nb * T, ny))
        h0_temp = zeros((nb * T, 1))
        Hz_temp = zeros((nb * T, nz))
        for i in range(T):
            for j in range(nb):
                Wy_temp[i * nb + j, ng * T + i * nb + j] = -1
                Hz_temp[i * nb + j, i * nb + j] = -1
        Tx = concatenate([Tx, Tx_temp])
        Wy = concatenate([Wy, Wy_temp])
        h0 = concatenate([h0, h0_temp])
        Hz = concatenate([Hz, Hz_temp])
        # 4) pd_shed>=0
        Tx_temp = zeros((nb * T, nx))
        Wy_temp = zeros((nb * T, ny))
        h0_temp = zeros((nb * T, 1))
        Hz_temp = zeros((nb * T, nz))
        for i in range(T):
            for j in range(nb):
                Wy_temp[i * nb + j, ng * T + i * nb + j] = 1
        Tx = concatenate([Tx, Tx_temp])
        Wy = concatenate([Wy, Wy_temp])
        h0 = concatenate([h0, h0_temp])
        Hz = concatenate([Hz, Hz_temp])
        # 5) Transmission lines flow limitaions
        # Tx_temp = zeros((nl * T, nx))
        # Wy_temp = zeros((nl * T, ny))
        # h0_temp = zeros((nl * T, 1))
        # Hz_temp = zeros((nl * T, nz))
        # for i in range(T):
        #     for j in range(nb):
        #         Wy_temp[i * nl:(i + 1) * nl, i * ng:(i + 1) * ng] = -(Distribution_factor * Cg).todense()
        #         Wy_temp[i * nl:(i + 1) * nl, ng * T + i * nb:ng * T + (i + 1) * nb] = -Distribution_factor.todense()
        #         h0_temp[i * nl:(i + 1) * nl, 0] = -branch[:, RATE_A]
        #         Hz_temp[i * nl:(i + 1) * nl, i * nb:(i + 1) * nb] = -Distribution_factor.todense()
        # Tx = concatenate([Tx, Tx_temp])
        # Wy = concatenate([Wy, Wy_temp])
        # h0 = concatenate([h0, h0_temp])
        # Hz = concatenate([Hz, Hz_temp])
        #
        # Tx_temp = zeros((nl * T, nx))
        # Wy_temp = zeros((nl * T, ny))
        # h0_temp = zeros((nl * T, 1))
        # Hz_temp = zeros((nl * T, nz))
        # for i in range(T):
        #     for j in range(nb):
        #         Wy_temp[i * nl:(i + 1) * nl, i * ng:(i + 1) * ng] = (Distribution_factor * Cg).todense()
        #         Wy_temp[i * nl:(i + 1) * nl, ng * T + i * nb:ng * T + (i + 1) * nb] = Distribution_factor.todense()
        #         h0_temp[i * nl:(i + 1) * nl, 0] = -branch[:, RATE_A]
        #         Hz_temp[i * nl:(i + 1) * nl, i * nb:(i + 1) * nb] = Distribution_factor.todense()
        # Tx = concatenate([Tx, Tx_temp])
        # Wy = concatenate([Wy, Wy_temp])
        # h0 = concatenate([h0, h0_temp])
        # Hz = concatenate([Hz, Hz_temp])

        # 6) Power balance equations
        # sum(pg)+sum(pd_shed)==sum(pd)
        Tx_temp = zeros((T, nx))
        Wy_temp = zeros((T, ny))
        h0_temp = zeros((T, 1))
        Hz_temp = zeros((T, nz))
        for i in range(T):
            Wy_temp[i, i * ng:(i + 1) * ng] = 1
            Wy_temp[i, ng * T + i * nb:ng * T + (i + 1) * nb] = 1
            Hz_temp[i, i * nb:(i + 1) * nb] = 1
        Tx = concatenate([Tx, Tx_temp, -Tx_temp])
        Wy = concatenate([Wy, Wy_temp, -Wy_temp])
        h0 = concatenate([h0, h0_temp, -h0_temp])
        Hz = concatenate([Hz, Hz_temp, -Hz_temp])

        # Construct the uncertainty set
        # 1) expect(Gz)==mu
        G = eye(nb * T)
        mu = pd_mean
        mu = mu.reshape((len(mu), 1))
        # 2) u<delta, the corelation among buses at each time period
        delta = profile * sum(bus[:, PD])
        delta = delta.reshape((len(delta), 1))
        # 3) The correlation among u and v
        # 3.1) pd <= (1+beta)*pd_mean
        C = eye(nb * T)
        D = zeros((nb * T, T))
        h = (1 + beta) * pd_mean
        # 3.2) -pd <= -(1-beta)*pd_mean
        C_temp = -eye(nb * T)
        D_temp = zeros((nb * T, T))
        h_temp = -(1 - beta) * pd_mean
        C = concatenate([C, C_temp])
        D = concatenate([D, D_temp])
        h = concatenate([h, h_temp])
        # 3.3) sum(pd) == u
        C_temp = zeros((T, nb * T))
        D_temp = -eye(T)
        h_temp = zeros((T, 1))
        for i in range(T): C_temp[i, i * nb:(i + 1) * nb] = 1
        C = concatenate([C, C_temp, -C_temp])
        D = concatenate([D, D_temp, -D_temp])
        h = concatenate([h, h_temp, -h_temp])
        # 4) objective value of the second stage
        Voll = 1e4
        d = concatenate([tile(gencost[:, 5], T), Voll * ones(nb * T)])
        d = d.reshape((len(d), 1))
        ## Formulate extended second stage problems
        # 1) add r, s and t, respectively
        ns = len(mu)
        nt = len(delta)
        npi = C.shape[0]
        lb = concatenate([-inf * ones((ny + ny * nz + ny * nu + 1 + ns, 1)), zeros((nt + npi, 1))])
        ub = inf * ones((ny + ny * nz + ny * nu + 1 + ns + nt + npi, 1))
        vtypes = ['c'] * (ny + ny * nz + ny * nu + 1 + ns + nt + npi)
        obj = concatenate([zeros((ny + ny * nz + ny * nu, 1)), ones((1, 1)), mu, delta, zeros((npi, 1))])
        # add first five constaints
        # H stands for inequality, m stands for the rhs; Hx<=m
        H = concatenate([d, zeros((ny * nz + ny * nu, 1)), -ones((1, 1)), zeros((ns + nt, 1)), h]).transpose()
        m = zeros((1, 1))

        Heq = lil_matrix((nz, ny + ny * nz + ny * nu + 1 + ns + nt + npi))
        meq = zeros((nz, 1))
        for i in range(nz):
            Heq[i, ny + i * ny:ny + (i + 1) * ny] = -d.transpose()
            Heq[i, ny + ny * nz + ny * nu + 1: ny + ny * nz + ny * nu + 1 + ns] = G[:, i].transpose()
            Heq[i, ny + ny * nz + ny * nu + 1 + ns + nt: ny + ny * nz + ny * nu + 1 + ns + nt + npi] = C[:,
                                                                                                       i].transpose()

        Heq_temp = lil_matrix((nu, ny + ny * nz + ny * nu + 1 + ns + nt + npi))
        meq_temp = zeros((nu, 1))
        for i in range(nu):
            Heq_temp[i, ny + ny * nz + i * ny:ny + ny * nz + (i + 1) * ny] = -d.transpose()
            Heq_temp[i, ny + ny * nz + ny * nu + ns + i] = 1
            Heq_temp[i, ny + ny * nz + ny * nu + 1 + ns + nt:ny + ny * nz + ny * nu + 1 + ns + nt + npi] = D[:,
                                                                                                           i].transpose()
        Heq = vstack([Heq, Heq_temp])
        meq = concatenate([meq, meq_temp])

        N = lil_matrix((H.shape[0], nx))
        Neq = lil_matrix((Heq.shape[0], nx))

        M = Tx.shape[0]
        H = hstack([H, lil_matrix(zeros((H.shape[0], npi * M)))])
        Heq = hstack([Heq, lil_matrix(zeros((Heq.shape[0], npi * M)))])
        lb = concatenate([lb, zeros((npi * M, 1))])
        ub = concatenate([ub, inf * ones((npi * M, 1))])
        vtypes += ['c'] * npi * M
        obj = concatenate([obj, zeros((npi * M, 1))])

        H_temp = lil_matrix((M, ny + ny * nz + ny * nu + 1 + ns + nt + npi + M * npi))
        m_temp = zeros((M, 1))
        for l in range(M):
            # For the novel added variables
            H_temp[l,ny + ny * nz + ny * nu + 1 + ns + nt + npi + l * npi: ny + ny * nz + ny * nu + 1 + ns + nt + npi + (
                        l + 1) * npi] = h.transpose()
            H_temp[l, 0:ny] = -Wy[l, :]
            m_temp[l, 0] = -h0[l]

        H = vstack([H, H_temp])
        m = concatenate([m, m_temp])
        N = vstack([N, -lil_matrix(Tx)])
        # For each uncertainty variable
        Heq_temp = lil_matrix((M * nz, ny + ny * nz + ny * nu + 1 + ns + nt + npi + M * npi))
        meq_temp = zeros((M * nz, 1))
        for l in range(M):
            for i in range(nz):
                Heq_temp[l * nz + i, ny + ny * i:ny + ny * (i + 1)] = Wy[l, :]
                Heq_temp[l * nz + i,
                ny + ny * nz + ny * nu + 1 + ns + nt + npi + l * npi:ny + ny * nz + ny * nu + 1 + ns + nt + npi + (
                            l + 1) * npi] = C[:, i].transpose()
                meq_temp[l * nz + i, 0] = Hz[l, i]
            # print(l / M)
        Heq = vstack([Heq, Heq_temp])
        meq = concatenate([meq, meq_temp])
        Neq = vstack([Neq, lil_matrix(zeros((M * nz, nx)))])

        Heq_temp = lil_matrix((M * nu, ny + ny * nz + ny * nu + 1 + ns + nt + npi + M * npi))
        meq_temp = zeros((M * nu, 1))
        for l in range(M):
            for i in range(nu):
                Heq_temp[l * nu + i, ny + ny * nz + i * ny:ny + ny * nz + (i + 1) * ny] = Wy[l, :]
                Heq_temp[l * nu + i,
                ny + ny * nz + ny * nu + 1 + ns + nt + npi + l * npi:ny + ny * nz + ny * nu + 1 + ns + nt + npi + (
                            l + 1) * npi] = D[:, i].transpose()
            # print(l / M)
        Heq = vstack([Heq, Heq_temp])
        meq = concatenate([meq, meq_temp])
        Neq = vstack([Neq, lil_matrix(zeros((M * nu, nx)))])

        model["A"] = hstack([model["A"], lil_matrix((model["A"].shape[0], len(lb)))])
        model["A"] = vstack([model["A"], hstack([N, H])])
        model["b"] = concatenate([model["b"], m])

        model["Aeq"] = hstack([model["Aeq"], lil_matrix((model["Aeq"].shape[0], len(lb)))])
        model["Aeq"] = vstack([model["Aeq"], hstack([Neq, Heq])])
        model["beq"] = concatenate([model["beq"], meq])

        model["c"] = concatenate([model["c"], obj])
        model["lb"] = concatenate([model["lb"], lb])
        model["ub"] = concatenate([model["ub"], ub])

        model["vtypes"] += vtypes

        return model

    def problem_solving(self, model):
        """
        :return:
        """
        (xx, obj, success) = miqp(model["c"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx

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

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                rug[j, i] = sol[RUG * ng * T + i * ng + j]
                rdg[j, i] = sol[RDG * ng * T + i * ng + j]
        pf = zeros((nl, T))
        Distribution_factor = self.Distribution_factor
        Pd = self.Pd
        profile = self.profile
        Cg = self.Cg
        for i in range(T):
            pf[:, i] = Distribution_factor * (Cg * pg[:, i] - Pd * profile[i])

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

    two_stage_unit_commitment_robust = TwoStageUnitCommitmentRobust()
    profile = array([0.64, 0.60, 0.58, 0.56, 0.56, 0.58, 0.64, 0.76, 0.87, 0.95, 0.99, 1.00])
    case_base = case6()
    case_base["Load_profile"] = profile

    model = two_stage_unit_commitment_robust.problem_formulation(case_base)
    sol = two_stage_unit_commitment_robust.problem_solving(model)
    sol = two_stage_unit_commitment_robust.result_check(sol)
