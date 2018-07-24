"""
Distributed unit commitment for micro-gird power park
"""

from numpy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack
from numpy import flatnonzero as find
from gurobipy import *

from distribution_system_optimization.test_cases import case33

from pypower.idx_brch import F_BUS, T_BUS, BR_X, BR_STATUS, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PMAX, PMIN
from pypower.ext2int import ext2int


class UnitCommitmentPowerPark():
    """
    Unit commitment for power park
    """

    def __init__(self):
        self.name = "Power park unit commitment"

    def problem_formulation(self, case, micro_grids, profile):
        """

        :param cases: Distribution network models
        :param micro_grids: Micro-grid models
        :param profile: Load-profile within the DC networks
        :return: Formulated centralized optimization problem
        """
        T = profile.length()
        self.T = T
        # Formulate the DC network reconfiguration
        case["branch"][:, BR_STATUS] = ones(case["branch"].shape[0])
        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nmg = len(micro_grids)
        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        m = zeros(nmg)  ## list of integration index
        Pmg_l = zeros(nmg)  ## list of lower boundary
        Pmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = micro_grids[i]["BUS"]
            Pmg_l[i] = micro_grids[i]["PMG"]["lb"]
            Pmg_u[i] = micro_grids[i]["PMG"]["ub"]

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))

        Branch_R = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A] / baseMVA

        Pij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = bus[:, VMIN] ** 2
        Pg_l = gen[:, PMIN] / baseMVA
        Alpha_l = zeros(nl)
        Beta_f_l = zeros(nl)
        Beta_t_l = zeros(nl)

        Pij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Alpha_u = ones(nl)
        Beta_f_u = ones(nl)
        Beta_t_u = ones(nl)
        bigM = max(Vm_u)
        # For the spanning tree constraints
        Root_node = find(bus[:, BUS_TYPE] == REF)
        Root_line = find(branch[:, F_BUS] == Root_node)

        Span_f = zeros((nb, nl))
        Span_t = zeros((nb, nl))
        for i in range(nb):
            Span_f[i, find(branch[:, F_BUS] == i)] = 1
            Span_t[i, find(branch[:, T_BUS] == i)] = 1

        Alpha_l[Root_line] = 1
        Alpha_u[Root_line] = 1
        Beta_f_l[Root_line] = 0
        Beta_f_l[Root_line] = 0
        PIJ = 0
        IIJ = 1
        VM = 2

        nx = (2 * nl + nb + ng + nmg) * T + 3 * nl  ## Dimension of decision variables
        NX = 2 * nl + nb + ng + nmg
        # 1) Lower, upper and types of variables
        lx = zeros(nx)
        ux = zeros(nx)
        vtypes = ["c"] * nx
        c = zeros(nx)
        q = zeros(nx)

        lx[0:nl] = Alpha_l
        ux[0:nl] = Alpha_u
        lx[nl:2 * nl] = Beta_f_l
        ux[nl:2 * nl] = Beta_f_u
        vtypes[nl:2 * nl] = ["b"] * nl

        lx[2 * nl:3 * nl] = Beta_t_l
        ux[2 * nl:3 * nl] = Beta_t_u
        vtypes[2 * nl:3 * nl] = ["b"] * nl

        for i in range(T):
            # Upper boundary
            lx[3 * nl + PIJ * nl:3 * nl + IIJ * nl] = Pij_l
            lx[3 * nl + IIJ * nl:3 * nl + VM * nl] = Iij_l
            lx[3 * nl + VM * nl:3 * nl + VM * nl + nb] = Vm_l
            lx[3 * nl + VM * nl + nb:3 * nl + VM * nl + nb + ng] = Pg_l
            lx[3 * nl + VM * nl + nb + ng:3 * nl + VM * nl + nb + ng + nmg] = Pmg_l

            # Lower boundary
            ux[3 * nl + PIJ * nl:3 * nl + IIJ * nl] = Pij_u
            ux[3 * nl + IIJ * nl:3 * nl + VM * nl] = Iij_u
            ux[3 * nl + VM * nl:3 * nl + VM * nl + nb] = Vm_u
            ux[3 * nl + VM * nl + nb:3 * nl + VM * nl + nb + ng] = Pg_u
            ux[3 * nl + VM * nl + nb + ng:3 * nl + VM * nl + nb + ng + nmg] = Pmg_u
            # Cost
            c[3 * nl + VM * nl + nb:3 * nl + VM * nl + nb + ng] = gencost[:, 5] * baseMVA
            q[3 * nl + VM * nl + nb:3 * nl + VM * nl + nb + ng] = gencost[:, 4] * baseMVA * baseMVA

        # Formulate equal constraints
        ## 2) Equal constraints
        # 2.1) Alpha = Beta_f + Beta_t
        Aeq_f = zeros((nl, nx))
        beq_f = zeros(nl)
        Aeq_f[:, 0: nl] = -eye(nl)
        Aeq_f[:, nl:2 * nl] = eye(nl)
        Aeq_f[:, 2 * nl: 3 * nl] = eye(nl)
        # 2.2) sum(alpha)=nb-1
        Aeq_alpha = zeros((1, nx))
        beq_alpha = zeros(1)
        Aeq_alpha[0, 0:  nl] = ones(nl)
        beq_alpha[0] = nb - 1
        # 2.3) Span_f*Beta_f+Span_t*Beta_t = Spanning_tree
        Aeq_span = zeros((nb, nx))
        beq_span = ones(nb)
        beq_span[Root_node] = 0
        Aeq_span[:, nl:2 * nl] = Span_f
        Aeq_span[:, 2 * nl:] = Span_t
        # 2.4) Power balance equation
        Aeq_p = hstack([Ct - Cf, -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, Cmg])
        beq_p = bus[:, PD] / baseMVA
        Aeq_power_balance = zeros((nb * T, nx))
        beq_power_balance = zeros(nb * T)

        for i in range(T):
            Aeq_power_balance[i * nb:(i + 1) * nb, 3 * nl + i * NX: 3 * nl + (i + 1) * NX] = Aeq_p
            beq_power_balance[i * nb:(i + 1) * nb] = beq_p * profile[i]

        Aeq = concatenate([Aeq_f, Aeq_alpha, Aeq_span, Aeq_power_balance])
        beq = concatenate([beq_f, beq_alpha, beq_span, beq_power_balance])

        ## 3) Inequality constraints
        # 3.1) Pij<=Iij*Pij_max
        A_pij = zeros((nl * T, nx))
        b_pij = zeros(nl * T)
        for i in range(T):
            A_pij[i * nl:(i + 1) * nl, 3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = eye(nl)
            A_pij[i * nl:(i + 1) * nl, 0: nl] = -diag(Pij_u)
        # 3.2) lij<=Iij*lij_max
        A_lij = zeros((nl * T, nx))
        b_lij = zeros(nl * T)
        for i in range(T):
            A_lij[i * nl:(i + 1) * nl, 3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = eye(nl)
            A_lij[i * nl:(i + 1) * nl, 0: nl] = -diag(Iij_u)
        # 3.3) KVL equation
        A_kvl = zeros((2 * nl * T, nx))
        b_kvl = zeros(2 * nl * T)
        for i in range(T):
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = -2 * diag(
                Branch_R)
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = diag(
                Branch_R ** 2)
            A_kvl[i * nl:(i + 1) * nl, 3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = (
                    Cf.T - Ct.T).toarray()
            A_kvl[i * nl:(i + 1) * nl, 0:nl] = eye(nl) * bigM
            b_kvl[i * nl:(i + 1) * nl, 0:nl] = ones(nl) * bigM

            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl,
            3 * nl + i * NX + PIJ * nl:3 * nl + i * NX + (PIJ + 1) * nl] = 2 * diag(Branch_R)
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl,
            3 * nl + i * NX + IIJ * nl:3 * nl + i * NX + (IIJ + 1) * nl] = -diag(Branch_R ** 2)
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl, 3 * nl + i * NX + VM * nl:3 * nl + i * NX + VM * nl + nb] = -(
                    Cf.T - Ct.T).toarray()
            A_kvl[nl * T + i * nl:nl * T + (i + 1) * nl, 0:nl] = eye(nl) * bigM
            b_kvl[nl * T + i * nl:nl * T + (i + 1) * nl, 0:nl] = ones(nl) * bigM

        A = concatenate([A_pij, A_lij, A_kvl])
        b = concatenate([b_pij, b_lij, b_kvl])

        ## For the microgrids

        model = Model("Network_reconfiguration")
        # Define the decision variables
        x = {}
        nx = lx.shape[0]

        for i in range(nx):
            if vtypes[i] == "c":
                x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
            elif vtypes[i] == "b":
                x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.BINARY)

        return model

    def micro_grid(self, micro_grid):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """
        from unit_commitment.distributed_unit_commitment.idx_unit_commitment import ICS, IG, IUG, IBIC_AC2DC, \
            PBIC_AC2DC, PG, PESS_DC, PMG, PBIC_DC2AC, PUG, PESS_CH, RUG, RESS, RG, NX
        T = self.T
        nx = NX * T
        ## 1)


if __name__ == "__main__":
    from pypower import runopf

    mpc = case33.case33()  # Default test case
    unit_commitment_power_park = UnitCommitmentPowerPark()

    sol = unit_commitment_power_park.problem_formulation(case=mpc)
