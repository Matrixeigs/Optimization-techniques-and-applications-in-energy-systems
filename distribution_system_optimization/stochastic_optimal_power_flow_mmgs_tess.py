"""
Stochastic optimal power flow with multiple microgrids and mobile energy storage systems
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date: 4 Jan 2019
The matrix is stored using lil_matrix
"""

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid
from transportation_systems.test_cases import case3, TIME, LOCATION

from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, lil_matrix
from numpy import flatnonzero as find
from numpy import array, tile, arange, random

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, RATE_A
from pypower.idx_bus import PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp

from copy import deepcopy

from distribution_system_optimization.data_format.idx_MG import PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, \
    PMESS, EESS, NX_MG, QBIC, QUG, QG


class StochasticDynamicOptimalPowerFlowTess():
    def __init__(self):
        self.name = "Stochastic optimal power flow with tess"

    def main(self, case, micro_grids, profile, tess, traffic_networks):
        """
        Main entrance for network reconfiguration problems
        :param case: electric network information
        :param profile: load profile within the distribution networks
        :param micrgrids: dictionary for microgrids
        :param tess: dictionary for tess
        :return: network reconfiguration, distribution network status, and microgrid status
        """
        # Time spans
        T = len(profile)
        self.T = T
        # NUmber of microgrids
        nmg = len(micro_grids)
        self.nmg = nmg
        # Number of tess
        nev = len(tess)
        self.nev = nev
        # Number of buses in the transportation networks
        nb_traffic = traffic_networks["bus"].shape[0]
        self.nb_traffic = nb_traffic
        # Formulate the second stage scenarios
        (profile_second_second, micro_grids_second_stage) = self.scenario_generation(profile=profile,
                                                                                     microgrids=micro_grids)

        # 1) Formulate the first stage optimization problem
        model_first_stage = self.first_stage_problem_formualtion(power_networks=case, micro_grids=micro_grids,
                                                                 tess=tess, traffic_networks=traffic_networks)
        # sol_first_stage = milp(model_first_stage["c"], Aeq=model_first_stage["Aeq"], beq=model_first_stage["beq"],
        #                        A=model_first_stage["A"], b=model_first_stage["b"], vtypes=model_first_stage["vtypes"],
        #                        xmax=model_first_stage["ub"], xmin=model_first_stage["lb"])
        # 2) Formulate the second stage optimization problem
        model_second_stage = self.second_stage_problem_formualtion(power_networks=case,
                                                                   micro_grids=micro_grids_second_stage[0],
                                                                   tess=tess, traffic_networks=traffic_networks,
                                                                   profile=profile_second_second[0, :])
        # 3) Obtain the results for first-stage and second stage optimization problems

        # 4) Verify the first-stage and second stage optization problem

        # 1.1) Distribution networks

        # 4) Formulate the second stage problem, under different scenarios

        # return sol_distribution_network, sol_microgrids, sol_tess
        return model_first_stage

    def first_stage_problem_formualtion(self, power_networks, micro_grids, tess, traffic_networks):
        """
        Problem formulation for the first stage optimization
        :param power_networks: Parameters for the power networks
        :param micro_grids: Parameters for the microgrids
        :param tess: Parameters for the mobile energy storage systems
        :param traffic_networks: Parameters for the transportation networks
        :return:
        """
        T = self.T  # Time slots
        nmg = self.nmg  # Number of mgs
        nev = self.nev  # Number of tess
        # Decision variables include, DGs within power networks, DGs within MGs, EESs within MGs and TESSs
        mpc = ext2int(power_networks)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nb = shape(mpc["bus"])[0]
        self.nb = nb
        # Boundary for DGs within distribution networks
        Pg_l = gen[:, PMIN] / baseMVA
        Rg_l = gen[:, PMIN] / baseMVA
        Pg_u = gen[:, PMAX] / baseMVA
        Rg_u = gen[:, PMAX] / baseMVA
        cg = gencost[:, 5] * baseMVA
        cr = zeros(ng)

        # Boundary for DGs within MGs
        Pg_mg_l = zeros(nmg)
        Rg_mg_l = zeros(nmg)
        Pg_mg_u = zeros(nmg)
        Rg_mg_u = zeros(nmg)
        cg_mg = zeros(nmg)
        cr_mg = zeros(nmg)

        for i in range(nmg):
            Pg_mg_l[i] = micro_grids[i]["DG"]["PMIN"] / 1000 / baseMVA
            Pg_mg_u[i] = micro_grids[i]["DG"]["PMAX"] / 1000 / baseMVA
            Rg_mg_u[i] = micro_grids[i]["DG"]["PMAX"] / 1000 / baseMVA
            cg_mg[i] = micro_grids[i]["DG"]["COST_B"] * 1000 / baseMVA

        # Boundary for ESSs within MGs
        Pess_ch_l = zeros(nmg)
        Pess_dc_l = zeros(nmg)
        Eess_l = zeros(nmg)
        Ress_l = zeros(nmg)
        Iess_l = zeros(nmg)

        Pess_ch_u = zeros(nmg)
        Pess_dc_u = zeros(nmg)
        Eess_u = zeros(nmg)
        Ress_u = zeros(nmg)
        Iess_u = ones(nmg)

        cess_ch = zeros(nmg)
        cess_dc = zeros(nmg)
        cess_r = zeros(nmg)
        cess = zeros(nmg)
        cess_i = zeros(nmg)

        for i in range(nmg):
            Pess_ch_u[i] = micro_grids[i]["ESS"]["PCH_MAX"] / 1000 / baseMVA
            Pess_dc_u[i] = (micro_grids[i]["ESS"]["PDC_MAX"] + micro_grids[i]["ESS"]["PCH_MAX"]) / 1000 / baseMVA
            Ress_u[i] = micro_grids[i]["ESS"]["PCH_MAX"] / 1000 / baseMVA
            Eess_l[i] = micro_grids[i]["ESS"]["EMIN"] / 1000 / baseMVA
            Eess_u[i] = micro_grids[i]["ESS"]["EMAX"] / 1000 / baseMVA

        NX_first_stage = ng * 2 + nmg * 2 + nmg * 5
        nVariables_first_stage = NX_first_stage * T
        # Formulate the boundaries
        lx = concatenate(
            [tile(concatenate([Pg_l, Rg_l, Pg_mg_l, Rg_mg_l, Pess_ch_l, Pess_dc_l, Eess_l, Ress_l, Iess_l]), T)])
        ux = concatenate(
            [tile(concatenate([Pg_u, Rg_u, Pg_mg_u, Rg_mg_u, Pess_ch_u, Pess_dc_u, Eess_u, Ress_u, Iess_u]), T)])
        # Objective value
        c = concatenate([tile(concatenate([cg, cr, cg_mg, cr_mg, cess_ch, cess_dc, cess, cess_r, cess_i]), T)])
        # Variable types
        vtypes = (["c"] * (ng * 2 + nmg * 2 + nmg * 4) + ["b"] * nmg) * T

        ## Constraint sets
        # 1) Pg+Rg<=Pgu
        A = zeros((ng * T, nVariables_first_stage))
        b = zeros(ng * T)
        for i in range(T):
            for j in range(ng):
                A[i * ng + j, i * NX_first_stage + j] = 1
                A[i * ng + j, i * NX_first_stage + ng + j] = 1
                b[i * ng + j] = Pg_u[j]
        # 2) Pg-Rg>=Pgl
        A_temp = zeros((ng * T, nVariables_first_stage))
        b_temp = zeros(ng * T)
        for i in range(i):
            for j in range(ng):
                A_temp[i * ng + j, i * NX_first_stage + j] = -1
                A_temp[i * ng + j, i * NX_first_stage + ng + j] = 1
                b_temp[i * ng + j] = -Pg_l[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3) Pg_mg+Rg_mg<=Pg_mg_u
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + j] = 1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg + j] = 1
                b_temp[i * nmg + j] = Pg_mg_u[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 4) Pg_mg-Rg_mg<=Pg_mg_l
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + j] = -1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg + j] = 1
                b_temp[i * nmg + j] = Pg_mg_l[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 5) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + j] = -1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg + j] = 1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg * 2 + j] = 1
                b_temp[i * nmg + j] = Pess_dc_u[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 6) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, ng * 2 + nmg * 2 + i] = 1
                A_temp[i * nmg + j, ng * 2 + nmg * 2 + nmg + i] = -1
                A_temp[i * nmg + j, ng * 2 + nmg * 2 + nmg * 2 + i] = 1
                b_temp[i * nmg + j] = Pess_ch_u[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 7) Energy storage balance equation
        Aeq = zeros((T * nmg, nVariables_first_stage))
        beq = zeros(T * nmg)
        for i in range(T):
            for j in range(nmg):
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg * 3 + j] = 1
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + j] = -micro_grids[j]["ESS"]["EFF_CH"]
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg + j] = 1 / micro_grids[j]["ESS"]["EFF_DC"]
                if i == 0:
                    beq[i * nmg + j] = micro_grids[j]["ESS"]["E0"] / 1000 / baseMVA
                else:
                    Aeq[i * nmg + j, (i - 1) * NX_first_stage + ng * 2 + nmg * 2 + nmg * 3 + j] = -1
        # 8) Pess_ch<=I*Pess_ch_max
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + j] = 1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg * 4 + j] = -Pess_ch_u[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 9) Pess_dc<=(1-I)*Pess_dc_max
        A_temp = zeros((nmg * T, nVariables_first_stage))
        b_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg + j] = 1
                A_temp[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg * 4 + j] = Pess_dc_u[j]
                b_temp[i * nmg + j] = Pess_dc_u[j]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        # 2) Transportation energy storage systems problem
        model_tess = {}
        for i in range(nev):
            model_tess[i] = self.problem_formulation_tess(tess=tess[i], traffic_networks=traffic_networks)
        # 3) Merge the DGs, ESSs and TESSs
        nVariables = nVariables_first_stage
        neq = Aeq.shape[0]
        nineq = A.shape[0]

        nVariables_index_tess = zeros(nev + 1)
        neq_index_tess = zeros(nev + 1)
        nineq_index_tess = zeros(nev + 1)
        nVariables_index_tess[0] = nVariables_first_stage
        neq_index_tess[0] = neq
        nineq_index_tess[0] = nineq

        for i in range(nev):
            nVariables_index_tess[i + 1] = nVariables_index_tess[i] + len(model_tess[i]["c"])
            neq_index_tess[i + 1] = neq_index_tess[i] + model_tess[i]["Aeq"].shape[0]
            nineq_index_tess[i + 1] = nineq_index_tess[i] + model_tess[i]["A"].shape[0]
            nVariables += len(model_tess[i]["c"])
            neq += int(model_tess[i]["Aeq"].shape[0])
            nineq += int(model_tess[i]["A"].shape[0])

            c = concatenate([c, model_tess[i]["c"]])
            lx = concatenate([lx, model_tess[i]["lb"]])
            ux = concatenate([ux, model_tess[i]["ub"]])
            vtypes += model_tess[i]["vtypes"]
            beq = concatenate([beq, model_tess[i]["beq"]])
            b = concatenate([b, model_tess[i]["b"]])

        A_full = zeros((int(nineq_index_tess[-1]), int(nVariables_index_tess[-1])))
        Aeq_full = zeros((int(neq_index_tess[-1]), int(nVariables_index_tess[-1])))

        if Aeq is not None:
            Aeq_full[0:int(neq_index_tess[0]), 0:int(nVariables_index_tess[0])] = Aeq
        if A is not None:
            A_full[0:int(nineq_index_tess[0]), 0:int(nVariables_index_tess[0])] = A

        for i in range(nev):
            Aeq_full[int(neq_index_tess[i]):int(neq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["Aeq"]

            A_full[int(nineq_index_tess[i]):int(nineq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["A"]

        model_first_stage = {"c": c,
                             "lb": lx,
                             "ub": ux,
                             "vtypes": vtypes,
                             "A": A_full,
                             "b": b,
                             "Aeq": Aeq_full,
                             "beq": beq, }

        return model_first_stage

    def second_stage_problem_formualtion(self, power_networks, micro_grids, tess, traffic_networks, profile):
        """
        Second-stage problem formulation, the decision variables includes DGs within power networks, DGs within MGs, EESs within MGs and TESSs and other systems' information
        :param power_networks:
        :param micro_grids:
        :param tess:
        :param traffic_networks:
        :return: The second stage problems as list, including coupling constraints, and other constraint set
        """
        # I) Formulate the problem for distribution systems operator
        T = self.T  # Time slots

        mpc = ext2int(power_networks)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nmg = self.nmg
        nev = self.nev

        self.nl = nl
        self.nb = nb
        self.ng = ng

        m = zeros(nmg)  ## list of integration index
        Pmg_l = zeros(nmg)  ## list of lower boundary
        Pmg_u = zeros(nmg)  ## list of upper boundary
        Qmg_l = zeros(nmg)  ## list of lower boundary
        Qmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = micro_grids[i]["BUS"]
            Pmg_l[i] = micro_grids[i]["UG"]["PMIN"] / 1000 / baseMVA
            Pmg_u[i] = micro_grids[i]["UG"]["PMAX"] / 1000 / baseMVA
            Qmg_l[i] = micro_grids[i]["UG"]["QMIN"] / 1000 / baseMVA
            Qmg_u[i] = micro_grids[i]["UG"]["QMAX"] / 1000 / baseMVA

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        self.f = f  ## record from bus for each branch

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))

        Branch_R = branch[:, BR_R]
        Branch_X = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        Slmax = branch[:, RATE_A] / baseMVA

        Pij_l = -Slmax
        Qij_l = -Slmax
        Iij_l = zeros(nl)
        Vm_l = bus[:, VMIN] ** 2
        Pg_l = gen[:, PMIN] / baseMVA
        Qg_l = gen[:, QMIN] / baseMVA

        Pij_u = Slmax
        Qij_u = Slmax
        Iij_u = Slmax
        Vm_u = bus[:, VMAX] ** 2
        Pg_u = 2 * gen[:, PMAX] / baseMVA
        Qg_u = 2 * gen[:, QMAX] / baseMVA

        nx = int(3 * nl + nb + 2 * ng + 2 * nmg)
        self.nx = nx  # Number of decision variable within each time slot

        lx = concatenate([tile(concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l, Pmg_l, Qmg_l]), T)])
        ux = concatenate([tile(concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u, Pmg_u, Qmg_u]), T)])

        vtypes = ["c"] * nx * T
        nVariables_distribution_network = nx * T  # Number of total decision variables

        # Add system level constraints
        # 1) Active power balance
        Aeq_p = zeros((nb * T, nVariables_distribution_network))
        beq_p = zeros(nb * T)
        for i in range(T):
            Aeq_p[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([Ct - Cf, zeros((nb, nl)),
                                                                       -diag(Ct * Branch_R) * Ct,
                                                                       zeros((nb, nb)), Cg,
                                                                       zeros((nb, ng)), -Cmg,
                                                                       zeros((nb, nmg))]).toarray()

            beq_p[i * nb:(i + 1) * nb] = profile[i] * bus[:, PD] / baseMVA

        # 2) Reactive power balance
        Aeq_q = zeros((nb * T, nVariables_distribution_network))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([zeros((nb, nl)), Ct - Cf,
                                                                       -diag(Ct * Branch_X) * Ct,
                                                                       zeros((nb, nb)),
                                                                       zeros((nb, ng)), Cg,
                                                                       zeros((nb, nmg)), -Cmg]).toarray()
            beq_q[i * nb:(i + 1) * nb] = profile[i] * bus[:, QD] / baseMVA
        # 3) KVL equation
        Aeq_kvl = zeros((nl * T, nVariables_distribution_network))
        beq_kvl = zeros(nl * T)

        for i in range(T):
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx: i * nx + nl] = -2 * diag(Branch_R)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + nl: i * nx + 2 * nl] = -2 * diag(Branch_X)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + 2 * nl: i * nx + 3 * nl] = diag(Branch_R ** 2) + diag(Branch_X ** 2)
            Aeq_kvl[i * nl:(i + 1) * nl, i * nx + 3 * nl:i * nx + 3 * nl + nb] = (Cf.T - Ct.T).toarray()

        Aeq = vstack([Aeq_p, Aeq_q, Aeq_kvl]).toarray()
        beq = concatenate([beq_p, beq_q, beq_kvl])

        # 4) Pij**2+Qij**2<=Vi*Iij
        Qc = dict()
        for t in range(T):
            for i in range(nl):
                Qc[t * nl + i] = [[int(t * nx + i), int(t * nx + i + nl),
                                   int(t * nx + i + 2 * nl), int(t * nx + f[i] + 3 * nl)],
                                  [int(t * nx + i), int(t * nx + i + nl),
                                   int(t * nx + f[i] + 3 * nl), int(t * nx + i + 2 * nl)],
                                  [1, 1, -1 / 2, -1 / 2]]

        c = zeros(nVariables_distribution_network)
        q = zeros(nVariables_distribution_network)
        c0 = 0
        for t in range(T):
            for i in range(ng):
                c[t * nx + i + 3 * nl + nb] = gencost[i, 5] * baseMVA
                q[t * nx + i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
                c0 += gencost[i, 6]
        # Coupling constraints between the distribution systems and micro_grids
        Ax2y = zeros((2 * nmg * T, nVariables_distribution_network))  # connection matrix with the microgrids
        for i in range(T):
            for j in range(nmg):
                Ax2y[i * nmg + j, i * nx + 3 * nl + nb + 2 * ng + j] = 1000 * baseMVA  # Active power
                Ax2y[nmg * T + i * nmg + j,
                     i * nx + 3 * nl + nb + 2 * ng + nmg + j] = 1000 * baseMVA  # Reactive power

        # II) Formulate the problem for microgrids
        model_microgrids = {}
        for i in range(nmg):
            model_microgrids[i] = self.problem_formulation_microgrid(micro_grid=micro_grids[i], tess=tess)
        # II.A) Combine the distribution system operation problem and microgrid systems
        if Aeq is not None:
            neq_distribution_network = Aeq.shape[0]
        else:
            neq_distribution_network = 0

        nineq_distribution_network = 0

        nVariables = int(nVariables_distribution_network)
        neq = int(neq_distribution_network)

        nVariables_index = zeros(nmg + 1)
        neq_index = zeros(nmg + 1)
        nineq_index = zeros(nmg + 1)

        nVariables_index[0] = int(nVariables_distribution_network)
        neq_index[0] = int(neq_distribution_network)
        nineq_index[0] = int(nineq_distribution_network)
        for i in range(nmg):
            nVariables_index[i + 1] = nVariables_index[i] + len(model_microgrids[i]["c"])
            neq_index[i + 1] = neq_index[i] + model_microgrids[i]["Aeq"].shape[0]
            nVariables += len(model_microgrids[i]["c"])
            neq += int(model_microgrids[i]["Aeq"].shape[0])

        Aeq_full = zeros((int(neq_index[-1]), int(nVariables_index[-1])))

        Aeq_full[0:neq_distribution_network, 0:nVariables_distribution_network] = Aeq
        for i in range(nmg):
            lx = concatenate([lx, model_microgrids[i]["lb"]])
            ux = concatenate([ux, model_microgrids[i]["ub"]])
            c = concatenate([c, model_microgrids[i]["c"]])
            q = concatenate([q, model_microgrids[i]["q"]])
            vtypes += model_microgrids[i]["vtypes"]
            beq = concatenate([beq, model_microgrids[i]["beq"]])
            Aeq_full[int(neq_index[i]):int(neq_index[i + 1]), int(nVariables_index[i]):int(nVariables_index[i + 1])] = \
                model_microgrids[i]["Aeq"]

        # Add coupling constraints, between the microgrids and distribution networks
        Ay2x = zeros((2 * nmg * T, int(nVariables_index[-1] - nVariables_index[0])))
        for i in range(T):
            for j in range(nmg):
                Ay2x[i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + PUG] = -1
                Ay2x[nmg * T + i * nmg + j, int(nVariables_index[j] - nVariables_index[0]) + i * NX_MG + QUG] = -1

        Aeq_temp = concatenate([Ax2y, Ay2x], axis=1)
        beq_temp = zeros(2 * nmg * T)

        Aeq_full = concatenate([Aeq_full, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        sol = miqcp(c, q, Aeq=Aeq_full, beq=beq, A=None, b=None, Qc=Qc, xmin=lx, xmax=ux)

        model_second_stage = {"c": c,
                              "q": q,
                              "lb": lx,
                              "ub": ux,
                              "vtypes": vtypes,
                              "A": None,
                              "b": None,
                              "Aeq": Aeq,
                              "beq": beq,
                              "Qc": Qc,
                              "c0": c0}

        return model_second_stage

    def problem_formulation_microgrid(self, micro_grid, tess):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24
        nev = self.nev

        Pmess_l = 0
        Pmess_u = 0
        for i in range(nev):
            Pmess_l -= tess[i]["PCMAX"]
            Pmess_u += tess[i]["PDMAX"]

        ## 1) boundary information and objective function
        nx = NX_MG * T
        lx = zeros(nx)
        ux = zeros(nx)
        c = zeros(nx)
        q = zeros(nx)
        vtypes = ["c"] * nx
        for i in range(T):
            ## 1.1) lower boundary
            lx[i * NX_MG + PG] = 0
            lx[i * NX_MG + QG] = micro_grid["DG"]["QMIN"]
            lx[i * NX_MG + PUG] = 0
            lx[i * NX_MG + QUG] = micro_grid["UG"]["QMIN"]
            lx[i * NX_MG + PBIC_DC2AC] = 0
            lx[i * NX_MG + PBIC_AC2DC] = 0
            lx[i * NX_MG + QBIC] = -micro_grid["BIC"]["SMAX"]
            lx[i * NX_MG + PESS_CH] = 0
            lx[i * NX_MG + PESS_DC] = 0
            lx[i * NX_MG + EESS] = micro_grid["ESS"]["EMIN"]
            lx[i * NX_MG + PMESS] = Pmess_l
            ## 1.2) upper boundary
            ux[i * NX_MG + PG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + QG] = micro_grid["DG"]["QMAX"]
            ux[i * NX_MG + PUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + QUG] = micro_grid["UG"]["QMAX"]
            ux[i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + QBIC] = micro_grid["BIC"]["SMAX"]
            ux[i * NX_MG + PESS_CH] = micro_grid["ESS"]["PCH_MAX"]
            ux[i * NX_MG + PESS_DC] = micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + EESS] = micro_grid["ESS"]["EMAX"]
            ux[i * NX_MG + PMESS] = Pmess_u
            ## 1.3) Objective functions
            c[i * NX_MG + PG] = micro_grid["DG"]["COST_A"]
            ## 1.4) Upper and lower boundary information
            if i == T:
                lx[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]
                ux[i * NX_MG + EESS] = micro_grid["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = zeros((T, nx))
        beq = zeros(T)
        for i in range(T):
            Aeq[i, i * NX_MG + PG] = 1
            Aeq[i, i * NX_MG + PUG] = 1
            Aeq[i, i * NX_MG + PBIC_AC2DC] = -1
            Aeq[i, i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["EFF_DC2AC"]
            beq[i] = micro_grid["PD"]["AC"][i]
        # b) DC bus equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["EFF_AC2DC"]
            Aeq_temp[i, i * NX_MG + PBIC_DC2AC] = -1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -1
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1
            Aeq_temp[i, i * NX_MG + PMESS] = 1  # The power injection from mobile energy storage systems
            beq_temp[i] = micro_grid["PD"]["DC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + QUG] = 1
            Aeq_temp[i, i * NX_MG + QBIC] = 1
            Aeq_temp[i, i * NX_MG + QG] = 1
            beq_temp[i] = micro_grid["QD"]["AC"][i]
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = zeros((T, nx))
        beq_temp = zeros(T)
        for i in range(T):
            Aeq_temp[i, i * NX_MG + EESS] = 1
            Aeq_temp[i, i * NX_MG + PESS_CH] = -micro_grid["ESS"]["EFF_CH"]
            Aeq_temp[i, i * NX_MG + PESS_DC] = 1 / micro_grid["ESS"]["EFF_DC"]
            if i == 0:
                beq_temp[i] = micro_grid["ESS"]["E0"]
            else:
                Aeq_temp[i, (i - 1) * NX_MG + EESS] = -1
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequality constraints
        # There is no inequality constraint.

        # sol = milp(c, Aeq=Aeq, beq=beq, A=None, b=None, xmin=lx, xmax=ux)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lx,
                            "ub": ux,
                            "vtypes": vtypes,
                            "A": None,
                            "b": None,
                            "Aeq": Aeq,
                            "beq": beq
                            }

        return model_micro_grid

    def problem_formulation_tess(self, tess, traffic_networks):
        """
        Problem formulation for transportation energy storage scheduling, including vehicle routine problem and etc.
        :param tess: specific tess information
        :param traffic_network: transportation network information
        :return:
        """
        nb_traffic = self.nb_traffic
        T = self.T
        nb = self.nb

        nl_traffic = traffic_networks["branch"].shape[0]

        # Formulate the connection matrix between the transportation networks and power networks
        connection_matrix = zeros(((2 * nl_traffic + nb_traffic) * T, 4))
        weight = zeros(((2 * nl_traffic + nb_traffic) * T, 1))
        for i in range(T):
            for j in range(nl_traffic):
                # Add from matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, F_BUS] = traffic_networks["branch"][j, F_BUS] + \
                                                                                  i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, T_BUS] = traffic_networks["branch"][j, T_BUS] + \
                                                                                  traffic_networks["branch"][j, TIME] * \
                                                                                  nb_traffic + i * nb_traffic
                weight[i * (2 * nl_traffic + nb_traffic) + j, 0] = 1
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, TIME] = traffic_networks["branch"][j, TIME]

            for j in range(nl_traffic):
                # Add to matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, F_BUS] = \
                    traffic_networks["branch"][j, T_BUS] + i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, T_BUS] = \
                    traffic_networks["branch"][j, F_BUS] + traffic_networks["branch"][j, TIME] * nb_traffic + \
                    i * nb_traffic

                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, TIME] = \
                    traffic_networks["branch"][j, TIME]

            for j in range(nb_traffic):
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, F_BUS] = \
                    j + i * nb_traffic  # This time slot
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, T_BUS] = \
                    j + (i + 1) * nb_traffic  # The next time slot

                if traffic_networks["bus"][j, LOCATION] >= 0:
                    connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, 3] = \
                        traffic_networks["bus"][j, LOCATION] + i * nb  # Location information

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb_traffic)
        connection_matrix = connection_matrix[index, :]

        # add two virtual nodes to represent the initial and end status of vehicles
        # special attention should be paid here, as the original index has been modified!
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        # From matrix
        temp = zeros((nb_traffic, 4))
        for i in range(nb_traffic): temp[i, 1] = i + 1
        connection_matrix = concatenate([temp, connection_matrix])

        # To matrix
        for i in range(nb_traffic):
            temp = zeros((1, 4))
            temp[0, 0] = nb_traffic * (T - 1) + i + 1
            temp[0, 1] = nb_traffic * T + 1
            if traffic_networks["bus"][i, LOCATION] >= 0:
                temp[0, 3] = traffic_networks["bus"][i, LOCATION] + (T - 1) * nb
            connection_matrix = concatenate([connection_matrix, temp])

        # Status transition matrix
        nl_traffic = connection_matrix.shape[0]
        nb_traffic_electric = sum((traffic_networks["bus"][:, 2]) >= 0)
        # 0 represents that, the bus is not within the power networks

        status_matrix = zeros((T, nl_traffic))
        for i in range(T):
            for j in range(nl_traffic):
                if connection_matrix[j, F_BUS] >= i * nb_traffic + 1 and \
                        connection_matrix[j, F_BUS] < (i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb_traffic + 1 and \
                        connection_matrix[j, T_BUS] > (i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros((T * nb_traffic + 2, nl_traffic))
        connection_matrix_t = zeros((T * nb_traffic + 2, nl_traffic))

        for i in range(T * nb_traffic + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        n_stops = find(connection_matrix[:, 3]).__len__()

        assert n_stops == nb_traffic_electric * T, "The number of bus stop is not right!"

        NX_traffic = nl_traffic + 4 * n_stops  # Status transition, charging status, charging rate, discharging rate, spinning reserve
        NX_status = nl_traffic
        lx = zeros(NX_traffic)
        ux = ones(NX_traffic)

        self.NX_traffic = NX_traffic
        self.nl_traffic = nl_traffic
        self.n_stops = n_stops

        ux[NX_status + 0 * n_stops:NX_status + 1 * n_stops] = 1
        ux[NX_status + 1 * n_stops:NX_status + 2 * n_stops] = tess["PDMAX"]
        ux[NX_status + 2 * n_stops:NX_status + 3 * n_stops] = tess["PCMAX"]
        ux[NX_status + 3 * n_stops:NX_status + 4 * n_stops] = tess["PCMAX"] + tess["PDMAX"]
        # The initial location and stop location
        lx[find(connection_matrix[:, F_BUS] == 0)] = tess["initial"]
        ux[find(connection_matrix[:, F_BUS] == 0)] = tess["initial"]
        lx[find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = tess["end"]
        ux[find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = tess["end"]

        vtypes = ["b"] * NX_status + ["b"] * n_stops + ["c"] * 3 * n_stops

        Aeq = connection_matrix_f - connection_matrix_t
        beq = zeros(T * nb_traffic + 2)
        beq[0] = 1
        beq[-1] = -1
        # statue constraints
        Aeq_temp = status_matrix
        beq_temp = ones(T)
        Aeq = concatenate([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        neq_traffic = Aeq.shape[0]
        # Fulfill the missing zeros
        Aeq = concatenate([Aeq, zeros((neq_traffic, 4 * n_stops))], axis=1)

        ## Inequality constraints
        index_stops = find(connection_matrix[:, 3])
        index_operation = arange(n_stops)
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, NX_status))
        # This mapping matrix plays an important role in the connection between the power network and traffic network
        ## 1) Stopping status
        A = zeros((3 * n_stops, NX_traffic))  # Charging, discharging status, RBS
        # Discharging
        A[0:n_stops, 0: NX_status] = -power_limit.toarray() * tess["PDMAX"]
        A[0:n_stops, NX_status + n_stops: NX_status + 2 * n_stops] = eye(n_stops)
        # Charging
        A[n_stops:n_stops * 2, 0: NX_status] = -power_limit.toarray() * tess["PCMAX"]
        A[n_stops:n_stops * 2, NX_status + 2 * n_stops:NX_status + 3 * n_stops] = eye(n_stops)
        # spinning reserve
        A[n_stops * 2: n_stops * 3, 0: NX_status] = -power_limit.toarray() * (tess["PCMAX"] + tess["PDMAX"])
        A[n_stops * 2:n_stops * 3, NX_status + 3 * n_stops:NX_status + 4 * n_stops] = eye(n_stops)
        b = zeros(3 * n_stops)

        ## 2) Operating status
        Arange = zeros((2 * n_stops, NX_traffic))
        brange = zeros(2 * n_stops)
        # 1) Pdc<(1-Ic)*Pdc_max
        Arange[0: n_stops, NX_status:NX_status + n_stops] = eye(n_stops) * tess["PDMAX"]
        Arange[0: n_stops, NX_status + n_stops: NX_status + n_stops * 2] = eye(n_stops)
        brange[0: n_stops] = ones(n_stops) * tess["PDMAX"]
        # 2) Pc<Ic*Pch_max
        Arange[n_stops:n_stops * 2, NX_status: NX_status + n_stops] = -eye(n_stops) * tess["PCMAX"]
        Arange[n_stops:n_stops * 2, NX_status + n_stops * 2: NX_status + n_stops * 3] = eye(n_stops)
        A = concatenate([A, Arange])
        b = concatenate([b, brange])

        ## 2) Power limitation
        Areserve = zeros((2 * n_stops, NX_traffic))
        breserve = zeros(2 * n_stops)
        # 1) Pdc-Pc+Rbs<=Pdc_max
        Areserve[0: n_stops, NX_status + n_stops: NX_status + n_stops * 2] = eye(n_stops)
        Areserve[0: n_stops, NX_status + n_stops * 2:NX_status + n_stops * 3] = -eye(n_stops)
        Areserve[0: n_stops, NX_status + n_stops * 3:NX_status + n_stops * 4] = eye(n_stops)
        breserve[0: n_stops] = ones(n_stops) * tess["PDMAX"]
        # 2) Pc-Pdc+Rbs<=Pc_max
        Areserve[n_stops:n_stops * 2, NX_status + n_stops: NX_status + n_stops * 2] = - eye(n_stops)
        Areserve[n_stops:n_stops * 2, NX_status + n_stops * 2:NX_status + n_stops * 3] = eye(n_stops)
        Areserve[n_stops:n_stops * 2, NX_status + n_stops * 3:NX_status + n_stops * 4] = eye(n_stops)
        breserve[n_stops:n_stops * 2] = ones(n_stops) * tess["PCMAX"]

        A = concatenate([A, Areserve])
        b = concatenate([b, breserve])

        # Add constraints on the energy status
        Aenergy = zeros((2 * T, NX_traffic))
        benergy = zeros(2 * T)
        for j in range(T):
            # minimal energy
            Aenergy[j, NX_status + n_stops: NX_status + n_stops + (j + 1) * nb_traffic_electric] = 1 / tess["EFF_DC"]
            Aenergy[j, NX_status + 2 * n_stops: NX_status + 2 * n_stops + (j + 1) * nb_traffic_electric] = \
                -tess["EFF_CH"]
            Aenergy[j, NX_status + 3 * n_stops + (j + 1) * nb_traffic_electric - 1] = 0.5
            if j != (T - 1):
                benergy[j] = tess["E0"] - tess["EMIN"]
            else:
                benergy[j] = 0
            # maximal energy
            Aenergy[T + j, NX_status + n_stops: NX_status + n_stops + (j + 1) * nb_traffic_electric] = \
                -1 / tess["EFF_DC"]
            Aenergy[T + j, NX_status + 2 * n_stops: i * NX_traffic + NX_status + 2 * n_stops +
                                                    (j + 1) * nb_traffic_electric] = tess["EFF_CH"]
            if j != (T - 1):
                benergy[T + j] = tess["EMAX"] - tess["E0"]
            else:
                benergy[T + j] = 0

        A = concatenate([A, Aenergy])
        b = concatenate([b, benergy])

        # sol = milp(zeros(NX_traffic), q=zeros(NX_traffic), Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux)

        model_tess = {"c": zeros(NX_traffic),
                      "q": zeros(NX_traffic),
                      "lb": lx,
                      "ub": ux,
                      "vtypes": vtypes,
                      "A": A,
                      "b": b,
                      "Aeq": Aeq,
                      "beq": beq,
                      "NX": NX_traffic, }

        return model_tess

    def solution_check_tess(self, sol):
        """
        :param sol: solutions for tess
        :return: decoupled solutions for tess
        """

        NX_traffic = self.NX_traffic
        nl_traffic = self.nl_traffic
        n_stops = self.n_stops
        nev = self.nev
        T = self.T

        tsn_ev = zeros((nl_traffic, nev))
        ich_ev = zeros((n_stops, nev))
        pdc_ev = zeros((n_stops, nev))
        pch_ev = zeros((n_stops, nev))
        rs_ev = zeros((n_stops, nev))

        for i in range(nev):
            for j in range(nl_traffic):
                tsn_ev[j, i] = sol[i * NX_traffic + j]
            for j in range(n_stops):
                ich_ev[j, i] = sol[i * NX_traffic + nl_traffic + 0 * n_stops + j]
            for j in range(n_stops):
                pdc_ev[j, i] = sol[i * NX_traffic + nl_traffic + 1 * n_stops + j]
            for j in range(n_stops):
                pch_ev[j, i] = sol[i * NX_traffic + nl_traffic + 2 * n_stops + j]
            for j in range(n_stops):
                rs_ev[j, i] = sol[i * NX_traffic + nl_traffic + 3 * n_stops + j]

        sol_tess = {"Tsn_ev": tsn_ev,
                    "Ich": ich_ev,
                    "Pdc": pdc_ev,
                    "Pch": pch_ev,
                    "Rs": rs_ev, }

        return sol_tess

    def scenario_generation(self, microgrids, profile, Ns=2):
        """
        Scenario generation function for the second-stage scheduling
        :return:
        """
        T = self.T
        nmg = self.nmg
        profile_second_stage = zeros((Ns, T))
        microgrids_second_stage = [0] * Ns
        for i in range(Ns):
            for j in range(T):
                profile_second_stage[i, j] = profile[j] * (0.9 + 0.1 * random.random())

        for i in range(Ns):
            microgrids_second_stage[i] = deepcopy(microgrids)
            for k in range(nmg):
                for j in range(T):
                    microgrids_second_stage[i][k]["PD"]["AC"][j] = microgrids_second_stage[i][k]["PD"]["AC"][j] * (
                            0.9 + 0.1 * random.random())
                    microgrids_second_stage[i][k]["QD"]["AC"][j] = microgrids_second_stage[i][k]["QD"]["AC"][j] * (
                            0.9 + 0.1 * random.random())
                    microgrids_second_stage[i][k]["PD"]["DC"][j] = microgrids_second_stage[i][k]["PD"]["DC"][j] * (
                            0.9 + 0.1 * random.random())

        return profile_second_stage, microgrids_second_stage

    def scenario_redunction(self):
        """
        Scenario generation function for the second-stage scheduling
        :return:
        """


if __name__ == "__main__":
    # Distribution network information
    mpc = case33.case33()  # Default test case
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41]) * 2

    # Microgrid information
    Profile = array([
        [0.64, 0.63, 0.65, 0.64, 0.66, 0.69, 0.75, 0.91, 0.95, 0.97, 1.00, 0.97, 0.97, 0.95, 0.98, 0.99, 0.95, 0.95,
         0.94, 0.95, 0.97, 0.93, 0.85, 0.69],
        [0.78, 0.75, 0.74, 0.74, 0.75, 0.81, 0.91, 0.98, 0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95,
         0.95, 0.95, 0.96, 0.95, 0.88, 0.82],
        [0.57, 0.55, 0.55, 0.56, 0.62, 0.70, 0.78, 0.83, 0.84, 0.89, 0.87, 0.82, 0.80, 0.80, 0.84, 0.89, 0.94, 0.98,
         1.00, 0.97, 0.87, 0.79, 0.72, 0.62]
    ])
    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["BUS"] = 2
    micro_grid_1["PD"]["AC_MAX"] = 10
    micro_grid_1["PD"]["DC_MAX"] = 10
    micro_grid_1["UG"]["PMIN"] = -500
    micro_grid_1["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_1["DG"]["PMAX"] = 100
    micro_grid_1["DG"]["QMAX"] = 100
    micro_grid_1["DG"]["QMIN"] = -100
    micro_grid_1["DG"]["COST_A"] = 0.015
    micro_grid_1["ESS"]["PDC_MAX"] = 50
    micro_grid_1["ESS"]["PCH_MAX"] = 50
    micro_grid_1["ESS"]["E0"] = 50
    micro_grid_1["ESS"]["EMIN"] = 10
    micro_grid_1["ESS"]["EMAX"] = 100
    micro_grid_1["BIC"]["PMAX"] = 200
    micro_grid_1["BIC"]["QMAX"] = 200
    micro_grid_1["BIC"]["SMAX"] = 200
    micro_grid_1["PD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["QD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"] * 0.2
    micro_grid_1["PD"]["DC"] = Profile[0] * micro_grid_1["PD"]["DC_MAX"]
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["BUS"] = 4
    micro_grid_2["PD"]["AC_MAX"] = 50
    micro_grid_2["PD"]["DC_MAX"] = 50
    micro_grid_2["UG"]["PMIN"] = -500
    micro_grid_2["UG"]["PMAX"] = 500
    micro_grid_1["UG"]["QMIN"] = -500
    micro_grid_1["UG"]["QMAX"] = 500
    micro_grid_2["DG"]["PMAX"] = 50
    micro_grid_1["DG"]["QMAX"] = 50
    micro_grid_1["DG"]["QMIN"] = -50
    micro_grid_2["DG"]["COST_A"] = 0.01
    micro_grid_2["ESS"]["PDC_MAX"] = 50
    micro_grid_2["ESS"]["PCH_MAX"] = 50
    micro_grid_2["ESS"]["E0"] = 15
    micro_grid_2["ESS"]["EMIN"] = 10
    micro_grid_2["ESS"]["EMAX"] = 50
    micro_grid_2["BIC"]["PMAX"] = 200
    micro_grid_2["BIC"]["QMAX"] = 200
    micro_grid_2["BIC"]["SMAX"] = 200
    micro_grid_2["PD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"]
    micro_grid_2["QD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"] * 0.2
    micro_grid_2["PD"]["DC"] = Profile[1] * micro_grid_2["PD"]["DC_MAX"]
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["BUS"] = 10
    micro_grid_3["PD"]["AC_MAX"] = 50
    micro_grid_3["PD"]["DC_MAX"] = 50
    micro_grid_3["UG"]["PMIN"] = -500
    micro_grid_3["UG"]["PMAX"] = 500
    micro_grid_3["UG"]["QMIN"] = -500
    micro_grid_3["UG"]["QMAX"] = 500
    micro_grid_3["DG"]["PMAX"] = 50
    micro_grid_3["DG"]["QMAX"] = 50
    micro_grid_3["DG"]["QMIN"] = -50
    micro_grid_3["DG"]["COST_A"] = 0.01
    micro_grid_3["ESS"]["PDC_MAX"] = 50
    micro_grid_3["ESS"]["PCH_MAX"] = 50
    micro_grid_3["ESS"]["E0"] = 20
    micro_grid_3["ESS"]["EMIN"] = 10
    micro_grid_3["ESS"]["EMAX"] = 50
    micro_grid_3["BIC"]["PMAX"] = 50
    micro_grid_3["BIC"]["QMAX"] = 200
    micro_grid_3["BIC"]["SMAX"] = 200
    micro_grid_3["PD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["QD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"] * 0.2
    micro_grid_3["PD"]["DC"] = Profile[2] * micro_grid_3["PD"]["DC_MAX"]
    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]

    ## Transportaion network information
    ev = []
    traffic_networks = case3.transportation_network()  # Default transportation networks
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    """
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 1, 0]),
               "PCMAX": 1000,
               "PDMAX": 1000,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 200,
               "PDMAX": 200,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 100,
               "EMAX": 200,
               "EMIN": 50,
               "COST_OP": 0.01,
               })
    """

    stochastic_dynamic_optimal_power_flow = StochasticDynamicOptimalPowerFlowTess()

    (sol_dso, sol_mgs, sol_tess) = stochastic_dynamic_optimal_power_flow.main(case=mpc, profile=load_profile.tolist(),
                                                                              micro_grids=case_micro_grids, tess=ev,
                                                                              traffic_networks=traffic_networks)

    print(max(sol_dso["residual"][0]))
