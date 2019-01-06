"""
Stochastic optimal power flow with multiple microgrids and mobile energy storage systems
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date: 4 Jan 2019
"""

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid
from transportation_systems.test_cases import case3, TIME, LOCATION

from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack
from numpy import flatnonzero as find
from numpy import array, tile, arange, random

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, RATE_A
from pypower.idx_bus import BUS_TYPE, REF, PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp

from copy import deepcopy

from distribution_system_optimization.data_format.idx_opf import PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, RUG, \
    RESS, RG, EESS, NX_MG, QBIC, QUG, QG


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
        sol_first_stage = milp(model_first_stage["c"], Aeq=model_first_stage["Aeq"], beq=model_first_stage["beq"],
                               A=model_first_stage["A"], b=model_first_stage["b"], vtypes=model_first_stage["vtypes"],
                               xmax=model_first_stage["ub"], xmin=model_first_stage["lb"])
        # 2) Formualte the second stage optimization problem

        # 3) Obtain the results for first-stage and second stage optimization problems

        # 4) Verify the first-stage and second stage optization problem

        # 1.1) Distribution networks

        # 4) Formulate the second stage problem, under different scenarios

        # return sol_distribution_network, sol_microgrids, sol_tess
        return sol_first_stage

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

        Pess_ch_u = zeros(nmg)
        Pess_dc_u = zeros(nmg)
        Eess_u = zeros(nmg)
        Ress_u = zeros(nmg)
        cess_ch = zeros(nmg)
        cess_dc = zeros(nmg)
        cess_r = zeros(nmg)
        cess = zeros(nmg)

        for i in range(nmg):
            Pess_ch_u[i] = micro_grids[i]["ESS"]["PCH_MAX"]
            Pess_dc_u[i] = micro_grids[i]["ESS"]["PDC_MAX"] + micro_grids[i]["ESS"]["PCH_MAX"]
            Ress_u[i] = micro_grids[i]["ESS"]["PCH_MAX"]
            Eess_l[i] = micro_grids[i]["ESS"]["EMIN"]
            Eess_u[i] = micro_grids[i]["ESS"]["EMAX"]

        NX_first_stage = ng * 2 + nmg * 2 + nmg * 4
        nx_first_stage = (ng * 2 + nmg * 2 + nmg * 4) * T
        # Formulate the boundaries
        lx = concatenate([tile(concatenate([Pg_l, Rg_l, Pg_mg_l, Rg_mg_l, Pess_ch_l, Pess_dc_l, Eess_l, Ress_l]), T)])
        ux = concatenate([tile(concatenate([Pg_u, Rg_u, Pg_mg_u, Rg_mg_u, Pess_ch_u, Pess_dc_u, Eess_u, Ress_u]), T)])
        # Objective value
        c = concatenate([tile(concatenate([cg, cr, cg_mg, cr_mg, cess_ch, cess_dc, cess, cess_r]), T)])
        # Variable types
        vtypes = ["c"] * nx_first_stage

        ## Constraint sets
        # 1) Pg+Rg<=Pgu
        A = zeros((ng, nx_first_stage))
        b = zeros(ng)
        for i in range(ng):
            A[i, i] = 1
            A[i, ng + i] = 1
            b[i] = Pg_u[i]
        # 2) Pg-Rg>=Pgl
        A_temp = zeros((ng, nx_first_stage))
        b_temp = zeros(ng)
        for i in range(ng):
            A_temp[i, i] = -1
            A_temp[i, ng + i] = 1
            b_temp[i] = -Pg_l[i]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3) Pg_mg+Rg_mg<=Pg_mg_u
        A_temp = zeros((nmg, nx_first_stage))
        b_temp = zeros(nmg)
        for i in range(nmg):
            A_temp[i, ng * 2 + i] = 1
            A_temp[i, ng * 2 + nmg + i] = 1
            b_temp[i] = Pg_mg_u[i]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 4) Pg_mg-Rg_mg<=Pg_mg_l
        A_temp = zeros((nmg, nx_first_stage))
        b_temp = zeros(nmg)
        for i in range(nmg):
            A_temp[i, ng * 2 + i] = -1
            A_temp[i, ng * 2 + nmg + i] = 1
            b_temp[i] = Pg_mg_l[i]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 5) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((nmg, nx_first_stage))
        b_temp = zeros(nmg)
        for i in range(nmg):
            A_temp[i, ng * 2 + nmg * 2 + i] = -1
            A_temp[i, ng * 2 + nmg * 2 + nmg + i] = 1
            A_temp[i, ng * 2 + nmg * 2 + nmg * 2 + i] = 1
            b_temp[i] = Pess_dc_u[i]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 6) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((nmg, nx_first_stage))
        b_temp = zeros(nmg)
        for i in range(nmg):
            A_temp[i, ng * 2 + nmg * 2 + i] = 1
            A_temp[i, ng * 2 + nmg * 2 + nmg + i] = -1
            A_temp[i, ng * 2 + nmg * 2 + nmg * 2 + i] = 1
            b_temp[i] = Pess_ch_u[i]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 7) Energy storage balance equation
        Aeq = zeros((T * nmg, nx_first_stage))
        beq = zeros(T * nmg)
        for i in range(T):
            for j in range(nmg):
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg * 3 + j] = 1
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + j] = -micro_grids[j]["ESS"]["EFF_CH"]
                Aeq[i * nmg + j, i * NX_first_stage + ng * 2 + nmg * 2 + nmg + j] = 1 / micro_grids[j]["ESS"]["EFF_DC"]
                if i == 0:
                    beq[i * nmg + j] = micro_grids[j]["ESS"]["E0"]
                else:
                    Aeq[i * nmg + j, (i - 1) * NX_first_stage + ng * 2 + nmg * 2 + nmg * 3 + j] = -1

        model_first_stage = {"c": c,
                             "lb": lx,
                             "ub": ux,
                             "vtypes": vtypes,
                             "A": A,
                             "b": b,
                             "Aeq": Aeq,
                             "beq": beq, }

        return model_first_stage

        # The decision variables include the output of generators within MGs, ESSs and TESSs scheduling

    def problem_formulation_microgrid(self, micro_grid):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24

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
            lx[i * NX_MG + RG] = 0
            lx[i * NX_MG + PUG] = 0
            lx[i * NX_MG + QUG] = micro_grid["UG"]["QMIN"]
            lx[i * NX_MG + RUG] = 0
            lx[i * NX_MG + PBIC_DC2AC] = 0
            lx[i * NX_MG + PBIC_AC2DC] = 0
            lx[i * NX_MG + QBIC] = -micro_grid["BIC"]["SMAX"]
            lx[i * NX_MG + PESS_CH] = 0
            lx[i * NX_MG + PESS_DC] = 0
            lx[i * NX_MG + RESS] = 0
            lx[i * NX_MG + EESS] = micro_grid["ESS"]["EMIN"]

            ## 1.2) upper boundary
            ux[i * NX_MG + PG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + QG] = micro_grid["DG"]["QMAX"]
            ux[i * NX_MG + RG] = micro_grid["DG"]["PMAX"]
            ux[i * NX_MG + PUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + QUG] = micro_grid["UG"]["QMAX"]
            ux[i * NX_MG + RUG] = micro_grid["UG"]["PMAX"]
            ux[i * NX_MG + PBIC_DC2AC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + PBIC_AC2DC] = micro_grid["BIC"]["PMAX"]
            ux[i * NX_MG + QBIC] = micro_grid["BIC"]["SMAX"]
            ux[i * NX_MG + PESS_CH] = micro_grid["ESS"]["PCH_MAX"]
            ux[i * NX_MG + PESS_DC] = micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + RESS] = micro_grid["ESS"]["PCH_MAX"] + micro_grid["ESS"]["PDC_MAX"]
            ux[i * NX_MG + EESS] = micro_grid["ESS"]["EMAX"]

            ## 1.3) Objective functions
            c[i * NX_MG + PG] = micro_grid["DG"]["COST_A"]
            # c[i * NX_MG + PUG] = micro_grid["UG"]["COST"][i]

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
        # 3) Formualte inequal constraints
        # 3.1) Pg+Rg<=Ig*Pgmax
        A = zeros((T, nx))
        b = zeros(T)
        for i in range(T):
            A[i, i * NX_MG + PG] = 1
            A[i, i * NX_MG + RG] = 1
            b[i] = micro_grid["DG"]["PMAX"]
        # 3.2) Pg-Rg>=Ig*Pgmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + RG] = 1
            A_temp[i, i * NX_MG + PG] = -1
            b_temp[i] = -micro_grid["DG"]["PMIN"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.3) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_DC] = 1
            A_temp[i, i * NX_MG + PESS_CH] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PDC_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.4) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PESS_CH] = 1
            A_temp[i, i * NX_MG + PESS_DC] = -1
            A_temp[i, i * NX_MG + RESS] = 1
            b_temp[i] = micro_grid["ESS"]["PCH_MAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.5) Pug+Rug<=Iug*Pugmax
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + PUG] = 1
            A_temp[i, i * NX_MG + RUG] = 1
            b_temp[i] = micro_grid["UG"]["PMAX"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])
        # 3.6) Pug-Rug>=Iug*Pugmin
        A_temp = zeros((T, nx))
        b_temp = zeros(T)
        for i in range(T):
            A_temp[i, i * NX_MG + RUG] = 1
            A_temp[i, i * NX_MG + PUG] = -1
            b_temp[i] = -micro_grid["DG"]["PMIN"]
        A = concatenate([A, A_temp])
        b = concatenate([b, b_temp])

        # sol = milp(c, q=q, Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux)

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lx,
                            "ub": ux,
                            "vtypes": vtypes,
                            "A": A,
                            "b": b,
                            "Aeq": Aeq,
                            "beq": beq,
                            "NX": NX_MG,
                            "PG": PG,
                            "QG": QG}

        return model_micro_grid

    def solution_check_microgrids(self, xx, nVariables_index):
        T = self.T
        nmg = self.nmg

        Pess_dc = zeros((nmg, T))
        Pess_ch = zeros((nmg, T))
        Ress = zeros((nmg, T))
        Eess = zeros((nmg, T))
        # b) Diesel generator group
        Pg = zeros((nmg, T))
        Qg = zeros((nmg, T))
        Rg = zeros((nmg, T))
        # c) Utility grid group
        Pug = zeros((nmg, T))
        Qug = zeros((nmg, T))
        Rug = zeros((nmg, T))
        # d) Bi-directional converter group
        Pbic_a2d = zeros((nmg, T))
        Pbic_d2a = zeros((nmg, T))
        Qbic = zeros((nmg, T))
        for i in range(T):
            for j in range(nmg):
                Pess_dc[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_DC]
                Pess_ch[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PESS_CH]
                Ress[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RESS]
                Eess[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + EESS]

                Pg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PG]
                Qg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QG]
                Rg[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RG]

                Pug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PUG]
                Qug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QUG]
                Rug[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + RUG]

                Pbic_a2d[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_AC2DC]
                Pbic_d2a[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + PBIC_DC2AC]
                Qbic[j, i] = xx[int(nVariables_index[j]) + i * NX_MG + QBIC]
        # e) voilation of bi-directional power flows
        vol_bic = zeros((nmg, T))
        vol_ess = zeros((nmg, T))
        for i in range(T):
            for j in range(nmg):
                vol_ess[j, i] = Pess_dc[j, i] * Pess_ch[j, i]
                vol_bic[j, i] = Pbic_a2d[j, i] * Pbic_d2a[j, i]

        sol_microgrids = {"PESS_DC": Pess_dc,
                          "PESS_CH": Pess_ch,
                          "RESS": Ress,
                          "EESS": Eess,
                          "PG": Pg,
                          "QG": Qg,
                          "RG": Rg,
                          "PUG": Pug,
                          "QUG": Qug,
                          "RUG": Rug,
                          "PBIC_AC2DC": Pbic_a2d,
                          "PBIC_DC2AC": Pbic_d2a,
                          "QBIC": Qbic,
                          "VOL_BIC": vol_bic,
                          "VOL_ESS": vol_ess, }

        return sol_microgrids

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

        # Formulate the connection matrix between the transportaion networks and power networks
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
        A = zeros((3 * n_stops, NX_traffic))  # Charging, discharging status,RBS
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
        # 2) Pc-Pdc<=Pc_max
        Areserve[n_stops:n_stops * 2, NX_status + n_stops: NX_status + n_stops * 2] = -eye(n_stops)
        Areserve[n_stops:n_stops * 2, NX_status + n_stops * 2:NX_status + n_stops * 3] = eye(n_stops)
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
                profile_second_stage[i, :] = profile[j] * (0.9 + 0.1 * random.random())

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
    micro_grid_1["BIC"]["PMAX"] = 100
    micro_grid_1["BIC"]["QMAX"] = 100
    micro_grid_1["BIC"]["SMAX"] = 100
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
    micro_grid_2["BIC"]["PMAX"] = 100
    micro_grid_2["BIC"]["QMAX"] = 100
    micro_grid_2["BIC"]["SMAX"] = 100
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
    micro_grid_3["BIC"]["QMAX"] = 100
    micro_grid_3["BIC"]["SMAX"] = 100
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
