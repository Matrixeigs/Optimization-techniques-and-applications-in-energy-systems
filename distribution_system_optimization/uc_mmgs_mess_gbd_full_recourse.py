"""
Stochastic optimal power flow with multiple microgrids and mobile energy storage systems
@author: Zhao Tianyang
@e-mail: zhaoty@ntu.edu.sg
@date: 7 August 2019
Major updates:
1) Solve using generalized benders decomposition
2) Using the
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
from pypower.idx_bus import PD, VMAX, VMIN, QD, BUS_I
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.quadratic_constained_cplex import quadratic_constrained_programming as qcqp
from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp
from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from copy import deepcopy

from distribution_system_optimization.data_format.idx_MG_PV import PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, \
    PMESS, EESS, NX_MG, QBIC, QUG, QG, PPV, PAC, PDC

from distribution_system_optimization.database_management_pv import DataBaseManagement
from solvers.scenario_reduction import ScenarioReduction

from multiprocessing import Pool
import os

from matplotlib import pyplot as plt
class StochasticUnitCommitmentTess():
    def __init__(self):
        self.name = "Unit commitment flow with tess"
        self.bigM = 1e3

    def main(self, power_networks, micro_grids, profile, pv_profile, mess, traffic_networks, ns=100,ns_reduced=50):
        """
        Main entrance for network reconfiguration problems
        :param case: electric network information
        :param profile: load profile within the distribution networks
        :param micrgrids: dictionary for microgrids
        :param tess: dictionary for tess
        :return: network reconfiguration, distribution network status, and microgrid status
        """
        T = len(profile)  # Time spans
        self.T = T
        nmg = len(micro_grids)  # Number of microgrids
        self.nmg = nmg
        nmes = len(mess)  # Number of mobile energy storage systems
        self.nmes = nmes
        nb_tra = traffic_networks["bus"].shape[0]  # Number of buses in the transportation networks
        self.nb_tra = nb_tra
        assert nb_tra == nmg, "The microgrids within the transportation networks are not synchronized!"

        # 1) Formulate the first stage optimization problem
        model_first_stage = self.first_stage_problem_formualtion(pns=power_networks, mgs=micro_grids, mess=mess,
                                                                 tns=traffic_networks)

        # 2) Formulate the second stage optimization problem
        # Formulate the second stage scenarios
        (ds_second_stage, mgs_second_stage, weight) = self.scenario_generation_reduction(profile=profile,
                                                                                         micro_grids=micro_grids, ns=ns,
                                                                                         pns=power_networks,
                                                                                         pv_profile=pv_profile,
                                                                                         ns_reduced=ns_reduced)
        ns -= ns_reduced
        model_second_stage = {}
        for i in range(ns):
            model_second_stage[i] = self.second_stage_problem_formualtion_gdb(pns=power_networks, mgs=mgs_second_stage[i],
                                                                          mess=mess, tns=traffic_networks,
                                                                          profile=ds_second_stage[i, :], index=i,
                                                                          weight=weight[i])

        # 3) Formulate the primal problem and dual-problem for each scenario
        # modify the first-stage problem, add ns variables, generate the first problem
        master_problem = deepcopy(model_first_stage)
        master_problem["c"] = concatenate([master_problem["c"], ones(ns)])
        master_problem["lb"] = concatenate([master_problem["lb"], zeros(ns)])
        master_problem["ub"] = concatenate([master_problem["ub"], ones(ns) * self.bigM * 1e4])
        master_problem["vtypes"] = master_problem["vtypes"] + ["c"] * ns
        master_problem["A"] = hstack([master_problem["A"], zeros((master_problem["A"].shape[0], ns))]).tolil()
        master_problem["Aeq"] = hstack([master_problem["Aeq"], zeros((master_problem["Aeq"].shape[0], ns))]).tolil()

        (sol_first_stage, obj_first_stage, success) = milp(master_problem["c"], Aeq=master_problem["Aeq"],
                                                           beq=master_problem["beq"],
                                                           A=master_problem["A"], b=master_problem["b"],
                                                           vtypes=master_problem["vtypes"],
                                                           xmax=master_problem["ub"], xmin=master_problem["lb"])

        assert success == 1, "The master problem is infeasible!"
        # 4) Solve this problem using Generalized Bender decomposition algorithm!
        iter = 0
        iter_max = 10000
        Gap_index = zeros(iter_max)
        LB_index = zeros(iter_max)
        UB_index = zeros(iter_max)
        LB_index[iter] = obj_first_stage
        UB_index[iter] = self.bigM * 1e5
        Gap_index[iter] = abs(UB_index[iter] - LB_index[iter]) / UB_index[iter]
        n_processors = os.cpu_count()

        while iter < iter_max and Gap_index[iter] > 0.2:
            problem_second_stage = {}
            problem_second_stage_relaxed = {}
            sub_problem = []
            for i in range(ns):
                (problem_second_stage[i], problem_second_stage_relaxed[i]) = \
                    self.second_stage_problem(x=sol_first_stage[0:self.nv_first_stage], model=model_second_stage[i])
                # print(max(problem_second_stage[i]["b"]))
                sub_problem.append([problem_second_stage[i], problem_second_stage_relaxed[i]])

            sol_second_stage = {}
            obj_second_stage = zeros(ns)
            success_second_stage = zeros(ns)
            slack_eq = {}
            slack_ineq = {}
            cuts = lil_matrix((ns, self.nv_first_stage+ns))
            b_cuts = zeros(ns)
            # sol = [0] * ns
            # for i in range(ns):
            #     sol[i] = sub_problem_solving(sub_problem[i])

            with Pool(n_processors) as p:
                sol = list(p.map(sub_problem_solving, sub_problem))

            for i in range(ns):
                sol_second_stage[i] = sol[i][0]
                obj_second_stage[i] = sol[i][1]
                success_second_stage[i] = sol[i][2]
                slack_eq[i] = sol[i][3]
                slack_ineq[i] = sol[i][4]
                if success_second_stage[i] == 0:
                    if sum(slack_ineq[i])<0:
                        cuts[i, 0:self.nv_first_stage] = -problem_second_stage_relaxed[i]["Ts"].transpose() * slack_ineq[i][0: problem_second_stage_relaxed[i]["Ts"].shape[0]]
                        b_cuts[i] = -(obj_second_stage[i] + array(slack_ineq[i][0: problem_second_stage_relaxed[i]["Ts"].shape[0]]).dot(
                            problem_second_stage[i]["Ts"] * array(sol_first_stage[0:self.nv_first_stage])))
                    else:
                        cuts[i, 0:self.nv_first_stage] = problem_second_stage_relaxed[i]["Ts"].transpose() * \
                                                         slack_ineq[i]
                        b_cuts[i] = -(obj_second_stage[i] - array(slack_ineq[i]).dot(
                            problem_second_stage[i]["Ts"] * array(sol_first_stage[0:self.nv_first_stage])))
                else:  # optimal cuts
                    if sum(slack_ineq[i])<0:
                        cuts[i, 0:self.nv_first_stage] = -problem_second_stage[i]["Ts"].transpose() * slack_ineq[i]
                        cuts[i, self.nv_first_stage +i] = -1
                        b_cuts[i] = -(obj_second_stage[i] + array(slack_ineq[i]).dot(
                            problem_second_stage[i]["Ts"] * array(sol_first_stage[0:self.nv_first_stage])))
                    else:
                        cuts[i, 0:self.nv_first_stage] = problem_second_stage[i]["Ts"].transpose() * slack_ineq[i]
                        cuts[i, self.nv_first_stage + i] = -1
                        b_cuts[i] = -(obj_second_stage[i] - array(slack_ineq[i]).dot(
                            problem_second_stage[i]["Ts"] * array(sol_first_stage[0:self.nv_first_stage])))

            master_problem["A"] = vstack([master_problem["A"], cuts]).tolil()
            master_problem["b"] = concatenate([master_problem["b"], b_cuts])

            iter += 1

            if sum(success_second_stage) < ns:
                UB_index[iter] = min(self.bigM * 1e6, UB_index[iter - 1])
                Gap_index[iter] = self.bigM
                print("The violation is {0}".format(sum(obj_second_stage)))
            else:
                UB_index[iter] = min(
                    model_first_stage["c"].dot(array(sol_first_stage[0:self.nv_first_stage])) + \
                    sum(obj_second_stage), UB_index[iter - 1])
                Gap_index[iter] = abs(UB_index[iter] - LB_index[iter - 1]) / UB_index[iter]
                # break
            (sol_first_stage, obj_first_stage, success) = milp(master_problem["c"], Aeq=master_problem["Aeq"],
                                                               beq=master_problem["beq"],
                                                               A=master_problem["A"], b=master_problem["b"],
                                                               vtypes=master_problem["vtypes"],
                                                               xmax=master_problem["ub"], xmin=master_problem["lb"])


            assert success == 1, "The master problem is infeasible!"
            LB_index[iter] = obj_first_stage
            # sol_first_stage_checked = self.first_stage_solution_validation(sol=sol_first_stage[0:self.nv_first_stage])
            # sol_second_stage_checked = {}
            # for i in range(ns):
            #     sol_second_stage_checked[i] = self.second_stage_solution_validation(sol_second_stage[i])

            # print("The derivation is {0}".format((array(sol_first_stage[0:self.nv_first_stage]) - sol_first_stage_0).dot(
            #         (array(sol_first_stage[0:self.nv_first_stage]) - sol_first_stage_0))))
            print("The lower boundary is {0}".format(LB_index[iter]))
            print("The upper boundary is {0}".format(UB_index[iter]))
            print("The gap is {0}".format(Gap_index[iter]))
            if iter % 100 == 0:
                plt.plot(LB_index)
                # plt.plot(UB_index)
                plt.show()
                plt.close()

        # Verify the solutions!
        # 5.1) Second-stage solution
        problem_second_stage = {}
        problem_second_stage_relaxed = {}
        sub_problem = []
        for i in range(ns):
            (problem_second_stage[i], problem_second_stage_relaxed[i]) = \
                self.second_stage_problem(x=sol_first_stage[0:self.nv_first_stage], model=model_second_stage[i])
            # print(max(problem_second_stage[i]["b"]))
            sub_problem.append([problem_second_stage[i], problem_second_stage_relaxed[i]])

        sol_second_stage = {}
        sol = [0] * ns
        for i in range(ns):
            sol[i] = sub_problem_solving(sub_problem[i])
            sol_second_stage[i] = sol[i][0]

        sol_first_stage = self.first_stage_solution_validation(sol=sol_first_stage[0:self.nv_first_stage])
        # 5.2) Second-stage solution
        sol_second_stage_checked = {}
        for i in range(ns):
            sol_second_stage_checked[i] = self.second_stage_solution_validation(sol_second_stage[i])
        # 6) Store the solutions into database!
        db_management = DataBaseManagement()
        db_management.create_table(table_name="distribution_networks", nl=self.nl, nb=self.nb, ng=self.ng)
        db_management.create_table(table_name="micro_grids", nmg=self.nmg)
        db_management.create_table(table_name="mobile_energy_storage_systems", nmg=self.nmg)
        db_management.create_table(table_name="first_stage_solutions", nmg=self.nmg, ng=self.ng)
        db_management.create_table(table_name="fisrt_stage_mess", nmg=self.nmg)

        for t in range(T):
            db_management.insert_data_first_stage(table_name="first_stage_solutions", time=t, ng=self.ng,
                                                  nmg=self.nmg,
                                                  pg=sol_first_stage["pg"][:, t].tolist(),
                                                  rg=sol_first_stage["rg"][:, t].tolist(),
                                                  pg_mg=sol_first_stage["pg_mg"][:, t].tolist(),
                                                  rg_mg=sol_first_stage["rg_mg"][:, t].tolist(),
                                                  pess_ch=sol_first_stage["pess_ch"][:, t].tolist(),
                                                  pess_dc=sol_first_stage["pess_dc"][:, t].tolist(),
                                                  ress=sol_first_stage["ress"][:, t].tolist(),
                                                  ess=sol_first_stage["eess"][:, t].tolist(),
                                                  iess=sol_first_stage["iess"][:, t].tolist())
        for i in range(nmes):
            for t in range(T):
                db_management.insert_data_first_stage_mess(table_name="fisrt_stage_mess", nmg=self.nmg, time=t,
                                                           mess=i,
                                                           imess=sol_first_stage["MESS"][i]["idc"][:, t].tolist(),
                                                           rmess=sol_first_stage["MESS"][i]["rmess"][:, t].tolist(),
                                                           pmess_ch=
                                                           sol_first_stage["MESS"][i]["pmess_ch"][:, t].tolist(),
                                                           pmess_dc=
                                                           sol_first_stage["MESS"][i]["pmess_dc"][:, t].tolist(),
                                                           mess_f_stop=sol_first_stage["MESS"][i]["VRP"][t + 1][0],
                                                           mess_t_stop=sol_first_stage["MESS"][i]["VRP"][t + 1][1])

        for i in range(ns):
            for t in range(T):
                db_management.insert_data_ds(table_name="distribution_networks", nl=self.nl, nb=self.nb, ng=self.ng,
                                             scenario=i, time=t,
                                             pij=sol_second_stage_checked[i]["DS"]["pij"][:, t].tolist(),
                                             qij=sol_second_stage_checked[i]["DS"]["qij"][:, t].tolist(),
                                             lij=sol_second_stage_checked[i]["DS"]["lij"][:, t].tolist(),
                                             vi=sol_second_stage_checked[i]["DS"]["vi"][:, t].tolist(),
                                             pg=sol_second_stage_checked[i]["DS"]["pg"][:, t].tolist(),
                                             qg=sol_second_stage_checked[i]["DS"]["qg"][:, t].tolist(), )
        for i in range(ns):
            for j in range(nmg):
                for t in range(T):
                    db_management.insert_data_mg(table_name="micro_grids", scenario=i, time=t, mg=j,
                                                 pg=sol_second_stage_checked[i]["MG"]["pg"][j, t],
                                                 qg=sol_second_stage_checked[i]["MG"]["qg"][j, t],
                                                 pug=sol_second_stage_checked[i]["MG"]["pug"][j, t],
                                                 qug=sol_second_stage_checked[i]["MG"]["qug"][j, t],
                                                 pbic_ac2dc=sol_second_stage_checked[i]["MG"]["pbic_ac2dc"][j, t],
                                                 pbic_dc2ac=sol_second_stage_checked[i]["MG"]["pbic_dc2ac"][j, t],
                                                 qbic=sol_second_stage_checked[i]["MG"]["qbic"][j, t],
                                                 pess_ch=sol_second_stage_checked[i]["MG"]["pess_ch"][j, t],
                                                 pess_dc=sol_second_stage_checked[i]["MG"]["pess_dc"][j, t],
                                                 eess=sol_second_stage_checked[i]["MG"]["eess"][j, t],
                                                 pmess=sol_second_stage_checked[i]["MG"]["pmess"][j, t],
                                                 ppv=sol_second_stage_checked[i]["MG"]["ppv"][j, t])
        for i in range(ns):
            for j in range(nmes):
                for t in range(T):
                    db_management.insert_data_mess(table_name="mobile_energy_storage_systems", scenario=i, time=t,
                                                   mess=j, nmg=self.nmg,
                                                   pmess_dc=
                                                   sol_second_stage_checked[i]["MESS"][j]["pmess_dc"][:,
                                                   t].tolist(),
                                                   pmess_ch=
                                                   sol_second_stage_checked[i]["MESS"][j]["pmess_ch"][:,
                                                   t].tolist(),
                                                   emess=sol_second_stage_checked[i]["MESS"][j]["emess"][0, t])
        # 7) Cross validation of the first-stage and second-stage decision variables
        tess_check = {}
        for i in range(ns):
            tess_temp = {}
            for j in range(nmes):
                tess_temp[j] = sol_second_stage_checked[i]["MESS"][j]["pmess_dc"] - \
                               sol_second_stage_checked[i]["MESS"][j]["pmess_ch"] - \
                               sol_first_stage["MESS"][j]["pmess_dc"] + \
                               sol_first_stage["MESS"][j]["pmess_ch"] - \
                               sol_first_stage["MESS"][j]["rmess"]
                tess_temp[j + nmes] = sol_second_stage_checked[i]["MESS"][j]["pmess_ch"] - \
                                      sol_second_stage_checked[i]["MESS"][j]["pmess_dc"] - \
                                      sol_first_stage["MESS"][j]["pmess_ch"] + \
                                      sol_first_stage["MESS"][j]["pmess_dc"] - \
                                      sol_first_stage["MESS"][j]["rmess"]
            tess_check[i] = tess_temp

        return sol_first_stage, sol_second_stage_checked

    def second_stage_problem(self, x, model):
        """
        Formulate the second-stage mathematical models
        :param x:
        :param model:
        :return:
        """
        primal_problem = deepcopy(model)
        if primal_problem["A"] is not None:
            primal_problem["A"] = vstack([model["Ws"], primal_problem["A"]])
            primal_problem["b"] = concatenate([model["hs"] - model["Ts"] * array(x), primal_problem["b"]])
        else:
            primal_problem["A"] = model["Ws"]
            primal_problem["b"] = model["hs"] - model["Ts"] * array(x)
        # Formulate the relaxed second stage problem
        try:
            primal_problem["A"] = primal_problem["A"].tolil()
            primal_problem["Aeq"] = primal_problem["Aeq"].tolil()
        except:
            pass

        # Formulate the infeasible problem,
        nx = len(primal_problem["c"])
        primal_problem_relaxed = deepcopy(primal_problem)
        primal_problem_relaxed["lb"] = concatenate([primal_problem_relaxed["lb"], zeros(1)])
        primal_problem_relaxed["ub"] = concatenate([primal_problem_relaxed["ub"], ones(1) * self.bigM * 1e3])
        primal_problem_relaxed["Aeq"] = hstack(
            [primal_problem_relaxed["Aeq"], zeros((primal_problem_relaxed["Aeq"].shape[0], 1))]).tolil()
        primal_problem_relaxed["A"] = hstack(
            [primal_problem_relaxed["A"], -ones((primal_problem_relaxed["A"].shape[0], 1))]).tolil()
        primal_problem_relaxed["c"] = concatenate([primal_problem["c"], self.bigM * ones(1)])
        # primal_problem_relaxed["c"] = concatenate([zeros(nx), ones(1)])
        primal_problem_relaxed["q"] = zeros(nx + 1)
        primal_problem["q"] = zeros(nx)
        # nx = len(primal_problem["c"])
        # primal_problem_relaxed = deepcopy(primal_problem)
        # primal_problem_relaxed["lb"] = concatenate([primal_problem_relaxed["lb"], zeros(1)])
        # primal_problem_relaxed["ub"] = concatenate([primal_problem_relaxed["ub"], ones(1) * self.bigM])
        # primal_problem_relaxed["A"] = vstack(
        #     [primal_problem_relaxed["A"], primal_problem_relaxed["Aeq"], -primal_problem_relaxed["Aeq"]])
        # primal_problem_relaxed["b"] = concatenate(
        #     [primal_problem_relaxed["b"], primal_problem_relaxed["beq"], -primal_problem_relaxed["beq"]])
        #
        # primal_problem_relaxed["A"] = hstack(
        #     [primal_problem_relaxed["A"], -ones((primal_problem_relaxed["A"].shape[0], 1))]).tolil()
        # primal_problem_relaxed["Aeq"] = None
        # primal_problem_relaxed["beq"] = None
        # primal_problem_relaxed["c"] = concatenate([zeros(nx), ones(1)])
        # primal_problem_relaxed["q"] = zeros(nx + 1)
        # primal_problem["q"] = zeros(nx)



        # nx_relaxed = primal_problem_relaxed["A"].shape[0]
        # primal_problem_relaxed["lb"] = concatenate([primal_problem_relaxed["lb"], zeros(nx_relaxed)])
        # primal_problem_relaxed["ub"] = concatenate([primal_problem_relaxed["ub"], ones(nx_relaxed) * 1e3])
        # primal_problem_relaxed["Aeq"] = hstack(
        #     [primal_problem_relaxed["Aeq"], zeros((primal_problem_relaxed["Aeq"].shape[0], nx_relaxed))]).tolil()
        # primal_problem_relaxed["A"] = hstack([primal_problem_relaxed["A"], -eye(nx_relaxed)]).tolil()
        # primal_problem_relaxed["c"] = concatenate([zeros(nx), ones(nx_relaxed)])
        # primal_problem_relaxed["q"] = zeros(nx + nx_relaxed)

        return primal_problem, primal_problem_relaxed

    def first_stage_problem_formualtion(self, pns, mgs, mess, tns):
        """
        Problem formulation for the first stage optimization,
        Decision variables include, DGs within power networks, DGs within MGs, EESs within MGs and TESSs
        :param power_networks: Parameters for the power networks
        :param micro_grids: Parameters for the microgrids
        :param tess: Parameters for the mobile energy storage systems
        :param traffic_networks: Parameters for the transportation networks
        :return: Formulated first-stage problem
        """
        T = self.T  # Time slots
        nmg = self.nmg  # Number of mgs
        nmes = self.nmes  # Number of tess

        mpc = ext2int(pns)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nb = shape(mpc["bus"])[0]
        self.nb = nb
        self.ng = ng
        # Obtain the initial status, start-up and shut down of generators
        Ig0 = gen[:, -1].astype(int)
        MIN_DOWN = gen[:, -2].astype(int)
        MIN_UP = gen[:, -3].astype(int)

        alpha_l = zeros(ng)
        beta_l = zeros(ng)
        Ig_l = ones(ng)
        pg_l = zeros(ng)  # Boundary for DGs within distribution networks
        rg_l = zeros(ng)

        alpha_u = ones(ng)
        beta_u = ones(ng)
        Ig_u = ones(ng)
        pg_u = gen[:, PMAX] / baseMVA
        rg_u = gen[:, PMAX] / baseMVA
        c_alpha = gencost[:, 0]
        c_beta = gencost[:, 1]
        c_ig = gencost[:, 6]
        cg = gencost[:, 5] * baseMVA
        cr = zeros(ng)

        pg_mg_l = zeros(nmg)  # Boundary for DGs within MGs
        rg_mg_l = zeros(nmg)
        pg_mg_u = zeros(nmg)
        rg_mg_u = zeros(nmg)
        cg_mg = zeros(nmg)
        cr_mg = zeros(nmg)
        for i in range(nmg):
            pg_mg_l[i] = mgs[i]["DG"]["PMIN"]
            pg_mg_u[i] = mgs[i]["DG"]["PMAX"]
            rg_mg_u[i] = mgs[i]["DG"]["PMAX"]
            cg_mg[i] = mgs[i]["DG"]["COST_B"]

        pes_ch_l = zeros(nmg)  # Lower boundary for ESSs within MGs
        pes_dc_l = zeros(nmg)
        ees_l = zeros(nmg)
        res_l = zeros(nmg)
        ies_l = zeros(nmg)

        pes_ch_u = zeros(nmg)  # Upper boundary for ESSs within MGs
        pes_dc_u = zeros(nmg)
        ees_u = zeros(nmg)
        res_u = zeros(nmg)
        ies_u = ones(nmg)

        ces_ch = zeros(nmg)  # Cost boundary for ESSs within MGs
        ces_dc = zeros(nmg)
        ces_r = zeros(nmg)
        ces = zeros(nmg)
        ces_i = zeros(nmg)

        for i in range(nmg):
            pes_ch_u[i] = mgs[i]["ESS"]["PCH_MAX"]
            pes_dc_u[i] = mgs[i]["ESS"]["PDC_MAX"] + mgs[i]["ESS"]["PCH_MAX"]
            res_u[i] = mgs[i]["ESS"]["PCH_MAX"]
            ees_l[i] = mgs[i]["ESS"]["EMIN"]
            ees_u[i] = mgs[i]["ESS"]["EMAX"]

        _nv_first_stage = ng * 5 + nmg * 2 + nmg * 5
        nv_first_stage = _nv_first_stage * T
        # Formulate the boundaries
        lb = concatenate(
            [tile(concatenate(
                [alpha_l, beta_l, Ig_l, pg_l, rg_l, pg_mg_l, rg_mg_l, pes_ch_l, pes_dc_l, res_l, ees_l, ies_l]), T)])
        ub = concatenate(
            [tile(concatenate(
                [alpha_u, beta_u, Ig_u, pg_u, rg_u, pg_mg_u, rg_mg_u, pes_ch_u, pes_dc_u, res_u, ees_u, ies_u]), T)])
        # Objective value
        c = concatenate(
            [tile(concatenate([c_alpha, c_beta, c_ig, cg, cr, cg_mg, cr_mg, ces_ch, ces_dc, ces, ces_r, ces_i]), T)])
        for t in range(T):
            for j in range(ng):
                c[t * _nv_first_stage + ng * 3 + j] = Price_wholesale[0, t] * 1000 *baseMVA
        # Variable types
        vtypes = (["b"] * ng * 3 + ["c"] * (ng * 2 + nmg * 2 + nmg * 4) + ["b"] * nmg) * T
        ## Constraint sets
        # 1) Pg+Rg<=PguIg
        A = lil_matrix((ng * T, nv_first_stage))
        b = zeros(ng * T)
        for t in range(T):
            for j in range(ng):
                A[t * ng + j, t * _nv_first_stage + ng * 3 + j] = 1
                A[t * ng + j, t * _nv_first_stage + ng * 4 + j] = 1
                A[t * ng + j, t * _nv_first_stage + ng * 2 + j] = -pg_u[j]
        # 2) Pg-Rg>=IgPgl
        A_temp = lil_matrix((ng * T, nv_first_stage))
        b_temp = zeros(ng * T)
        for t in range(T):
            for j in range(ng):
                A_temp[t * ng + j, t * _nv_first_stage + ng * 3 + j] = -1
                A_temp[t * ng + j, t * _nv_first_stage + ng * 4 + j] = 1
                A_temp[t * ng + j, t * _nv_first_stage + j] = pg_l[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 3) Start-up and shut-down constraints of DGs
        UP_LIMIT = zeros(ng).astype(int)
        DOWN_LIMIT = zeros(ng).astype(int)
        for i in range(ng):
            UP_LIMIT[i] = T - MIN_UP[i]
            DOWN_LIMIT[i] = T - MIN_DOWN[i]
        # 3.1) Up limit
        A_temp = lil_matrix((sum(UP_LIMIT), nv_first_stage))
        b_temp = zeros(sum(UP_LIMIT))
        for i in range(ng):
            for t in range(MIN_UP[i], T):
                for k in range(t - MIN_UP[i], t):
                    A_temp[sum(UP_LIMIT[0:i]) + t - MIN_UP[i], k * _nv_first_stage + i] = 1
                A_temp[sum(UP_LIMIT[0:i]) + t - MIN_UP[i], t * _nv_first_stage + ng * 2 + i] = -1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # # 3.2) Down limit
        A_temp = lil_matrix((sum(DOWN_LIMIT), nv_first_stage))
        b_temp = ones(sum(DOWN_LIMIT))
        for i in range(ng):
            for t in range(MIN_DOWN[i], T):
                for k in range(t - MIN_DOWN[i], t):
                    A_temp[sum(DOWN_LIMIT[0:i]) + t - MIN_DOWN[i], k * _nv_first_stage + ng + i] = 1
                A_temp[sum(DOWN_LIMIT[0:i]) + t - MIN_DOWN[i], t * _nv_first_stage + ng * 2 + i] = 1
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 4) Status transformation of each unit
        Aeq = lil_matrix((T * ng, nv_first_stage))
        beq = zeros(T * ng)
        for i in range(ng):
            for t in range(T):
                Aeq[i * T + t, t * _nv_first_stage + i] = 1
                Aeq[i * T + t, t * _nv_first_stage + ng + i] = -1
                Aeq[i * T + t, t * _nv_first_stage + ng * 2 + i] = -1
                if t != 0:
                    Aeq[i * T + t, (t - 1) * _nv_first_stage + ng * 2 + i] = 1
                else:
                    beq[i * T + t] = -Ig0[i]

        # 3) Pg_mg+Rg_mg<=Pg_mg_u
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + j] = 1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg + j] = 1
                b_temp[t * nmg + j] = pg_mg_u[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 4) Pg_mg-Rg_mg<=Pg_mg_l
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + j] = -1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg + j] = 1
                b_temp[t * nmg + j] = pg_mg_l[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 5) Pess_dc-Pess_ch+Ress<=Pess_dc_max
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + j] = -1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg + j] = 1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg * 2 + j] = 1
                b_temp[t * nmg + j] = pes_dc_u[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 6) Pess_ch-Pess_dc+Ress<=Pess_ch_max
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage +ng * 5 + nmg * 2 + j] = 1
                A_temp[t * nmg + j, t * _nv_first_stage +ng * 5 + nmg * 2 + nmg + j] = -1
                A_temp[t * nmg + j, t * _nv_first_stage +ng * 5 + nmg * 2 + nmg * 2 + j] = 1
                b_temp[t * nmg + j] = pes_ch_u[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 7) Energy storage balance equation
        Aeq_temp = lil_matrix((T * nmg, nv_first_stage))
        beq_temp = zeros(T * nmg)
        for t in range(T):
            for j in range(nmg):
                Aeq_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg * 3 + j] = 1
                Aeq_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + j] = -mgs[j]["ESS"]["EFF_CH"]
                Aeq_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg + j] = 1 / mgs[j]["ESS"]["EFF_DC"]
                if t == 0:
                    beq_temp[t * nmg + j] = mgs[j]["ESS"]["E0"]
                else:
                    Aeq_temp[t * nmg + j, (t - 1) * _nv_first_stage + ng * 5 + nmg * 2 + nmg * 3 + j] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 8) Pess_ch<=I*Pess_ch_max
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + j] = 1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg * 4 + j] = -pes_ch_u[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])
        # 9) Pess_dc<=(1-I)*Pess_dc_max
        A_temp = lil_matrix((nmg * T, nv_first_stage))
        b_temp = zeros(nmg * T)
        for t in range(T):
            for j in range(nmg):
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg + j] = 1
                A_temp[t * nmg + j, t * _nv_first_stage + ng * 5 + nmg * 2 + nmg * 4 + j] = pes_dc_u[j]
                b_temp[t * nmg + j] = pes_dc_u[j]
        A = vstack([A, A_temp])
        b = concatenate([b, b_temp])

        # 2) Transportation energy storage systems problem
        model_mess = {}
        for i in range(nmes):
            model_mess[i] = self.problem_formulation_tess(mess=mess[i], tns=tns)
        # 3) Merge the DGs, ESSs and TESSs
        neq = Aeq.shape[0]
        nineq = A.shape[0]

        nV_index = zeros(nmes + 1).astype(int)
        neq_index = zeros(nmes + 1).astype(int)
        nineq_index = zeros(nmes + 1).astype(int)
        nV_index[0] = nv_first_stage
        neq_index[0] = neq
        nineq_index[0] = nineq

        for i in range(nmes):
            nV_index[i + 1] = nV_index[i] + len(model_mess[i]["c"])
            neq_index[i + 1] = neq_index[i] + model_mess[i]["Aeq"].shape[0]
            nineq_index[i + 1] = nineq_index[i] + model_mess[i]["A"].shape[0]
            neq += model_mess[i]["Aeq"].shape[0]
            nineq += model_mess[i]["A"].shape[0]
            # Merge the objective function, boundaries, types and rhs
            c = concatenate([c, model_mess[i]["c"]])
            lb = concatenate([lb, model_mess[i]["lb"]])
            ub = concatenate([ub, model_mess[i]["ub"]])
            vtypes += model_mess[i]["vtypes"]
            beq = concatenate([beq, model_mess[i]["beq"]])
            b = concatenate([b, model_mess[i]["b"]])

        A_full = lil_matrix((nineq_index[-1], nV_index[-1]))
        Aeq_full = lil_matrix((neq_index[-1], nV_index[-1]))

        if Aeq is not None: Aeq_full[0:int(neq_index[0]), 0:int(nV_index[0])] = Aeq
        if A is not None: A_full[0:int(nineq_index[0]), 0:int(nV_index[0])] = A

        for i in range(nmes):
            Aeq_full[neq_index[i]:neq_index[i + 1], nV_index[i]:nV_index[i + 1]] = model_mess[i]["Aeq"]
            A_full[nineq_index[i]:nineq_index[i + 1], nV_index[i]:nV_index[i + 1]] = model_mess[i]["A"]

        self.nv_first_stage = nV_index[-1]  # The number of first stage decision variables
        self._nv_first_stage = _nv_first_stage
        model_first_stage = {"c": c,
                             "lb": lb,
                             "ub": ub,
                             "vtypes": vtypes,
                             "A": A_full,
                             "b": b,
                             "Aeq": Aeq_full,
                             "beq": beq, }

        return model_first_stage

    def first_stage_solution_validation(self, sol):
        """
        Validation of the first-stage solution
        :param sol: The first stage solution
        :return: the first stage solution
        """
        T = self.T
        ng = self.ng
        nmg = self.nmg
        nmes = self.nmes
        # Set-points of DGs within DSs, MGs and ESSs
        _nv_first_stage = self._nv_first_stage
        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        Ig = zeros((ng, T))
        Pg = zeros((ng, T))
        Rg = zeros((ng, T))
        Pg_mg = zeros((nmg, T))
        Rg_mg = zeros((nmg, T))
        Pess_dc = zeros((nmg, T))
        Pess_ch = zeros((nmg, T))
        Ress = zeros((nmg, T))
        Eess = zeros((nmg, T))
        Iess = zeros((nmg, T))
        for i in range(T):
            alpha[:, i] = sol[_nv_first_stage * i:_nv_first_stage * i + ng]
            beta[:, i] = sol[_nv_first_stage * i + ng:_nv_first_stage * i + ng * 2]
            Ig[:, i] = sol[_nv_first_stage * i + ng * 2:_nv_first_stage * i + ng * 3]
            Pg[:, i] = sol[_nv_first_stage * i + ng * 3:_nv_first_stage * i + ng * 4]
            Rg[:, i] = sol[_nv_first_stage * i + ng * 4:_nv_first_stage * i + ng * 5]
            Pg_mg[:, i] = sol[_nv_first_stage * i + ng * 5:_nv_first_stage * i + ng * 5 + nmg]
            Rg_mg[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg:_nv_first_stage * i + ng * 5 + nmg * 2]
            Pess_ch[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg * 2:_nv_first_stage * i + ng * 5 + nmg * 3]
            Pess_dc[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg * 3:_nv_first_stage * i + ng * 5 + nmg * 4]
            Ress[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg * 4:_nv_first_stage * i + ng * 5 + nmg * 5]
            Eess[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg * 5:_nv_first_stage * i + ng * 5 + nmg * 6]
            Iess[:, i] = sol[_nv_first_stage * i + ng * 5 + nmg * 6:_nv_first_stage * i + ng * 5 + nmg * 7]

        # Set-points and scheduling of mobile energy storage systems
        nv_tra = self.nv_tra
        nl_tra = self.nl_tra
        n_stops = self.n_stops
        nb_tra_ele = self.nb_tra_ele
        sol_ev = {}
        for i in range(nmes):
            ev_temp = {}
            ev_temp["VRP"] = []
            for t in range(nl_tra):
                if sol[_nv_first_stage * T + nv_tra * i + t] > 0:  # obtain the solution for vrp
                    if self.connection_matrix[t, TIME] > 0:
                        for j in range(int(self.connection_matrix[t, TIME])):
                            ev_temp["VRP"].append(((self.connection_matrix[t, F_BUS] - 1) % nmg,
                                                   (self.connection_matrix[t, T_BUS] - 1) % nmg))
                    else:
                        ev_temp["VRP"].append(((self.connection_matrix[t, F_BUS] - 1) % nmg,
                                               (self.connection_matrix[t, T_BUS] - 1) % nmg))

            ev_temp["idc"] = zeros((nmg, T))
            ev_temp["pmess_dc"] = zeros((nmg, T))
            ev_temp["pmess_ch"] = zeros((nmg, T))
            ev_temp["rmess"] = zeros((nmg, T))
            for t in range(T):
                for j in range(nmg):
                    ev_temp["idc"][j, t] = sol[_nv_first_stage * T + nv_tra * i + nl_tra + nb_tra_ele * t + j]
                    ev_temp["pmess_dc"][j, t] = \
                        sol[_nv_first_stage * T + nv_tra * i + nl_tra + n_stops + nb_tra_ele * t + j]
                    ev_temp["pmess_ch"][j, t] = \
                        sol[_nv_first_stage * T + nv_tra * i + nl_tra + n_stops * 2 + nb_tra_ele * t + j]
                    ev_temp["rmess"][j, t] = \
                        sol[_nv_first_stage * T + nv_tra * i + nl_tra + n_stops * 3 + nb_tra_ele * t + j]
            sol_ev[i] = ev_temp

        sol_first_stage = {"alpha": alpha,
                           "beta": beta,
                           "ig": Ig,
                           "rg": Rg,
                           "pg": Pg,
                           "pg_mg": Pg_mg,
                           "rg_mg": Rg_mg,
                           "pess_ch": Pess_ch,
                           "pess_dc": Pess_dc,
                           "ress": Ress,
                           "eess": Eess,
                           "iess": Iess,
                           "MESS": sol_ev,
                           }
        return sol_first_stage

    def second_stage_problem_formualtion_gdb(self, pns, mgs, mess, tns, profile, index=0, weight=1):
        """
        Second-stage problem formulation, the decision variables includes DGs within power networks, DGs within MGs, EESs within MGs and TESSs and other systems' information
        :param power_networks:
        :param micro_grids:
        :param tess:
        :param traffic_networks:
        :return: The second stage problems as list, including coupling constraints, and other constraint set
        """
        # I) Formulate the problem for distribution systems operator
        T = self.T
        mpc = ext2int(pns)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nd = sum(bus[:, PD] > 0)
        nmg = self.nmg
        nmes = self.nmes

        self.nl = nl
        self.nb = nb
        self.ng = ng
        self.nd = nd

        m = zeros(nmg)  ## list of integration index
        pmg_l = zeros(nmg)  ## list of lower boundary
        pmg_u = zeros(nmg)  ## list of upper boundary
        qmg_l = zeros(nmg)  ## list of lower boundary
        qmg_u = zeros(nmg)  ## list of upper boundary
        for i in range(nmg):
            m[i] = mgs[i]["BUS"]
            pmg_l[i] = mgs[i]["UG"]["PMIN"] / 1000 / baseMVA
            pmg_u[i] = mgs[i]["UG"]["PMAX"] / 1000 / baseMVA
            qmg_l[i] = mgs[i]["UG"]["QMIN"] / 1000 / baseMVA
            qmg_u[i] = mgs[i]["UG"]["QMAX"] / 1000 / baseMVA

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        d = bus[bus[:, PD] > 0, BUS_I].astype(int)  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        self.f = f  ## record from bus for each branch

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cd = sparse((ones(nd), (d, range(nd))), (nb, nd))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cmg = sparse((ones(nmg), (m, range(nmg))), (nb, nmg))

        Branch_R = branch[:, BR_R]
        Branch_X = branch[:, BR_X]
        Cf = Cf.T
        Ct = Ct.T
        # Obtain the boundary information
        slmax = branch[:, RATE_A] / baseMVA

        pij_l = -slmax
        qij_l = -slmax
        lij_l = zeros(nl)
        vm_l = bus[:, VMIN] ** 2
        pg_l = zeros(ng)
        qg_l = gen[:, QMIN] / baseMVA
        pd_l = zeros(nd)

        pij_u = slmax
        qij_u = slmax
        lij_u = slmax
        vm_u = bus[:, VMAX] ** 2
        pg_u = gen[:, PMAX] / baseMVA
        qg_u = gen[:, QMAX] / baseMVA
        pd_u = bus[d, PD] / baseMVA

        _nv_second_stage = int(3 * nl + nb + 2 * ng + 2 * nmg + nd)
        self._nv_second_stage = _nv_second_stage  # Number of decision variable within each time slot

        lb = concatenate([tile(concatenate([pij_l, qij_l, lij_l, vm_l, pg_l, qg_l, pmg_l, qmg_l, pd_l]), T)])
        ub = concatenate([tile(concatenate([pij_u, qij_u, lij_u, vm_u, pg_u, qg_u, pmg_u, qmg_u, pd_u]), T)])

        vtypes = ["c"] * _nv_second_stage * T
        nv_ds = _nv_second_stage * T  # Number of total decision variables
        self.nv_ds = nv_ds
        for i in range(T):
            temp = profile[i * nb:(i + 1) * nb] / baseMVA
            ub[i*_nv_second_stage+3*nl+nb+2*ng+2*nmg: (i+1)*_nv_second_stage] = temp[d]*2
        # Add system level constraints
        # 1) Active power balance
        Aeq_p = lil_matrix((nb * T, nv_ds))
        beq_p = zeros(nb * T)
        for i in range(T):
            Aeq_p[i * nb:(i + 1) * nb, i * _nv_second_stage: (i + 1) * _nv_second_stage] = \
                hstack([Ct - Cf, zeros((nb, nl)),
                        -diag(Ct * Branch_R) * Ct,
                        zeros((nb, nb)), Cg,
                        zeros((nb, ng)), -Cmg,
                        zeros((nb, nmg)), Cd])

            beq_p[i * nb:(i + 1) * nb] = profile[i * nb:(i + 1) * nb] / baseMVA

        # 2) Reactive power balance
        Aeq_q = lil_matrix((nb * T, nv_ds))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, i * _nv_second_stage: (i + 1) * _nv_second_stage] = \
                hstack([zeros((nb, nl)), Ct - Cf,
                        -diag(Ct * Branch_X) * Ct,
                        zeros((nb, nb)),
                        zeros((nb, ng)), Cg,
                        zeros((nb, nmg)), -Cmg, Cd.dot(diag(bus[d, QD] / bus[d, PD]))])
            for j in range(nb):
                if bus[j, PD] > 0:
                    beq_q[i * nb:(i + 1) * nb] = profile[i * nb + j] / bus[j, PD] * bus[j, QD] / baseMVA
        # 3) KVL equation
        Aeq_kvl = lil_matrix((nl * T, nv_ds))
        beq_kvl = zeros(nl * T)

        for i in range(T):
            Aeq_kvl[i * nl:(i + 1) * nl, i * _nv_second_stage: i * _nv_second_stage + nl] = -2 * diag(Branch_R)
            Aeq_kvl[i * nl:(i + 1) * nl, i * _nv_second_stage + nl: i * _nv_second_stage + 2 * nl] = -2 * diag(Branch_X)
            Aeq_kvl[i * nl:(i + 1) * nl, i * _nv_second_stage + 2 * nl: i * _nv_second_stage + 3 * nl] = diag(
                Branch_R ** 2) + diag(Branch_X ** 2)
            Aeq_kvl[i * nl:(i + 1) * nl, i * _nv_second_stage + 3 * nl:i * _nv_second_stage + 3 * nl + nb] = (
                    Cf.T - Ct.T).toarray()

        Aeq = vstack([Aeq_p, Aeq_q, Aeq_kvl])
        beq = concatenate([beq_p, beq_q, beq_kvl])

        c = zeros(nv_ds)
        q = zeros(nv_ds)
        c0 = 0
        for t in range(T):
            for i in range(ng):
                c[t * _nv_second_stage + i + 3 * nl + nb] = gencost[i, 5] * baseMVA
                q[t * _nv_second_stage + i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
                c0 += gencost[i, 6]
            for i in range(nd):
                c[t * _nv_second_stage + i + 3 * nl + nb + 2 * ng + 2 * nmg] = Voll * baseMVA  # The load shedding cost
        # Coupling constraints between the distribution systems and micro_grids
        Ax2y = lil_matrix((2 * nmg * T, nv_ds))  # connection matrix with the microgrids
        for i in range(T):
            for j in range(nmg):
                # Active power
                Ax2y[i * nmg + j, i * _nv_second_stage + 3 * nl + nb + 2 * ng + j] = 1000 * baseMVA
                # Reactive power
                Ax2y[nmg * T + i * nmg + j, i * _nv_second_stage + 3 * nl + nb + 2 * ng + nmg + j] = 1000 * baseMVA

        # II) Formulate the problem for microgrids
        model_microgrids = {}
        for i in range(nmg):
            model_microgrids[i] = self.problem_formulation_microgrid(mg=mgs[i], mess=mess)
        # II.A) Combine the distribution system operation problem and microgrid systems
        if Aeq is not None:
            neq_ds = Aeq.shape[0]
        else:
            neq_ds = 0

        nVariables = int(nv_ds)
        neq = int(neq_ds)

        nv_index = zeros(nmg + 1).astype(int)
        neq_index = zeros(nmg + 1).astype(int)
        nv_index[0] = nv_ds
        neq_index[0] = int(neq_ds)
        for i in range(nmg):
            nv_index[i + 1] = nv_index[i] + len(model_microgrids[i]["c"])
            neq_index[i + 1] = neq_index[i] + model_microgrids[i]["Aeq"].shape[0]
            nVariables += len(model_microgrids[i]["c"])
            neq += int(model_microgrids[i]["Aeq"].shape[0])

        Aeq_full = lil_matrix((int(neq_index[-1]), int(nv_index[-1])))
        Aeq_full[0:neq_ds, 0:nv_ds] = Aeq
        for i in range(nmg):
            lb = concatenate([lb, model_microgrids[i]["lb"]])
            ub = concatenate([ub, model_microgrids[i]["ub"]])
            c = concatenate([c, model_microgrids[i]["c"]])
            q = concatenate([q, model_microgrids[i]["q"]])
            vtypes += model_microgrids[i]["vtypes"]
            beq = concatenate([beq, model_microgrids[i]["beq"]])
            Aeq_full[neq_index[i]:neq_index[i + 1], nv_index[i]:nv_index[i + 1]] = model_microgrids[i]["Aeq"]

        # Add coupling constraints, between the microgrids and distribution networks
        Ay2x = lil_matrix((2 * nmg * T, nv_index[-1] - nv_index[0]))
        for i in range(T):
            for j in range(nmg):
                Ay2x[i * nmg + j, int(nv_index[j] - nv_index[0]) + i * NX_MG + PUG] = -1
                Ay2x[nmg * T + i * nmg + j, int(nv_index[j] - nv_index[0]) + i * NX_MG + QUG] = -1

        Aeq_temp = hstack([Ax2y, Ay2x])
        beq_temp = zeros(2 * nmg * T)

        Aeq_full = vstack([Aeq_full, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # III) Formulate the optimization problem for tess in the second stage optimization
        model_tess = {}
        for i in range(nmes):
            model_tess[i] = self.problem_formulation_tess_second_stage(mess=mess[i])
        # III.1) Merge the models of mirogrids and distribution
        # Formulate the index
        nv_index_ev = zeros(1 + nmes).astype(int)
        self.nv_index_ev = nv_index_ev
        neq_index_temp = zeros(1 + nmes).astype(int)
        nv_index_ev[0] = int(Aeq_full.shape[1])
        neq_index_temp[0] = int(Aeq_full.shape[0])
        for i in range(nmes):
            nv_index_ev[i + 1] = nv_index_ev[i] + len(model_tess[i]["c"])
            neq_index_temp[i + 1] = neq_index_temp[i] + model_tess[i]["Aeq"].shape[0]

        Aeq = lil_matrix((int(neq_index_temp[-1]), int(nv_index_ev[-1])))
        Aeq[0:int(neq_index_temp[0]), 0:int(nv_index_ev[0])] = Aeq_full
        for i in range(nmes):
            lb = concatenate([lb, model_tess[i]["lb"]])
            ub = concatenate([ub, model_tess[i]["ub"]])
            c = concatenate([c, model_tess[i]["c"]])
            q = concatenate([q, model_tess[i]["q"]])
            vtypes += model_tess[i]["vtypes"]
            beq = concatenate([beq, model_tess[i]["beq"]])
            Aeq[neq_index_temp[i]:neq_index_temp[i + 1], nv_index_ev[i]:nv_index_ev[i + 1]] = model_tess[i]["Aeq"]
        # III.2) Coupling constraints between the microgrids and mobile energy storage systems
        # Additional equal constraints, nmg*T
        Aeq_temp = lil_matrix((nmg * T, nv_index_ev[-1]))
        beq_temp = zeros(nmg * T)
        for i in range(nmg):
            for t in range(T):
                Aeq_temp[i * T + t, nv_index[i] + t * NX_MG + PMESS] = 1  # TESSs injections to the MGs
                for j in range(nmes):
                    Aeq_temp[i * T + t, nv_index_ev[j] + t * self.nb_tra_ele + i] = -1  # Discharging
                    Aeq_temp[i * T + t,
                             nv_index_ev[j] + self.nb_tra_ele * T + t * self.nb_tra_ele + i] = 1  # Sort by order
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate((beq, beq_temp))
        nv_second_stage = nv_index_ev[-1]
        nv_first_stage = self.nv_first_stage
        self.nv_second_stage = nv_second_stage
        Qc = dict()
        # 4) Pij**2+Qij**2<=Vi*Iij
        for t in range(T):
            for i in range(nl):
                Qc[t * nl + i] = [
                    [int(t * _nv_second_stage + i),
                     int(t * _nv_second_stage + i + nl),
                     int(t * _nv_second_stage + i + 2 * nl),
                     int(t * _nv_second_stage + f[i] + 3 * nl)],
                    [int(t * _nv_second_stage + i),
                     int(t * _nv_second_stage + i + nl),
                     int(t * _nv_second_stage + f[i] + 3 * nl),
                     int(t * _nv_second_stage + i + 2 * nl)],
                    [1, 1, -1 / 2, -1 / 2]]
        Rc = zeros(nl * T)
        # 5) (Pbic_ac2dc+Pbic_dc2ac)**2+Qbic**2<=Sbic**2
        Rc_temp = zeros(nmg * T)
        for i in range(nmg):
            for t in range(T):
                Qc[T * nl + T * i + t] = [
                    [int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_AC2DC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_DC2AC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_AC2DC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_DC2AC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + QBIC)],
                    [int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_AC2DC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_DC2AC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_DC2AC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + PBIC_AC2DC),
                     int(nv_ds + NX_MG * T * i + NX_MG * t + QBIC)],
                    [1, 1, 1, 1, 1]]
                Rc_temp[i * T + t] = mgs[i]["BIC"]["SMAX"] ** 2
        Rc = concatenate([Rc, Rc_temp])

        ## IV. Coupling constraints between the first stage and second stage decision variables
        # pg, pg_mg, pess_mg, pess_tess
        # Ts*x+Ws*ys<=hs
        ## IV) Formulate the coupling constraints between the first-stage and second-stage problems
        # 1) -Pg -Rg + pg <= 0
        _nv_first_stage = self._nv_first_stage
        Ts = lil_matrix((ng * T, nv_first_stage))
        Ws = lil_matrix((ng * T, nv_second_stage))
        hs = zeros(ng * T)
        for i in range(T):
            for j in range(ng):
                Ts[i * ng + j, i * _nv_first_stage + ng * 3 + j] = -1
                Ts[i * ng + j, i * _nv_first_stage + ng * 4 + j] = -1
                Ws[i * ng + j, i * _nv_second_stage + 3 * nl + nb + j] = 1
        # 2) Pg-Rg - pg <= 0
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        for i in range(T):
            for j in range(ng):
                Ts_temp[i * ng + j, i * _nv_first_stage + ng * 3 + j] = 1
                Ts_temp[i * ng + j, i * _nv_first_stage + ng * 4 + j] = -1
                Ws_temp[i * ng + j, i * _nv_second_stage + 3 * nl + nb + j] = -1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        # 3) Qg <= IgQg_max
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        for i in range(T):
            for j in range(ng):
                Ts_temp[i * ng + j, i * _nv_first_stage + ng * 2 + j] = -qg_u[j]
                Ws_temp[i * ng + j, i * _nv_second_stage + 3 * nl + nb + ng + j] = 1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        # 4) Qg >= IgQg_min
        Ts_temp = lil_matrix((ng * T, nv_first_stage))
        Ws_temp = lil_matrix((ng * T, nv_second_stage))
        hs_temp = zeros(ng * T)
        for i in range(T):
            for j in range(ng):
                Ts_temp[i * ng + j, i * _nv_first_stage + ng * 2 + j] = qg_l[j]
                Ws_temp[i * ng + j, i * _nv_second_stage + 3 * nl + nb + ng + j] = -1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))

        # 5) -Pg_mg - Rg_mg + pg_mg <= 0
        Ts_temp = lil_matrix((nmg * T, nv_first_stage))
        Ws_temp = lil_matrix((nmg * T, nv_second_stage))
        hs_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + j] = -1
                Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg + j] = -1
                Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PG] = 1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        # 6) Pg_mg - Rg_mg - pg_mg <= 0
        Ts_temp = lil_matrix((nmg * T, nv_first_stage))
        Ws_temp = lil_matrix((nmg * T, nv_second_stage))
        hs_temp = zeros(nmg * T)
        for i in range(T):
            for j in range(nmg):
                Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + j] = 1
                Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg + j] = -1
                Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PG] = -1
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        # 7) pess_dc - pess_ch <= Pess_dc - Pess_ch + Ress
        # Ts_temp = lil_matrix((nmg * T, nv_first_stage))
        # Ws_temp = lil_matrix((nmg * T, nv_second_stage))
        # hs_temp = zeros(nmg * T)
        # for i in range(T):
        #     for j in range(nmg):
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 2 + j] = 1  # Charging
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 3 + j] = -1  # Dis-charging
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 4 + j] = -1  # Reserve
        #         Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PESS_CH] = -1
        #         Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PESS_DC] = 1
        # Ts = vstack((Ts, Ts_temp))
        # Ws = vstack((Ws, Ws_temp))
        # hs = concatenate((hs, hs_temp))
        # # 8) pess_ch - pess_dc <= Pess_ch - Pess_dc + Ress
        # Ts_temp = lil_matrix((nmg * T, nv_first_stage))
        # Ws_temp = lil_matrix((nmg * T, nv_second_stage))
        # hs_temp = zeros(nmg * T)
        # for i in range(T):
        #     for j in range(nmg):
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 2 + j] = -1  # Charging
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 3 + j] = 1  # Dis-charging
        #         Ts_temp[i * nmg + j, i * _nv_first_stage + ng * 5 + nmg * 4 + j] = -1  # Reserve
        #         Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PESS_CH] = 1
        #         Ws_temp[i * nmg + j, nv_index[j] + i * NX_MG + PESS_DC] = -1
        # Ts = vstack((Ts, Ts_temp))
        # Ws = vstack((Ws, Ws_temp))
        # hs = concatenate((hs, hs_temp))
        # 9) ptss_ch - ptss_dc <= Ptss_ch - Ptss_dc + Rtss
        nv_tra = self.nv_tra
        nl_tra = self.nl_tra
        n_stops = self.n_stops

        Ts_temp = lil_matrix((nmg * T * nmes, nv_first_stage))
        Ws_temp = lil_matrix((nmg * T * nmes, nv_second_stage))
        hs_temp = zeros(nmg * T * nmes)
        for i in range(nmes):
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops: _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*2] = eye(nmg * T)
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*2:_nv_first_stage * T + nv_tra * i + nl_tra + n_stops*3] = -eye(nmg * T)
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*3:_nv_first_stage * T + nv_tra * i + nl_tra + n_stops*4] = -eye(nmg * T)
            Ws_temp[i * nmg * T:(i + 1) * nmg * T, nv_index_ev[i] + nmg * T * 0: nv_index_ev[i] + nmg * T * 1] = -eye(nmg * T)
            Ws_temp[i * nmg * T:(i + 1) * nmg * T, nv_index_ev[i] + nmg * T * 1: nv_index_ev[i] + nmg * T * 2] = eye(nmg * T)
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))
        # 10) ptss_dc - ptss_ch <= Ptss_dc - Ptss_ch + Rtss
        Ts_temp = lil_matrix((nmg * T * nmes, nv_first_stage))
        Ws_temp = lil_matrix((nmg * T * nmes, nv_second_stage))
        hs_temp = zeros(nmg * T * nmes)
        for i in range(nmes):
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops: _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*2] = -eye(nmg * T)
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*2:_nv_first_stage * T + nv_tra * i + nl_tra + n_stops*3] = eye(nmg * T)
            Ts_temp[i * nmg * T:(i + 1) * nmg * T, _nv_first_stage * T + nv_tra * i + nl_tra + n_stops*3:_nv_first_stage * T + nv_tra * i + nl_tra + n_stops*4] = -eye(nmg * T)
            Ws_temp[i * nmg * T:(i + 1) * nmg * T, nv_index_ev[i] + nmg * T * 0: nv_index_ev[i] + nmg * T * 1] = eye(nmg * T)
            Ws_temp[i * nmg * T:(i + 1) * nmg * T, nv_index_ev[i] + nmg * T * 1: nv_index_ev[i] + nmg * T * 2] = -eye(nmg * T)
        Ts = vstack((Ts, Ts_temp))
        Ws = vstack((Ws, Ws_temp))
        hs = concatenate((hs, hs_temp))

        # sol = miqcp(c, q, Aeq=Aeq, beq=beq, A=None, b=None, Qc=Qc, xmin=lx, xmax=ux)
        # Modify the Aeq

        model_second_stage = {"c": c * weight,
                              "q": q * weight,
                              "lb": lb,
                              "ub": ub,
                              "vtypes": vtypes,
                              "A": None,
                              "b": None,
                              "Aeq": Aeq,
                              "beq": beq,
                              "Qc": Qc,
                              "rc": Rc,
                              "c0": c0,
                              "Ts": Ts,
                              "Ws": Ws,
                              "hs": hs}

        # model_second_stage["c"] = concatenate([model_second_stage["c"], ones(2) * Voll])
        # model_second_stage["q"] = concatenate([model_second_stage["q"], zeros(2)])
        # model_second_stage["lb"] = concatenate([model_second_stage["lb"], zeros(2)])
        # model_second_stage["ub"] = concatenate([model_second_stage["ub"], ones(2)*self.bigM])
        # model_second_stage["vtypes"] += ["c"]*2
        # model_second_stage["Aeq"] = hstack([model_second_stage["Aeq"], ones((model_second_stage["Aeq"].shape[0],1)), -ones((model_second_stage["Aeq"].shape[0],1))])
        # model_second_stage["Ws"] = hstack([model_second_stage["Ws"], zeros((model_second_stage["Ws"].shape[0], 2))])
        # qcqp(model_second_stage["c"],model_second_stage["q"],Aeq=model_second_stage["Aeq"].tolil(),beq=model_second_stage["beq"],A=model_second_stage["A"],b=model_second_stage["b"],xmin=model_second_stage["lb"],xmax=model_second_stage["ub"],Qc=model_second_stage["Qc"],rc=model_second_stage["rc"])

        return model_second_stage


    def second_stage_solution_validation(self, sol):
        """
        :param sol: The second stage solution under specific scenario
        :return: for each value
        """
        T = self.T
        nb = self.nb
        ng = self.ng
        nl = self.nl
        nmg = self.nmg
        nmes = self.nmes
        f = self.f

        # Solutions for distribution networks
        ds_sol = {}
        _nv_second_stage = self._nv_second_stage
        nv_index_ev = self.nv_index_ev
        ds_sol["pij"] = zeros((nl, T))
        ds_sol["qij"] = zeros((nl, T))
        ds_sol["lij"] = zeros((nl, T))
        ds_sol["vi"] = zeros((nb, T))
        ds_sol["pg"] = zeros((ng, T))
        ds_sol["qg"] = zeros((ng, T))
        ds_sol["pmg"] = zeros((nmg, T))
        ds_sol["qmg"] = zeros((nmg, T))
        ds_sol["gap"] = zeros((nl, T))
        for i in range(T):
            ds_sol["pij"][:, i] = sol[_nv_second_stage * i:_nv_second_stage * i + nl]
            ds_sol["qij"][:, i] = sol[_nv_second_stage * i + nl:_nv_second_stage * i + nl * 2]
            ds_sol["lij"][:, i] = sol[_nv_second_stage * i + nl * 2:_nv_second_stage * i + nl * 3]
            ds_sol["vi"][:, i] = sol[_nv_second_stage * i + nl * 3:_nv_second_stage * i + nl * 3 + nb]
            ds_sol["pg"][:, i] = sol[_nv_second_stage * i + nl * 3 + nb:_nv_second_stage * i + nl * 3 + nb + ng]
            ds_sol["qg"][:, i] = sol[_nv_second_stage * i + nl * 3 + nb + ng:
                                     _nv_second_stage * i + nl * 3 + nb + ng * 2]
            ds_sol["pmg"][:, i] = sol[_nv_second_stage * i + nl * 3 + nb + ng * 2:
                                      _nv_second_stage * i + nl * 3 + nb + ng * 2 + nmg]
            ds_sol["qmg"][:, i] = sol[_nv_second_stage * i + nl * 3 + nb + ng * 2 + nmg:
                                      _nv_second_stage * i + nl * 3 + nb + ng * 2 + nmg * 2]
            for j in range(nl):
                ds_sol["gap"][j, i] = ds_sol["pij"][j, i] ** 2 + ds_sol["qij"][j, i] ** 2 - \
                                      ds_sol["lij"][j, i] * ds_sol["vi"][int(f[j]), i]
        # Solutions for the microgrids
        mg_sol = {}
        mg_sol["pg"] = zeros((nmg, T))
        mg_sol["qg"] = zeros((nmg, T))
        mg_sol["pug"] = zeros((nmg, T))
        mg_sol["qug"] = zeros((nmg, T))
        mg_sol["pbic_ac2dc"] = zeros((nmg, T))
        mg_sol["pbic_dc2ac"] = zeros((nmg, T))
        mg_sol["qbic"] = zeros((nmg, T))
        mg_sol["pess_ch"] = zeros((nmg, T))
        mg_sol["pess_dc"] = zeros((nmg, T))
        mg_sol["eess"] = zeros((nmg, T))
        mg_sol["pmess"] = zeros((nmg, T))
        mg_sol["ppv"] = zeros((nmg, T))
        for i in range(nmg):
            for t in range(T):
                mg_sol["pg"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PG]
                mg_sol["qg"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + QG]
                mg_sol["pug"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PUG]
                mg_sol["qug"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + QUG]
                mg_sol["pbic_ac2dc"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PBIC_AC2DC]
                mg_sol["pbic_dc2ac"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PBIC_DC2AC]
                mg_sol["qbic"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + QBIC]
                mg_sol["pess_ch"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PESS_CH]
                mg_sol["pess_dc"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PESS_DC]
                mg_sol["eess"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + EESS]
                mg_sol["ppv"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PPV]
                mg_sol["pmess"][i, t] = sol[_nv_second_stage * T + NX_MG * T * i + NX_MG * t + PMESS]
        mg_sol["gap"] = mg_sol["pbic_ac2dc"].__mul__(mg_sol["pbic_dc2ac"])
        # Solutions for the mess
        n_stops = self.n_stops
        mess_sol = {}

        for i in range(nmes):
            mess_temp = {}
            mess_temp["pmess_dc"] = zeros((nmg, T))
            mess_temp["pmess_ch"] = zeros((nmg, T))
            mess_temp["emess"] = zeros((1, T))
            for t in range(T):
                for j in range(nmg):
                    mess_temp["pmess_dc"][j, t] = sol[nv_index_ev[i] + t * self.nb_tra_ele + j]
                    mess_temp["pmess_ch"][j, t] = sol[nv_index_ev[i] + T * self.nb_tra_ele + t * self.nb_tra_ele + j]
                mess_temp["emess"][0, t] = sol[nv_index_ev[i] + T * self.nb_tra_ele * 2 + t]
            mess_sol[i] = mess_temp

        second_stage_solution = {}
        second_stage_solution["DS"] = ds_sol
        second_stage_solution["MG"] = mg_sol
        second_stage_solution["MESS"] = mess_sol

        return second_stage_solution

    def problem_formulation_microgrid(self, mg, mess):
        """
        Unit commitment problem formulation of single micro_grid
        :param micro_grid:
        :return:
        """

        try:
            T = self.T
        except:
            T = 24
        nmes = self.nmes

        pmess_l = 0
        pmess_u = 0
        for i in range(nmes):
            pmess_l -= mess[i]["PCMAX"]
            pmess_u += mess[i]["PDMAX"]

        ## 1) boundary information and objective function
        nv = NX_MG * T
        lb = zeros(nv)
        ub = zeros(nv)
        c = zeros(nv)
        q = zeros(nv)
        vtypes = ["c"] * nv
        for t in range(T):
            ## 1.1) lower boundary
            lb[t * NX_MG + PG] = 0
            lb[t * NX_MG + QG] = mg["DG"]["QMIN"]
            lb[t * NX_MG + PUG] = 0
            lb[t * NX_MG + QUG] = mg["UG"]["QMIN"]
            lb[t * NX_MG + PBIC_DC2AC] = 0
            lb[t * NX_MG + PBIC_AC2DC] = 0
            lb[t * NX_MG + QBIC] = -mg["BIC"]["SMAX"]
            lb[t * NX_MG + PESS_CH] = 0
            lb[t * NX_MG + PESS_DC] = 0
            lb[t * NX_MG + EESS] = mg["ESS"]["EMIN"]
            lb[t * NX_MG + PPV] = 0
            lb[t * NX_MG + PAC] = 0
            lb[t * NX_MG + PDC] = 0
            lb[t * NX_MG + PMESS] = pmess_l
            ## 1.2) upper boundary
            ub[t * NX_MG + PG] = mg["DG"]["PMAX"]
            ub[t * NX_MG + QG] = mg["DG"]["QMAX"]
            ub[t * NX_MG + PUG] = mg["UG"]["PMAX"]
            ub[t * NX_MG + QUG] = mg["UG"]["QMAX"]
            ub[t * NX_MG + PBIC_DC2AC] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + PBIC_AC2DC] = mg["BIC"]["PMAX"]
            ub[t * NX_MG + QBIC] = mg["BIC"]["SMAX"]
            ub[t * NX_MG + PESS_CH] = mg["ESS"]["PCH_MAX"]
            ub[t * NX_MG + PESS_DC] = mg["ESS"]["PDC_MAX"]
            ub[t * NX_MG + EESS] = mg["ESS"]["EMAX"]
            ub[t * NX_MG + PPV] = mg["PV"]["PROFILE"][t]
            ub[t * NX_MG + PMESS] = pmess_u
            ub[t * NX_MG + PAC] = mg["PD"]["AC"][t]
            ub[t * NX_MG + PDC] = mg["PD"]["DC"][t]
            ## 1.3) Objective functions
            c[t * NX_MG + PG] = mg["DG"]["COST_A"]
            c[t * NX_MG + PESS_CH] = mg["ESS"]["COST_OP"]
            c[t * NX_MG + PESS_DC] = mg["ESS"]["COST_OP"]
            c[t * NX_MG + PPV] = mg["PV"]["COST"]
            c[t * NX_MG + PAC] = Voll/1000
            c[t * NX_MG + PDC] = Voll/1000
            # c[t * NX_MG + PBIC_AC2DC] = mg["ESS"]["COST_OP"]
            # c[t * NX_MG + PBIC_DC2AC] = mg["ESS"]["COST_OP"]
            # c[t * NX_MG + PUG] = mg["DG"]["COST_A"]
            # c[t * NX_MG + PMESS] = 0.001
            ## 1.4) Upper and lower boundary information
            if t == T - 1:
                lb[t * NX_MG + EESS] = mg["ESS"]["E0"]
                ub[t * NX_MG + EESS] = mg["ESS"]["E0"]

        # 2) Formulate the equal constraints
        # 2.1) Power balance equation
        # a) AC bus equation
        Aeq = lil_matrix((T, nv))
        beq = zeros(T)
        for t in range(T):
            Aeq[t, t * NX_MG + PAC] = 1
            Aeq[t, t * NX_MG + PG] = 1
            Aeq[t, t * NX_MG + PUG] = 1
            Aeq[t, t * NX_MG + PBIC_AC2DC] = -1
            Aeq[t, t * NX_MG + PBIC_DC2AC] = mg["BIC"]["EFF_DC2AC"]
            beq[t] = mg["PD"]["AC"][t]
        # b) DC bus equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + PBIC_AC2DC] = mg["BIC"]["EFF_AC2DC"]
            Aeq_temp[t, t * NX_MG + PBIC_DC2AC] = -1
            Aeq_temp[t, t * NX_MG + PESS_CH] = -1
            Aeq_temp[t, t * NX_MG + PESS_DC] = 1
            Aeq_temp[t, t * NX_MG + PPV] = 1
            Aeq_temp[t, t * NX_MG + PDC] = 1
            Aeq_temp[t, t * NX_MG + PMESS] = 1  # The power injection from mobile energy storage systems
            beq_temp[t] = mg["PD"]["DC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # c) AC reactive power balance equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + QUG] = 1
            Aeq_temp[t, t * NX_MG + QBIC] = 1
            Aeq_temp[t, t * NX_MG + QG] = 1
            beq_temp[t] = mg["QD"]["AC"][t]
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])

        # 2.2) Energy storage balance equation
        Aeq_temp = lil_matrix((T, nv))
        beq_temp = zeros(T)
        for t in range(T):
            Aeq_temp[t, t * NX_MG + EESS] = 1
            Aeq_temp[t, t * NX_MG + PESS_CH] = -mg["ESS"]["EFF_CH"]
            Aeq_temp[t, t * NX_MG + PESS_DC] = 1 / mg["ESS"]["EFF_DC"]
            if t == 0:
                beq_temp[t] = mg["ESS"]["E0"]
            else:
                Aeq_temp[t, (t - 1) * NX_MG + EESS] = -1
        Aeq = vstack([Aeq, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        # 3) Formualte inequality constraints
        # There is no inequality constraint.

        # sol = milp(c, Aeq=Aeq.tolil(), beq=beq, A=None, b=None, xmin=lb, xmax=ub)
        # print(sol[2])

        model_micro_grid = {"c": c,
                            "q": q,
                            "lb": lb,
                            "ub": ub,
                            "vtypes": vtypes,
                            "A": None,
                            "b": None,
                            "Aeq": Aeq,
                            "beq": beq
                            }

        return model_micro_grid

    def problem_formulation_tess(self, mess, tns):
        """
        Problem formulation for transportation energy storage scheduling, including vehicle routine problem and etc.
        :param tess: specific tess information
        :param traffic_network: transportation network information
        :return:
        """
        nb_tra = self.nb_tra
        T = self.T
        nb = self.nb
        nl_tra = tns["branch"].shape[0]
        # Formulate the connection matrix between the transportation networks and power networks
        connection_matrix = zeros(((2 * nl_tra + nb_tra) * T, 4))
        weight = zeros((2 * nl_tra + nb_tra) * T)
        for i in range(T):
            for j in range(nl_tra):
                # Add from matrix
                connection_matrix[i * (2 * nl_tra + nb_tra) + j, F_BUS] = tns["branch"][j, F_BUS] + i * nb_tra
                connection_matrix[i * (2 * nl_tra + nb_tra) + j, T_BUS] = tns["branch"][j, T_BUS] + \
                                                                          tns["branch"][j, TIME] * nb_tra + i * nb_tra
                weight[i * (2 * nl_tra + nb_tra) + j] = 1
                connection_matrix[i * (2 * nl_tra + nb_tra) + j, TIME] = tns["branch"][j, TIME]

            for j in range(nl_tra):
                # Add to matrix
                connection_matrix[i * (2 * nl_tra + nb_tra) + j + nl_tra, F_BUS] = tns["branch"][j, T_BUS] + i * nb_tra
                connection_matrix[i * (2 * nl_tra + nb_tra) + j + nl_tra, T_BUS] = tns["branch"][j, F_BUS] + \
                                                                                   tns["branch"][j, TIME] * nb_tra + \
                                                                                   i * nb_tra
                weight[i * (2 * nl_tra + nb_tra) + j + nl_tra] = 1
                connection_matrix[i * (2 * nl_tra + nb_tra) + j + nl_tra, TIME] = tns["branch"][j, TIME]

            for j in range(nb_tra):
                connection_matrix[i * (2 * nl_tra + nb_tra) + 2 * nl_tra + j, F_BUS] = j + i * nb_tra
                connection_matrix[i * (2 * nl_tra + nb_tra) + 2 * nl_tra + j, T_BUS] = j + (i + 1) * nb_tra

                if tns["bus"][j, LOCATION] >= 0:
                    connection_matrix[i * (2 * nl_tra + nb_tra) + 2 * nl_tra + j, 3] = tns["bus"][j, LOCATION] + i * nb

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb_tra)
        connection_matrix = connection_matrix[index, :]
        weight = weight[index]

        # add two virtual nodes to represent the initial and end status of vehicles
        # special attention should be paid here, as the original index has been modified!
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        # From matrix
        temp = zeros((nb_tra, 4))
        weight_temp = zeros(nb_tra)
        for i in range(nb_tra):
            temp[i, 1] = i + 1
        connection_matrix = concatenate([temp, connection_matrix])
        weight = concatenate([weight_temp, weight])

        # To matrix
        for i in range(nb_tra):
            temp = zeros((1, 4))
            temp[0, 0] = nb_tra * (T - 1) + i + 1
            temp[0, 1] = nb_tra * T + 1
            if tns["bus"][i, LOCATION] >= 0:
                temp[0, 3] = tns["bus"][i, LOCATION] + (T - 1) * nb
            connection_matrix = concatenate([connection_matrix, temp])
            weight_temp = zeros(1)
            weight = concatenate([weight, weight_temp])

        # Status transition matrix
        nl_tra = connection_matrix.shape[0]
        # 0 represents that, the bus is not within the power networks
        nb_tra_ele = sum((tns["bus"][:, 2]) >= 0)
        status_matrix = zeros((T, nl_tra))
        for i in range(T):
            for j in range(nl_tra):
                if connection_matrix[j, F_BUS] >= i * nb_tra + 1 and connection_matrix[j, F_BUS] < (i + 1) * nb_tra + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb_tra + 1 and connection_matrix[j, T_BUS] > (i + 1) * nb_tra + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = zeros((T * nb_tra + 2, nl_tra))
        connection_matrix_t = zeros((T * nb_tra + 2, nl_tra))

        for i in range(T * nb_tra + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        n_stops = find(connection_matrix[:, 3]).__len__()

        assert n_stops == nb_tra_ele * T, "The number of bus stop is not right!"

        nv_tra = nl_tra + 4 * n_stops  # Status transition, discharging status, charging rate, discharging rate, spinning reserve
        lx = zeros(nv_tra)
        ux = ones(nv_tra)

        self.nv_tra = nv_tra
        self.nl_tra = nl_tra
        self.n_stops = n_stops
        self.nb_tra_ele = nb_tra_ele
        self.connection_matrix = connection_matrix

        ux[nl_tra + 0 * n_stops:nl_tra + 1 * n_stops] = 1
        ux[nl_tra + 1 * n_stops:nl_tra + 2 * n_stops] = mess["PDMAX"]
        ux[nl_tra + 2 * n_stops:nl_tra + 3 * n_stops] = mess["PCMAX"]
        ux[nl_tra + 3 * n_stops:nl_tra + 4 * n_stops] = mess["PCMAX"] + mess["PDMAX"]
        # The initial location and stop location
        lx[find(connection_matrix[:, F_BUS] == 0)] = mess["initial"]
        ux[find(connection_matrix[:, F_BUS] == 0)] = mess["initial"]
        lx[find(connection_matrix[:, T_BUS] == T * nb_tra + 1)] = mess["end"]
        ux[find(connection_matrix[:, T_BUS] == T * nb_tra + 1)] = mess["end"]

        vtypes = ["b"] * nl_tra + ["b"] * n_stops + ["c"] * 3 * n_stops

        Aeq = connection_matrix_f - connection_matrix_t
        beq = zeros(T * nb_tra + 2)
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
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, nl_tra))
        # This mapping matrix plays an important role in the connection between the power network and traffic network
        ## 1) Stopping status
        A = zeros((3 * n_stops, nv_tra))  # Charging, discharging status, RBS
        # Discharging
        A[0:n_stops, 0: nl_tra] = -power_limit.toarray() * mess["PDMAX"]
        A[0:n_stops, nl_tra + n_stops: nl_tra + 2 * n_stops] = eye(n_stops)
        # Charging
        A[n_stops:n_stops * 2, 0: nl_tra] = -power_limit.toarray() * mess["PCMAX"]
        A[n_stops:n_stops * 2, nl_tra + 2 * n_stops:nl_tra + 3 * n_stops] = eye(n_stops)
        # spinning reserve
        A[n_stops * 2: n_stops * 3, 0: nl_tra] = -power_limit.toarray() * (mess["PCMAX"] + mess["PDMAX"])
        A[n_stops * 2:n_stops * 3, nl_tra + 3 * n_stops:nl_tra + 4 * n_stops] = eye(n_stops)
        b = zeros(3 * n_stops)

        ## 2) Operating status
        Arange = zeros((2 * n_stops, nv_tra))
        brange = zeros(2 * n_stops)
        # 1) Pdc<(1-Ic)*Pdc_max
        Arange[0: n_stops, nl_tra:nl_tra + n_stops] = eye(n_stops) * mess["PDMAX"]
        Arange[0: n_stops, nl_tra + n_stops: nl_tra + n_stops * 2] = eye(n_stops)
        brange[0: n_stops] = ones(n_stops) * mess["PDMAX"]
        # 2) Pc<Ic*Pch_max
        Arange[n_stops:n_stops * 2, nl_tra: nl_tra + n_stops] = -eye(n_stops) * mess["PCMAX"]
        Arange[n_stops:n_stops * 2, nl_tra + n_stops * 2: nl_tra + n_stops * 3] = eye(n_stops)
        A = concatenate([A, Arange])
        b = concatenate([b, brange])

        ## 2) Power limitation
        Areserve = zeros((2 * n_stops, nv_tra))
        breserve = zeros(2 * n_stops)
        # 1) Pdc-Pc+Rbs<=Pdc_max
        Areserve[0: n_stops, nl_tra + n_stops: nl_tra + n_stops * 2] = eye(n_stops)
        Areserve[0: n_stops, nl_tra + n_stops * 2:nl_tra + n_stops * 3] = -eye(n_stops)
        Areserve[0: n_stops, nl_tra + n_stops * 3:nl_tra + n_stops * 4] = eye(n_stops)
        breserve[0: n_stops] = ones(n_stops) * mess["PDMAX"]
        # 2) Pc-Pdc+Rbs<=Pc_max
        Areserve[n_stops:n_stops * 2, nl_tra + n_stops: nl_tra + n_stops * 2] = - eye(n_stops)
        Areserve[n_stops:n_stops * 2, nl_tra + n_stops * 2:nl_tra + n_stops * 3] = eye(n_stops)
        Areserve[n_stops:n_stops * 2, nl_tra + n_stops * 3:nl_tra + n_stops * 4] = eye(n_stops)
        breserve[n_stops:n_stops * 2] = ones(n_stops) * mess["PCMAX"]

        A = concatenate([A, Areserve])
        b = concatenate([b, breserve])

        # Add constraints on the energy status
        Aenergy = zeros((2 * T, nv_tra))
        benergy = zeros(2 * T)
        for j in range(T):
            # minimal energy
            Aenergy[j, nl_tra + n_stops:nl_tra + n_stops + (j + 1) * nb_tra_ele] = 1 / mess["EFF_DC"]
            Aenergy[j, nl_tra + 2 * n_stops:nl_tra + 2 * n_stops + (j + 1) * nb_tra_ele] = -mess["EFF_CH"]
            # Aenergy[j, NX_status + 3 * n_stops + (j + 1) * nb_traffic_electric - 1] = 0.5
            if j != (T - 1):
                benergy[j] = mess["E0"] - mess["EMIN"]
            else:
                benergy[j] = 0
            # maximal energy
            Aenergy[T + j, nl_tra + n_stops: nl_tra + n_stops + (j + 1) * nb_tra_ele] = -1 / mess["EFF_DC"]
            Aenergy[T + j, nl_tra + 2 * n_stops:nl_tra + 2 * n_stops + (j + 1) * nb_tra_ele] = mess["EFF_CH"]
            if j != (T - 1):
                benergy[T + j] = mess["EMAX"] - mess["E0"]
            else:
                benergy[T + j] = 0

        A = concatenate([A, Aenergy])
        b = concatenate([b, benergy])
        c = concatenate([connection_matrix[:, TIME]*10, zeros(n_stops * 3), ones(n_stops)*mess["COST_OP"]*0.1])
        # sol = milp(zeros(NX_traffic), q=zeros(NX_traffic), Aeq=Aeq, beq=beq, A=A, b=b, xmin=lx, xmax=ux)

        model_tess = {"c": c,
                      "q": zeros(nv_tra),
                      "lb": lx,
                      "ub": ux,
                      "vtypes": vtypes,
                      "A": A,
                      "b": b,
                      "Aeq": Aeq,
                      "beq": beq,
                      "NV": nv_tra, }

        return model_tess

    def problem_formulation_tess_second_stage(self, mess):
        """
        Problem formulation for transportation energy storage scheduling, including vehicle routine problem and etc.
        :param tess: specific tess information
        :param traffic_network: transportation network information
        :return:
        """
        T = self.T
        n_stops = self.n_stops  # Number of stops in
        nb_tra_ele = self.nb_tra_ele

        nv = 2 * n_stops + T  # Status transition, charging status, charging rate, discharging rate, spinning reserve
        lb = zeros(nv)
        ub = zeros(nv)

        lb[n_stops * 2:nv] = mess["EMIN"]

        ub[n_stops * 0:n_stops * 1] = mess["PDMAX"]
        ub[n_stops * 1:n_stops * 2] = mess["PCMAX"]
        ub[n_stops * 2:nv] = mess["EMAX"]
        lb[-1] = mess["E0"]  # energy storage systems end status
        ub[-1] = mess["E0"]  # energy storage systems end status

        vtypes = ["c"] * nv
        # The energy status dynamics
        Aeq = zeros((T, nv))
        beq = zeros(T)

        for t in range(T):
            Aeq[t, n_stops * 2 + t] = 1
            Aeq[t, n_stops + nb_tra_ele * t:n_stops + nb_tra_ele * (t + 1)] = -mess["EFF_CH"]
            Aeq[t, nb_tra_ele * t: nb_tra_ele * (t + 1)] = 1 / mess["EFF_DC"]
            if t == 0:
                beq[t] = mess["E0"]
            else:
                Aeq[t, n_stops * 2 + t - 1] = -1

        c = concatenate((ones(n_stops * 2) * mess["COST_OP"]* 0.9, zeros(T)))
        # sol = milp(c, Aeq=Aeq, beq=beq, A=None, b=None, xmin=lx, xmax=ux)

        model_tess = {"c": c,
                      "q": zeros(nv),
                      "lb": lb,
                      "ub": ub,
                      "vtypes": vtypes,
                      "A": None,
                      "b": None,
                      "Aeq": Aeq,
                      "beq": beq,
                      "NX": nv, }

        return model_tess

    def scenario_generation_reduction(self, micro_grids, profile, pns, pv_profile, update=0, ns=2, ns_reduced=50,
                                      std=0.03, interval=0.05, std_pv=0.1):
        """
        Scenario generation function for the second-stage scheduling
        Stochastic variables include 1) loads in distribution networks, active loads for 2) AC bus and 3)DC bus.
        The assumption is that, the
        1) loads in distribution networks follow normal distribution nb*T
        2) loads for AC bus and DC bus follow uniform distribution nmg*T*4
        :return:
        """
        T = self.T
        nmg = self.nmg
        nb = self.nb
        db_management = DataBaseManagement()

        if update > 0:
            # 1) scenario generation
            bus_load = zeros((ns, nb * T))
            mg_load = zeros((ns, nmg * T * 2))
            mg_pv = zeros((ns, nmg * T))
            weight = ones(ns) / ns
            for i in range(ns):
                for t in range(T):
                    for j in range(nb):
                        bus_load[i, t * nb + j] = pns["bus"][j, PD] * (1 + random.normal(0, std)) * profile[t]

                    pv_rand = random.normal(0, std_pv)  # all PV are correlated!
                    for j in range(nmg):
                        mg_load[i, t * nmg + j] = micro_grids[j]["PD"]["AC"][t] * \
                                                  (1 + random.uniform(-interval, interval))
                        mg_load[i, nmg * T + t * nmg + j] = micro_grids[j]["PD"]["DC"][t] * \
                                                            (1 + random.uniform(-interval, interval))
                        mg_pv[i, t * nmg + j] = micro_grids[j]["PV"]["PMAX"] * pv_profile[t] * \
                                                (1 + pv_rand)

            # 2) scenario reduction
            scenario_reduction = ScenarioReduction()
            (scenario_reduced, weight_reduced) = \
                scenario_reduction.run(scenario=concatenate([bus_load, mg_load, mg_pv], axis=1), weight=weight,
                                       n_reduced=ns_reduced, power=2)
            # 3) Store the data into database
            db_management.create_table("scenarios", nb=nb, nmg=nmg)
            for i in range(ns - ns_reduced):
                for t in range(T):
                    db_management.insert_data_scenario("scenarios", scenario=i, weight=weight_reduced[i], time=t, nb=nb,
                                                       pd=scenario_reduced[i, t * nb:(t + 1) * nb].tolist(), nmg=nmg,
                                                       pd_ac=scenario_reduced[i, nb * T + t * nmg:
                                                                                 nb * T + (t + 1) * nmg].tolist(),
                                                       pd_dc=scenario_reduced[i, nb * T + nmg * T + t * nmg:
                                                                                 nb * T + nmg * T + (
                                                                                         t + 1) * nmg].tolist(),
                                                       ppv=scenario_reduced[i, nb * T + nmg * T * 2 + t * nmg:
                                                                               nb * T + nmg * T * 2 + (
                                                                                       t + 1) * nmg].tolist())
        else:
            # 4) if not updated, inquery the database
            scenario_reduced = zeros((ns - ns_reduced, nb * T + nmg * T * 3))
            weight_reduced = zeros(ns - ns_reduced)
            for i in range(ns - ns_reduced):
                for t in range(T):
                    data = db_management.inquery_data_scenario(table_name="scenarios", scenario=i, time=t)
                    weight_reduced[i] = data[1]
                    scenario_reduced[i, nb * t:nb * (t + 1)] = array(data[3:nb + 3])
                    scenario_reduced[i, nb * T + nmg * t:nb * T + nmg * (t + 1)] = array(data[nb + 3:nb + 3 + nmg])
                    scenario_reduced[i, nb * T + nmg * T + nmg * t:nb * T + nmg * T + nmg * (t + 1)] = \
                        array(data[nb + 3 + nmg:nb + 3 + nmg * 2])
                    scenario_reduced[i, nb * T + nmg * T * 2 + nmg * t:nb * T + nmg * T * 2 + nmg * (t + 1)] = \
                        array(data[nb + 3 + nmg * 2:nb + 3 + nmg * 3])
            # assert sum(weight_reduced) == 1, "The weight factor is not right!"

        # 4) return value
        ds_load_profile = scenario_reduced[:, 0:nb * T] * 2
        mgs_load_profile = scenario_reduced[:, nb * T:nb * T + nmg * T * 2] * 1
        pv_load_profile = scenario_reduced[:, nb * T + nmg * T * 2:] * 1.5

        # profile_second_stage = zeros((ns, T))
        microgrids_second_stage = [0] * (ns - ns_reduced)
        # for i in range(ns):
        #     for j in range(T):
        #         profile_second_stage[i, j] = profile[j] * (1 + 0.5 * random.random())
        #
        for i in range(ns - ns_reduced):
            microgrids_second_stage[i] = deepcopy(micro_grids)
            for j in range(nmg):
                microgrids_second_stage[i][j]["PV"]["PROFILE"] = zeros(T)
                for t in range(T):
                    microgrids_second_stage[i][j]["PD"]["AC"][t] = mgs_load_profile[i, t * nmg + j]
                    microgrids_second_stage[i][j]["QD"]["AC"][t] = mgs_load_profile[i, t * nmg + j] * 0.2
                    microgrids_second_stage[i][j]["PD"]["DC"][t] = mgs_load_profile[i, T * nmg + t * nmg + j]
                    microgrids_second_stage[i][j]["PV"]["PROFILE"][t] = pv_load_profile[i, t * nmg + j]

        return ds_load_profile, microgrids_second_stage, weight_reduced


def sub_problem_solving(problem_second_stage):
    sol = list(qcqp(problem_second_stage[0]["c"], problem_second_stage[0]["q"], Aeq=problem_second_stage[0]["Aeq"],
                    beq=problem_second_stage[0]["beq"], A=problem_second_stage[0]["A"], b=problem_second_stage[0]["b"],
                    Qc=problem_second_stage[0]["Qc"], rc=problem_second_stage[0]["rc"],
                    xmin=problem_second_stage[0]["lb"], xmax=problem_second_stage[0]["ub"]))
    # print(max(problem_second_stage[0]["A"]*sol[0]-problem_second_stage[0]["b"]))

    if sol[2] == 0:
        sol = list(qcqp(problem_second_stage[1]["c"], problem_second_stage[1]["q"], Aeq=problem_second_stage[1]["Aeq"],
                        beq=problem_second_stage[1]["beq"], A=problem_second_stage[1]["A"],
                        b=problem_second_stage[1]["b"],
                        Qc=problem_second_stage[1]["Qc"], rc=problem_second_stage[1]["rc"],
                        xmin=problem_second_stage[1]["lb"], xmax=problem_second_stage[1]["ub"]))
        sol[2] = 0
        if sol[2] !=1:
            print("The relaxed problem has not been solved!")
            problem_second_stage[1]["ub"][-1] = problem_second_stage[1]["ub"][-1] / 100
            for j in range(10):
                problem_second_stage[1]["ub"][-1] = problem_second_stage[1]["ub"][-1] * 10
                sol = list(qcqp(problem_second_stage[1]["c"],
                                problem_second_stage[1]["q"],
                                Aeq=problem_second_stage[1]["Aeq"],
                                beq=problem_second_stage[1]["beq"],
                                A=problem_second_stage[1]["A"],
                                b=problem_second_stage[1]["b"],
                                Qc=problem_second_stage[1]["Qc"],
                                rc=problem_second_stage[1]["rc"],
                                xmin=problem_second_stage[1]["lb"],
                                xmax=problem_second_stage[1]["ub"]))
                try:
                    if sol[2] == 1:
                        print("The relaxed problem has been solved!")
                        break
                except:
                    continue

        # sol[2] = 0

    return sol


if __name__ == "__main__":
    mpc = case33.case33()  # Default test case
    T = 24
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41])
    load_profile = load_profile[0:T]

    Price_wholesale = array([
        [0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.45, 0.45,
         0.45, 0.45, 0.45, 0.21, 0.21, 0.21],
    ])

    # Microgrid information
    Profile = array([
        [0.64, 0.63, 0.65, 0.64, 0.66, 0.69, 0.75, 0.91, 0.95, 0.97, 1.00, 0.97, 0.97, 0.95, 0.98, 0.99, 0.95, 0.95,
         0.94, 0.95, 0.97, 0.93, 0.85, 0.69],
        [0.78, 0.75, 0.74, 0.74, 0.75, 0.81, 0.91, 0.98, 0.99, 0.99, 1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.96, 0.95,
         0.95, 0.95, 0.96, 0.95, 0.88, 0.82],
        [0.57, 0.55, 0.55, 0.56, 0.62, 0.70, 0.78, 0.83, 0.84, 0.89, 0.87, 0.82, 0.80, 0.80, 0.84, 0.89, 0.94, 0.98,
         1.00, 0.97, 0.87, 0.79, 0.72, 0.62]
    ])
    Profile = Profile[:, 0:T]
    PV_profile = array(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 0.05, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35,
         0.14, 0.02, 0.02, 0.00, 0.00, 0.00])
    PV_profile = PV_profile[0:T]

    micro_grid_1 = deepcopy(micro_grid)
    micro_grid_1["BUS"] = 2
    micro_grid_1["PD"]["AC_MAX"] = 1000
    micro_grid_1["PD"]["DC_MAX"] = 1000
    micro_grid_1["UG"]["PMIN"] = -5000
    micro_grid_1["UG"]["PMAX"] = 5000
    micro_grid_1["UG"]["QMIN"] = -5000
    micro_grid_1["UG"]["QMAX"] = 5000
    micro_grid_1["DG"]["PMAX"] = 1000
    micro_grid_1["DG"]["QMAX"] = 1000
    micro_grid_1["DG"]["QMIN"] = -1000
    micro_grid_1["DG"]["COST_A"] = 0.1808
    micro_grid_1["DG"]["COST_B"] = 3.548 * 10
    micro_grid_1["ESS"]["PDC_MAX"] = 500
    micro_grid_1["ESS"]["COST_OP"] = 0.108
    micro_grid_1["ESS"]["PCH_MAX"] = 500
    micro_grid_1["ESS"]["E0"] = 500
    micro_grid_1["ESS"]["EMIN"] = 100
    micro_grid_1["ESS"]["EMAX"] = 1000
    micro_grid_1["BIC"]["PMAX"] = 1000
    micro_grid_1["BIC"]["QMAX"] = 1000
    micro_grid_1["BIC"]["SMAX"] = 1000
    micro_grid_1["PV"]["PMAX"] = 1000
    micro_grid_1["PV"]["COST"] = 0.0376
    micro_grid_1["PD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"]
    micro_grid_1["QD"]["AC"] = Profile[0] * micro_grid_1["PD"]["AC_MAX"] * 0.2
    micro_grid_1["PD"]["DC"] = Profile[0] * micro_grid_1["PD"]["DC_MAX"]
    # micro_grid_1["MG"]["PMIN"] = 0
    # micro_grid_1["MG"]["PMAX"] = 0

    micro_grid_2 = deepcopy(micro_grid)
    micro_grid_2["BUS"] = 4
    micro_grid_2["PD"]["AC_MAX"] = 1000
    micro_grid_2["PD"]["DC_MAX"] = 1000
    micro_grid_2["UG"]["PMIN"] = -5000
    micro_grid_2["UG"]["PMAX"] = 5000
    micro_grid_2["UG"]["QMIN"] = -5000
    micro_grid_2["UG"]["QMAX"] = 5000
    micro_grid_2["DG"]["PMAX"] = 1000
    micro_grid_2["DG"]["QMAX"] = 1000
    micro_grid_2["DG"]["QMIN"] = -1000
    micro_grid_2["DG"]["COST_A"] = 0.1808
    micro_grid_2["DG"]["COST_B"] = 3.548 * 10
    micro_grid_2["ESS"]["COST_OP"] = 0.108
    micro_grid_2["ESS"]["PDC_MAX"] = 500
    micro_grid_2["ESS"]["PCH_MAX"] = 500
    micro_grid_2["ESS"]["E0"] = 500
    micro_grid_2["ESS"]["EMIN"] = 100
    micro_grid_2["ESS"]["EMAX"] = 1000
    micro_grid_2["BIC"]["PMAX"] = 1000
    micro_grid_2["BIC"]["QMAX"] = 1000
    micro_grid_2["BIC"]["SMAX"] = 1000
    micro_grid_2["PV"]["PMAX"] = 1000
    micro_grid_2["PV"]["COST"] = 0.0376
    micro_grid_2["PD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"]
    micro_grid_2["QD"]["AC"] = Profile[1] * micro_grid_2["PD"]["AC_MAX"] * 0.2
    micro_grid_2["PD"]["DC"] = Profile[1] * micro_grid_2["PD"]["DC_MAX"]
    # micro_grid_2["MG"]["PMIN"] = 0
    # micro_grid_2["MG"]["PMAX"] = 0

    micro_grid_3 = deepcopy(micro_grid)
    micro_grid_3["BUS"] = 10
    micro_grid_3["PD"]["AC_MAX"] = 1000
    micro_grid_3["PD"]["DC_MAX"] = 1000
    micro_grid_3["UG"]["PMIN"] = -5000
    micro_grid_3["UG"]["PMAX"] = 5000
    micro_grid_3["UG"]["QMIN"] = -5000
    micro_grid_3["UG"]["QMAX"] = 5000
    micro_grid_3["DG"]["PMAX"] = 1000
    micro_grid_3["DG"]["QMAX"] = 1000
    micro_grid_3["DG"]["QMIN"] = -1000
    micro_grid_3["DG"]["COST_A"] = 0.1808
    micro_grid_3["DG"]["COST_B"] = 3.548 * 10
    micro_grid_3["ESS"]["COST_OP"] = 0.108
    micro_grid_3["ESS"]["PDC_MAX"] = 500
    micro_grid_3["ESS"]["PCH_MAX"] = 500
    micro_grid_3["ESS"]["E0"] = 500
    micro_grid_3["ESS"]["EMIN"] = 100
    micro_grid_3["ESS"]["EMAX"] = 1000
    micro_grid_3["BIC"]["PMAX"] = 1000
    micro_grid_3["BIC"]["QMAX"] = 1000
    micro_grid_3["BIC"]["SMAX"] = 1000
    micro_grid_3["PV"]["PMAX"] = 1000
    micro_grid_3["PV"]["COST"] = 0.0376
    micro_grid_3["PD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"]
    micro_grid_3["QD"]["AC"] = Profile[2] * micro_grid_3["PD"]["AC_MAX"] * 0.2
    micro_grid_3["PD"]["DC"] = Profile[2] * micro_grid_3["PD"]["DC_MAX"]
    case_micro_grids = [micro_grid_1, micro_grid_2, micro_grid_3]

    ## Transportaion network information
    ev = []
    traffic_networks = case3.transportation_network()  # Default transportation networks
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 500,
               "PDMAX": 500,
               "EFF_CH": 0.95,
               "EFF_DC": 0.9,
               "E0": 500,
               "EMAX": 1000,
               "EMIN": 100,
               "COST_OP": 0.108/2,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 1, 0]),
               "PCMAX": 500,
               "PDMAX": 500,
               "EFF_CH": 0.95,
               "EFF_DC": 0.9,
               "E0": 500,
               "EMAX": 1000,
               "EMIN": 100,
               "COST_OP": 0.108/2,
               })

    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 500,
               "PDMAX": 500,
               "EFF_CH": 0.95,
               "EFF_DC": 0.9,
               "E0": 500,
               "EMAX": 1000,
               "EMIN": 100,
               "COST_OP": 0.108/2,
               })

    Voll = 38*1e3 # $/MWh

    stochastic_dynamic_optimal_power_flow = StochasticUnitCommitmentTess()

    (sol_first_stgae, sol_second_stage) = stochastic_dynamic_optimal_power_flow.main(power_networks=mpc, mess=ev,
                                                                                     profile=load_profile.tolist(),
                                                                                     pv_profile=PV_profile,
                                                                                     micro_grids=case_micro_grids,
                                                                                     traffic_networks=traffic_networks,
                                                                                     ns=1000, ns_reduced=980)

    print(sol_second_stage[0]['DS']['gap'].max())
