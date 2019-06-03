"""
Dynamic optimal power flow with multiple microgrids
"""

from distribution_system_optimization.test_cases import case33
from micro_grids.test_cases.cases_unit_commitment import micro_grid
from transportation_systems.test_cases import case3, TIME, LOCATION

from scipy import zeros, shape, ones, diag, concatenate, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack
from numpy import flatnonzero as find
from numpy import array, tile, arange

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, RATE_A
from pypower.idx_bus import PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from solvers.mixed_integer_quadratic_constrained_cplex import mixed_integer_quadratic_constrained_programming as miqcp
from copy import deepcopy

from distribution_system_optimization.data_format.idx_opf import PBIC_AC2DC, PG, PESS_DC, PBIC_DC2AC, PUG, PESS_CH, RUG, \
    RESS, RG, EESS, NX_MG, QBIC, QUG, QG


class DynamicOptimalPowerFlowTess():
    def __init__(self):
        self.name = "Dynamic optimal power flow with tess"

    def main(self, case, profile, tess, traffic_networks):
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
        # Number of buses in the transportation networks
        nb_traffic = traffic_networks["bus"].shape[0]
        self.nb_traffic = nb_traffic
        # Number of tess
        nev = len(tess)
        self.nev = nev

        # 1) Formulate the constraints for each system
        # 1.1) Distribution networks
        model_distribution_networks = self.problem_formualtion_distribution_networks(case=case, profile=profile,
                                                                                     tess=tess,
                                                                                     traffic_networks=traffic_networks)
        # 1.2) Transportation energy storage systems
        model_tess = {}
        for i in range(nev):
            model_tess[i] = self.problem_formulation_tess(tess=tess[i], traffic_networks=traffic_networks)

        # 2) System level modelling
        # 2.1) Merge the model between distribution networks and microgrdis
        nVariables_distribution_network = len(model_distribution_networks["c"])
        if model_distribution_networks["Aeq"] is not None:
            neq_distribution_network = model_distribution_networks["Aeq"].shape[0]
        else:
            neq_distribution_network = 0
        if model_distribution_networks["A"] is not None:
            nineq_distribution_network = model_distribution_networks["A"].shape[0]
        else:
            nineq_distribution_network = 0

        nVariables = int(nVariables_distribution_network)
        neq = int(neq_distribution_network)
        nineq = int(nineq_distribution_network)

        lx = model_distribution_networks["lb"]
        ux = model_distribution_networks["ub"]
        c = model_distribution_networks["c"]
        vtypes = model_distribution_networks["vtypes"]

        if model_distribution_networks["beq"] is not None:
            beq = model_distribution_networks["beq"]
        else:
            beq = zeros(0)

        if model_distribution_networks["b"] is not None:
            b = model_distribution_networks["b"]
        else:
            b = zeros(0)

        Qc = model_distribution_networks["Qc"]
        q = model_distribution_networks["q"]

        # 2.2) Merge the model between distribution networks and transportation networks
        NX_traffic = self.NX_traffic

        nVariables_index_tess = zeros(nev + 1)
        neq_index_tess = zeros(nev + 1)
        nineq_index_tess = zeros(nev + 1)
        nVariables_index_tess[0] = nVariables_distribution_network
        neq_index_tess[0] = neq_distribution_network
        nineq_index_tess[0] = nineq_distribution_network

        for i in range(nev):
            nVariables_index_tess[i + 1] = nVariables_index_tess[i] + len(model_tess[i]["c"])
            neq_index_tess[i + 1] = neq_index_tess[i] + model_tess[i]["Aeq"].shape[0]
            nineq_index_tess[i + 1] = nineq_index_tess[i] + model_tess[i]["A"].shape[0]
            nVariables += len(model_tess[i]["c"])
            neq += int(model_tess[i]["Aeq"].shape[0])
            nineq += int(model_tess[i]["A"].shape[0])

            c = concatenate([c, model_tess[i]["c"]])
            q = concatenate([q, model_tess[i]["q"]])
            lx = concatenate([lx, model_tess[i]["lb"]])
            ux = concatenate([ux, model_tess[i]["ub"]])
            vtypes += model_tess[i]["vtypes"]
            beq = concatenate([beq, model_tess[i]["beq"]])
            b = concatenate([b, model_tess[i]["b"]])

        A_full = zeros((int(nineq_index_tess[-1]), int(nVariables_index_tess[-1])))
        Aeq_full = zeros((int(neq_index_tess[-1]), int(nVariables_index_tess[-1])))
        Aeq = model_distribution_networks["Aeq"]
        A = model_distribution_networks["A"]

        if Aeq is not None:
            Aeq_full[0:int(neq_index_tess[0]), 0:int(nVariables_index_tess[0])] = Aeq
        if A is not None:
            A_full[0:int(nineq_index_tess[0]), 0:int(nVariables_index_tess[0])] = A

        for i in range(nev):
            Aeq_full[int(neq_index_tess[i]):int(neq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["Aeq"]

            A_full[int(nineq_index_tess[i]):int(nineq_index_tess[i + 1]),
            int(nVariables_index_tess[i]):int(nVariables_index_tess[i + 1])] = model_tess[i]["A"]

        # Coupling constraints between distribution networks and tess
        Az2x = zeros((2 * nb_traffic * T, int(nVariables_index_tess[-1] - nVariables_index_tess[0])))
        n_stops = self.n_stops
        NX_status = self.nl_traffic

        for i in range(nev):
            Az2x[0:n_stops, i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + 2 * n_stops] = \
                -eye(n_stops)  # Discharging
            Az2x[0:n_stops, i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 3 * n_stops] = \
                eye(n_stops)  # Charging

        Aeq_temp = concatenate([model_distribution_networks["Ax2z"], Az2x], axis=1)
        beq_temp = zeros(2 * nb_traffic * T)

        Aeq = concatenate([Aeq_full, Aeq_temp])
        beq = concatenate([beq, beq_temp])
        A = A_full
        # 3) Solve the problem
        rc = zeros(len(Qc))
        (xx, obj, success) = miqcp(c, q, Aeq=Aeq, beq=beq, vtypes=vtypes, A=A, b=b, Qc=Qc, rc=rc, xmin=lx, xmax=ux)

        # 4) Check the solutions, including microgrids and distribution networks
        # 4.1) Scheduling plan of distribution networks
        sol_distribution_network = self.solution_check_distribution_network(xx[0:nVariables_distribution_network])
        # 4.2) Scheduling plan of each MG
        # a) Energy storage system group

        sol_tess = self.solution_check_tess(sol=xx[int(nVariables_index_tess[0]):int(nVariables_index_tess[-1])])

        return sol_distribution_network, sol_tess

    def problem_formualtion_distribution_networks(self, case, profile, tess, traffic_networks):
        T = self.T

        mpc = ext2int(case)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections
        nb_traffic = self.nb_traffic

        self.nl = nl
        self.nb = nb
        self.ng = ng
        nev = self.nev

        n = traffic_networks["bus"][:, -1]  ## list of integration index
        Pev_l = zeros(nb_traffic)  ## lower boundary for energy exchange
        Pev_u = zeros(nb_traffic)  ## upper boundary for energy exchange
        for i in range(nb_traffic):
            for j in range(nev):
                Pev_l[i] = Pev_l[i] - tess[j]["PCMAX"] / 1000 / baseMVA
                Pev_u[i] = Pev_u[i] + tess[j]["PDMAX"] / 1000 / baseMVA

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices
        self.f = f  ## record from bus for each branch

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
        Cev = sparse((ones(nb_traffic), (n, range(nb_traffic))), (nb, nb_traffic))

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

        nx = int(3 * nl + nb + 2 * ng + nb_traffic)
        self.nx = nx  # Number of decision variable within each time slot

        lx = concatenate([tile(concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l, Pev_l]), T)])
        ux = concatenate([tile(concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u, Pev_u]), T)])

        vtypes = ["c"] * nx * T
        NX = nx * T  # Number of total decision variables

        # Add system level constraints
        # 1) Active power balance
        Aeq_p = zeros((nb * T, NX))
        beq_p = zeros(nb * T)
        for i in range(T):
            Aeq_p[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([Ct - Cf, zeros((nb, nl)),
                                                                       -diag(Ct * Branch_R) * Ct,
                                                                       zeros((nb, nb)), Cg,
                                                                       zeros((nb, ng)), Cev]).toarray()

            beq_p[i * nb:(i + 1) * nb] = profile[i] * bus[:, PD] / baseMVA

        # 2) Reactive power balance
        Aeq_q = zeros((nb * T, NX))
        beq_q = zeros(nb * T)
        for i in range(T):
            Aeq_q[i * nb:(i + 1) * nb, i * nx: (i + 1) * nx] = hstack([zeros((nb, nl)), Ct - Cf,
                                                                       -diag(Ct * Branch_X) * Ct,
                                                                       zeros((nb, nb)),
                                                                       zeros((nb, ng)), Cg,
                                                                       zeros((nb, nb_traffic))]).toarray()
            beq_q[i * nb:(i + 1) * nb] = profile[i] * bus[:, QD] / baseMVA
        # 3) KVL equation
        Aeq_kvl = zeros((nl * T, NX))
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

        c = zeros(NX)
        q = zeros(NX)
        c0 = 0
        for t in range(T):
            for i in range(ng):
                c[t * nx + i + 3 * nl + nb] = gencost[i, 5] * baseMVA
                q[t * nx + i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
                c0 += gencost[i, 6]

        # The boundary information
        Ax2z = zeros((2 * nb_traffic * T, NX))  # connection matrix with the tess
        for i in range(T):
            for j in range(nb_traffic):
                Ax2z[i * nb_traffic + j, i * nx + 3 * nl + nb + 2 * ng + j] = 1000 * baseMVA  # Active power

        # sol = miqcp(c, q, Aeq=Aeq, beq=beq, A=None, b=None, Qc=Qc, xmin=lx, xmax=ux)

        model_distribution_grid = {"c": c,
                                   "q": q,
                                   "lb": lx,
                                   "ub": ux,
                                   "vtypes": vtypes,
                                   "A": None,
                                   "b": None,
                                   "Aeq": Aeq,
                                   "beq": beq,
                                   "Qc": Qc,
                                   "c0": c0,
                                   "Ax2z": Ax2z}

        return model_distribution_grid

    def solution_check_distribution_network(self, xx):
        """
        solution check for distribution networks solution
        :param xx:
        :return:
        """
        nl = self.nl
        nb = self.nb
        ng = self.ng
        T = self.T
        nx = self.nx
        nb_traffic = self.nb_traffic
        f = self.f

        Pij = zeros((nl, T))
        Qij = zeros((nl, T))
        Iij = zeros((nl, T))
        Vi = zeros((nb, T))
        Pg = zeros((ng, T))
        Qg = zeros((ng, T))
        Pev = zeros((nb_traffic, T))
        for i in range(T):
            Pij[:, i] = xx[i * nx:i * nx + nl]
            Qij[:, i] = xx[i * nx + nl: i * nx + 2 * nl]
            Iij[:, i] = xx[i * nx + 2 * nl:i * nx + 3 * nl]
            Vi[:, i] = xx[i * nx + 3 * nl: i * nx + 3 * nl + nb]
            Pg[:, i] = xx[i * nx + 3 * nl + nb: i * nx + 3 * nl + nb + ng]
            Qg[:, i] = xx[i * nx + 3 * nl + nb + ng: i * nx + 3 * nl + nb + 2 * ng]
            Pev[:, i] = xx[i * nx + 3 * nl + nb + 2 * ng:i * nx + 3 * nl + nb + 2 * ng + nb_traffic]

        primal_residual = zeros((nl, T))
        for t in range(T):
            for i in range(nl):
                primal_residual[i, t] = Pij[i, t] * Pij[i, t] + Qij[i, t] * Qij[i, t] - Iij[i, t] * Vi[int(f[i]), t]

        sol = {"Pij": Pij,
               "Qij": Qij,
               "Iij": Iij,
               "Vi": Vi,
               "Pg": Pg,
               "Qg": Qg,
               "Pev": Pev,
               "residual": primal_residual}

        return sol

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
        A = zeros((2 * n_stops, NX_traffic))  # Charging, discharging status
        # Discharging
        A[0:n_stops, 0: NX_status] = -power_limit.toarray() * tess["PDMAX"]
        A[0:n_stops, NX_status + n_stops: NX_status + 2 * n_stops] = eye(n_stops)
        # Charging
        A[n_stops:n_stops * 2, 0: NX_status] = -power_limit.toarray() * tess["PCMAX"]

        A[n_stops:n_stops * 2, NX_status + 2 * n_stops:NX_status + 3 * n_stops] = eye(n_stops)

        b = zeros(2 * n_stops)

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

        model_tess = {"c": zeros(NX_traffic),
                      "q": zeros(NX_traffic),
                      "lb": lx,
                      "ub": ux,
                      "vtypes": vtypes,
                      "A": A,
                      "b": b,
                      "Aeq": Aeq,
                      "beq": beq,
                      "NX": NX_traffic}

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

        tsn_ev = zeros((nl_traffic, nev))
        ich_ev = zeros((n_stops, nev))
        pdc_ev = zeros((n_stops, nev))
        pch_ev = zeros((n_stops, nev))

        for i in range(nev):
            for j in range(nl_traffic):
                tsn_ev[j, i] = sol[i * NX_traffic + j]
            for j in range(n_stops):
                ich_ev[j, i] = sol[i * NX_traffic + nl_traffic + 0 * n_stops + j]
            for j in range(n_stops):
                pdc_ev[j, i] = sol[i * NX_traffic + nl_traffic + 1 * n_stops + j]
            for j in range(n_stops):
                pch_ev[j, i] = sol[i * NX_traffic + nl_traffic + 2 * n_stops + j]

        sol_tess = {"Tsn_ev": tsn_ev,
                    "Ich": ich_ev,
                    "Pdc": pdc_ev,
                    "Pch": pch_ev}

        return sol_tess


if __name__ == "__main__":
    # Distribution network information
    mpc = case33.case33()  # Default test case
    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41]) * 2

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

    dynamic_optimal_power_flow = DynamicOptimalPowerFlowTess()

    (sol_dso, sol_tess) = dynamic_optimal_power_flow.main(case=mpc, profile=load_profile.tolist(), tess=ev,
                                                          traffic_networks=traffic_networks)

    print(max(sol_dso["residual"][0]))
