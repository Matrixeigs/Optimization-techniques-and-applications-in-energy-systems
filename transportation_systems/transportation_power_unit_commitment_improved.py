"""
Transportation power based unit commitment based on IEEE-118 test system

@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
Using the sparse matrix structure to test big systems
"""
# Import data format
from numpy import array, zeros, ones, concatenate, shape, arange, eye, r_
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import vstack, hstack, lil_matrix
from transportation_systems.test_cases import case3, TIME, LOCATION, case6
from transportation_systems.test_cases import case3_modified
# Import data format for electricity networks
from unit_commitment.test_cases.case118 import F_BUS, T_BUS, BR_X, RATE_A
from unit_commitment.test_cases.case118 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
    MIN_UP, RUG, RDG, COLD_START
from unit_commitment.test_cases.case118 import BUS_ID, PD
from unit_commitment.test_cases.case118 import case118
from numpy import flatnonzero as find

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp

from unit_commitment.data_format.data_format_contigency import ALPHA, BETA, IG, PG, RS, RD, RU, THETA, NG


class TrafficPowerUnitCommitment():
    """
    Jointed traffic power networks unit commitment
    """

    def __init__(self):
        self.name = "Traffic power unit commitment"

    def run(self, electricity_networks, traffic_networks, electric_vehicles, delta=0.05, delta_r=0.02, alpha_s=0.5,
            alpha_r=0.5):
        """

        :param electricity_networks:
        :param traffic_networks:
        :param electric_vehicles:
        :param load_profile:
        :return:
        """

        nev = len(electric_vehicles)

        baseMVA, bus, gen, branch, profile = electricity_networks["baseMVA"], electricity_networks["bus"], \
                                             electricity_networks["gen"], electricity_networks["branch"], \
                                             electricity_networks["Load_profile"]
        # Modify the bus, gen and branch matrix
        bus[:, BUS_ID] = bus[:, BUS_ID] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1

        T = profile.shape[0]  # Dispatch horizon
        self.T = T

        ng = shape(electricity_networks['gen'])[0]  # number of schedule injections
        nl = shape(electricity_networks['branch'])[0]  ## number of branches
        nb = shape(electricity_networks['bus'])[0]  ## number of branches
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

        # Initial generator status
        u0 = [0] * ng  # Initial generation status
        ur = [0] * ng  # Initial generation status
        dr = [0] * ng  # Initial generation status

        for i in range(ng):
            u0[i] = int(gen[i, I0] > 0)  # if gen[i, I0] > 0, the generator is on-line, else it is off-line
            if u0[i] > 0:
                ur[i] = max(gen[i, MIN_UP] - gen[i, I0], 0)
            elif gen[i, I0] < 0:
                dr[i] = max(gen[i, MIN_DOWN] + gen[i, I0], 0)

        nx = NG * T * ng + nb * T + nl * T
        lb = zeros((nx, 1))
        ub = zeros((nx, 1))

        vtypes = ["c"] * nx

        for i in range(T):
            for j in range(ng):
                # lower boundary
                lb[ALPHA * ng * T + i * ng + j] = 0
                lb[BETA * ng * T + i * ng + j] = 0
                lb[IG * ng * T + i * ng + j] = 0
                lb[PG * ng * T + i * ng + j] = 0
                lb[RS * ng * T + i * ng + j] = 0
                lb[RU * ng * T + i * ng + j] = 0
                lb[RD * ng * T + i * ng + j] = 0

                # upper boundary
                ub[ALPHA * ng * T + i * ng + j] = 1
                ub[BETA * ng * T + i * ng + j] = 1
                ub[IG * ng * T + i * ng + j] = 1
                ub[PG * ng * T + i * ng + j] = gen[j, PG_MAX]
                ub[RS * ng * T + i * ng + j] = gen[j, RUG] / 6  # The spinning reserve capacity
                ub[RU * ng * T + i * ng + j] = gen[j, RUG] / 12  # The regulation up reserve capacity
                ub[RD * ng * T + i * ng + j] = gen[j, RUG] / 12  # The regulation down reserve capacity
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "B"

        # The bus angle
        for i in range(T):
            for j in range(nb):
                lb[NG * ng * T + i * nb + j] = -360
                ub[NG * ng * T + i * nb + j] = 360
        # The power flow
        for i in range(T):
            for j in range(nl):
                lb[NG * ng * T + T * nb + i * nl + j] = -branch[j, RATE_A]
                ub[NG * ng * T + T * nb + i * nl + j] = branch[j, RATE_A]

        c = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gen[j, COLD_START]  # Start-up cost
                c[IG * ng * T + i * ng + j] = gen[j, COST_C]
                c[PG * ng * T + i * ng + j] = gen[j, COST_B]
        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = lil_matrix((T * nb, nx))
        beq = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            Aeq[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()

            Aeq[i * nb:(i + 1) * nb,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -(
                Cft.transpose()).todense()

            beq[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]

        # 2.2) Status transformation of each unit
        Aeq_temp = lil_matrix((T * ng, nx))
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

        Aeq = vstack((Aeq, Aeq_temp),format='lil')
        beq = concatenate((beq, beq_temp), axis=0)

        # 2.3) Transmission line flows
        Aeq_temp = lil_matrix((T * nl, nx))
        beq_temp = zeros((T * nl, 1))
        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            Aeq_temp[i * nl:(i + 1) * nl,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -eye(nl)
            Aeq_temp[i * nl:(i + 1) * nl, THETA * ng * T + i * nb:THETA * ng * T + (i + 1) * nb] = X.dot(
                Cft.todense())

        Aeq = vstack((Aeq, Aeq_temp),format='lil')
        beq = concatenate((beq, beq_temp), axis=0)
        # 2.4) Power range limitation
        Aineq = lil_matrix((T * ng, nx))
        bineq = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aineq[i * ng + j, BETA * ng * T + i * ng + j] = 1
                bineq[i * ng + j] = 1

        Aineq_temp = lil_matrix((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PG_MIN]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = vstack((Aineq, Aineq_temp),format='lil')
        bineq = concatenate((bineq, bineq_temp), axis=0)

        Aineq_temp = lil_matrix((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PG_MAX]
                Aineq_temp[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1

        Aineq = vstack((Aineq, Aineq_temp))
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # 2.5) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(ur[i])
            DOWN_LIMIT[i] = T - int(dr[i])
        # 2.5.1) Up limit
        Aineq_temp = lil_matrix((sum(UP_LIMIT), nx))
        bineq_temp = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gen[i, MIN_UP]), T):
                for k in range(j - int(gen[i, MIN_UP]), j):
                    Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = vstack((Aineq, Aineq_temp),format='lil')
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.2) Down limit
        Aineq_temp = lil_matrix((sum(DOWN_LIMIT), nx))
        bineq_temp = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gen[i, MIN_DOWN]), T):
                for k in range(j - int(gen[i, MIN_DOWN]), j):
                    Aineq_temp[
                        sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]), BETA * ng * T + k * ng + i] = 1
                Aineq_temp[sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        Aineq = vstack((Aineq, Aineq_temp),format='lil')
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.5.3) Modify the upper and lower boundary of generation status
        for j in range(ng):
            for i in range(int(dr[j] + ur[j])):
                # lower boundary
                lb[IG * ng * T + i * ng + j] = u0[j]
                # upper boundary
                ub[IG * ng * T + i * ng + j] = u0[j]

        # 2.6) Ramp constraints:
        # 2.6.1) Ramp up limitation
        Aineq_temp = lil_matrix((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, IG * ng * T + j * ng + i] = -gen[i, RUG]
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = -gen[i, PG_MIN]
        Aineq = vstack((Aineq, Aineq_temp),format='lil')
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.2) Ramp up limitation
        Aineq_temp = lil_matrix((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, IG * ng * T + (j + 1) * ng + i] = -gen[i, RDG]
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = -gen[i, PG_MIN]
        Aineq = vstack((Aineq, Aineq_temp),format='lil')
        bineq = concatenate((bineq, bineq_temp), axis=0)

        # The transportable energy storage set
        # For each vehicle
        nb_traffic = traffic_networks["bus"].shape[0]
        nl_traffic = traffic_networks["branch"].shape[0]
        nb_traffic_electric = sum((traffic_networks["bus"][:, 2]) >= 0)
        # Develop the connection matrix
        connection_matrix = zeros(((2 * nl_traffic + nb_traffic) * T, 4))
        weight = zeros(((2 * nl_traffic + nb_traffic) * T, 1))
        for i in range(T):
            for j in range(nl_traffic):
                # Add from matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, F_BUS] = traffic_networks["branch"][
                                                                                      j, F_BUS] + i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, T_BUS] = traffic_networks["branch"][j, T_BUS] + \
                                                                                  traffic_networks["branch"][
                                                                                      j, TIME] * nb_traffic + i * nb_traffic
                weight[i * (2 * nl_traffic + nb_traffic) + j, 0] = 1
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j, TIME] = traffic_networks["branch"][j, TIME]

            for j in range(nl_traffic):
                # Add to matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, F_BUS] = \
                    traffic_networks["branch"][j, T_BUS] + i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, T_BUS] = \
                    traffic_networks["branch"][j, F_BUS] + traffic_networks["branch"][
                        j, TIME] * nb_traffic + i * nb_traffic

                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, TIME] = \
                    traffic_networks["branch"][j, TIME]

            for j in range(nb_traffic):
                connection_matrix[i * (
                        2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, F_BUS] = j + i * nb_traffic  # This time slot
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, T_BUS] = j + (
                        i + 1) * nb_traffic  # The next time slot
                if traffic_networks["bus"][j, LOCATION] >= 0:
                    connection_matrix[i * (
                            2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, 3] = traffic_networks["bus"][
                                                                                        j, LOCATION] + i * nb  # Location information

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb_traffic)
        connection_matrix = connection_matrix[index, :]

        # add two virtual nodes to represent the initial and end status of vehicles
        connection_matrix[:, F_BUS] += 1
        connection_matrix[:, T_BUS] += 1
        for i in range(nb_traffic):
            temp = zeros((1, 4))
            temp[0, 1] = i + 1
            connection_matrix = concatenate([connection_matrix, temp])

        # Delete the out of range lines
        for i in range(nb_traffic):
            temp = zeros((1, 4))
            temp[0, 0] = nb_traffic * (T - 1) + i + 1
            temp[0, 1] = nb_traffic * T + 1
            if traffic_networks["bus"][i, LOCATION] >= 0:
                temp[0, 3] = traffic_networks["bus"][i, LOCATION] + (T - 1) * nb
            connection_matrix = concatenate([connection_matrix, temp])

        # Status transition matrix
        nl_traffic = connection_matrix.shape[0]
        status_matrix = zeros((T, nl_traffic))
        for i in range(T):
            for j in range(nl_traffic):
                if connection_matrix[j, F_BUS] >= i * nb_traffic + 1 and connection_matrix[j, F_BUS] < (
                        i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1

                if connection_matrix[j, F_BUS] <= i * nb_traffic + 1 and connection_matrix[j, T_BUS] > (
                        i + 1) * nb_traffic + 1:
                    status_matrix[i, j] = 1
        # Update connection matrix
        connection_matrix_f = lil_matrix((T * nb_traffic + 2, nl_traffic))
        connection_matrix_t = lil_matrix((T * nb_traffic + 2, nl_traffic))

        for i in range(T * nb_traffic + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        n_stops = find(connection_matrix[:, 3]).__len__()

        NX_traffic = nl_traffic + 6 * n_stops  # The status transition, charging status, charging rate, discharging rate, spinning reserve, up reserve, down reserve
        NX_status = nl_traffic
        lb_traffic = zeros((NX_traffic * nev, 1))
        ub_traffic = ones((NX_traffic * nev, 1))

        self.NX_traffic = NX_traffic
        self.nl_traffic = nl_traffic
        self.n_stops = n_stops
        self.nev = nev

        for i in range(nev):
            ub_traffic[i * NX_traffic + NX_status + 0 * n_stops:i * NX_traffic + NX_status + 1 * n_stops] = 1
            ub_traffic[i * NX_traffic + NX_status + 1 * n_stops:i * NX_traffic + NX_status + 2 * n_stops] = \
                ev[i]["PDMAX"]
            ub_traffic[i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 3 * n_stops] = \
                ev[i]["PCMAX"]
            ub_traffic[i * NX_traffic + NX_status + 3 * n_stops:i * NX_traffic + NX_status + 4 * n_stops] = \
                ev[i]["PCMAX"] + ev[i]["PDMAX"]
            ub_traffic[i * NX_traffic + NX_status + 4 * n_stops:i * NX_traffic + NX_status + 5 * n_stops] = \
                ev[i]["PCMAX"] + ev[i]["PDMAX"]
            ub_traffic[i * NX_traffic + NX_status + 5 * n_stops:i * NX_traffic + NX_status + 6 * n_stops] = \
                ev[i]["PCMAX"] + ev[i]["PDMAX"]

            lb_traffic[i * NX_traffic + find(connection_matrix[:, F_BUS] == 0), 0] = ev[i]["initial"]
            ub_traffic[i * NX_traffic + find(connection_matrix[:, F_BUS] == 0), 0] = ev[i]["initial"]
            lb_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1), 0] = ev[i]["end"]
            ub_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1), 0] = ev[i]["end"]

        vtypes_traffic = (["b"] * status_matrix.shape[1] + ["b"] * T * nb_traffic + ["c"] * 5 * T * nb_traffic) * nev

        Aeq_traffic = connection_matrix_f - connection_matrix_t
        beq_traffic = zeros(Aeq_traffic.shape[0])
        beq_traffic[0] = 1
        beq_traffic[-1] = -1
        # statue constraints
        Aeq_temp_traffic = lil_matrix(status_matrix)
        beq_temp_traffic = ones(status_matrix.shape[0])
        Aeq_traffic = vstack([Aeq_traffic, Aeq_temp_traffic],format='lil')
        beq_traffic = concatenate([beq_traffic, beq_temp_traffic])
        neq_traffic = Aeq_traffic.shape[0]
        Aeq_traffic = hstack([Aeq_traffic, lil_matrix((neq_traffic, 6 * n_stops))],format='lil')

        Aeq_traffic_full = lil_matrix((neq_traffic * nev, NX_traffic * nev))
        beq_traffic_full = zeros(neq_traffic * nev)
        for i in range(nev):
            Aeq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic, i * NX_traffic:(i + 1) * NX_traffic] = Aeq_traffic
            beq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic] = beq_traffic

        # Add constraints between the charging/discharging status and status
        index_stops = find(connection_matrix[:, 3])
        index_operation = arange(n_stops)
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, NX_status))

        Aev = lil_matrix((5 * n_stops * nev, NX_traffic * nev))  # Charging, discharging status,RBS,rbu,rbd

        for i in range(nev):
            # Discharging
            Aev[i * n_stops * 5:i * n_stops * 5 + n_stops,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * ev[i]["PDMAX"]

            Aev[i * n_stops * 5:i * n_stops * 5 + n_stops,
            i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + 2 * n_stops] = eye(n_stops)
            # Charging
            Aev[i * n_stops * 5 + n_stops:i * n_stops * 5 + n_stops * 2,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * ev[i]["PCMAX"]

            Aev[i * n_stops * 5 + n_stops:i * n_stops * 5 + n_stops * 2,
            i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 3 * n_stops] = eye(n_stops)
            # spinning reserve
            Aev[i * n_stops * 5 + n_stops * 2:i * n_stops * 5 + n_stops * 3,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * (ev[i]["PCMAX"] + ev[i]["PDMAX"])

            Aev[i * n_stops * 5 + n_stops * 2:i * n_stops * 5 + n_stops * 3,
            i * NX_traffic + NX_status + 3 * n_stops:i * NX_traffic + NX_status + 4 * n_stops] = eye(n_stops)
            # up reserve
            Aev[i * n_stops * 5 + n_stops * 3:i * n_stops * 5 + n_stops * 4,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * (ev[i]["PCMAX"] + ev[i]["PDMAX"])

            Aev[i * n_stops * 5 + n_stops * 3:i * n_stops * 5 + n_stops * 4,
            i * NX_traffic + NX_status + 4 * n_stops:i * NX_traffic + NX_status + 5 * n_stops] = eye(n_stops)
            # down reserve
            Aev[i * n_stops * 5 + n_stops * 4:i * n_stops * 5 + n_stops * 5,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * (ev[i]["PCMAX"] + ev[i]["PDMAX"])

            Aev[i * n_stops * 5 + n_stops * 4:i * n_stops * 5 + n_stops * 5,
            i * NX_traffic + NX_status + 5 * n_stops:i * NX_traffic + NX_status + 6 * n_stops] = eye(n_stops)

        bev = zeros(5 * n_stops * nev)
        A = Aev
        # Add constraints on the charging and discharging
        Arange = lil_matrix((2 * n_stops * nev, NX_traffic * nev))
        brange = zeros(2 * n_stops * nev)
        for i in range(nev):
            # 1) Pdc<(1-Ic)*Pdc_max
            Arange[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status:i * NX_traffic + NX_status + n_stops] = eye(n_stops) * ev[i]["PDMAX"]
            Arange[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops * 2] = eye(n_stops)
            brange[i * n_stops * 2:i * n_stops * 2 + n_stops] = ones(n_stops) * ev[i]["PDMAX"]
            # 2) Pc<Ic*Pch_max
            Arange[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2,
            i * NX_traffic + NX_status:i * NX_traffic + NX_status + n_stops] = -eye(n_stops) * ev[i]["PCMAX"]
            Arange[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2,
            i * NX_traffic + NX_status + n_stops * 2:i * NX_traffic + NX_status + n_stops * 3] = eye(n_stops)
        A = vstack([A, Arange],format='lil')
        b = concatenate([bev, brange])

        # Add constraints on the charging and discharging
        Areserve = lil_matrix((2 * n_stops * nev, NX_traffic * nev))
        breserve = zeros(2 * n_stops * nev)
        for i in range(nev):
            # 1) Pdc-Pc+Rbs+rbu<=Pdc_max
            Areserve[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops * 2] = eye(n_stops)
            Areserve[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status + n_stops * 2:i * NX_traffic + NX_status + n_stops * 3] = -eye(n_stops)
            Areserve[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status + n_stops * 3:i * NX_traffic + NX_status + n_stops * 4] = eye(n_stops)
            Areserve[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status + n_stops * 4:i * NX_traffic + NX_status + n_stops * 5] = eye(n_stops)
            breserve[i * n_stops * 2:i * n_stops * 2 + n_stops] = ones(n_stops) * ev[i]["PDMAX"]
            # 2) Pc-Pdc+rbd<=Pc_max
            Areserve[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2,
            i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops * 2] = -eye(n_stops)
            Areserve[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2,
            i * NX_traffic + NX_status + n_stops * 2:i * NX_traffic + NX_status + n_stops * 3] = eye(n_stops)
            Areserve[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2,
            i * NX_traffic + NX_status + n_stops * 5:i * NX_traffic + NX_status + n_stops * 6] = eye(n_stops)
            breserve[i * n_stops * 2 + n_stops:i * n_stops * 2 + n_stops * 2] = ones(n_stops) * ev[i]["PCMAX"]
        A = vstack([A, Areserve],format='lil')
        b = concatenate([b, breserve])

        # Add constraints on the energy status
        Aenergy = lil_matrix((2 * T * nev, NX_traffic * nev))
        benergy = zeros(2 * T * nev)
        for i in range(nev):
            for j in range(T):
                # minimal energy
                Aenergy[i * T * 2 + j, i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (
                        j + 1) * nb_traffic_electric] = 1 / ev[i]["EFF_DC"]
                Aenergy[i * T * 2 + j,
                i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 2 * n_stops + (
                        j + 1) * nb_traffic_electric] = -ev[i]["EFF_CH"]
                Aenergy[i * T * 2 + j, i * NX_traffic + NX_status + 3 * n_stops + (
                        j + 1) * nb_traffic_electric - 1] = alpha_s
                Aenergy[i * T * 2 + j, i * NX_traffic + NX_status + 4 * n_stops + (
                        j + 1) * nb_traffic_electric - 1] = alpha_r
                if j != (T - 1):
                    benergy[i * T * 2 + j] = ev[i]["E0"] - ev[i]["EMIN"]
                else:
                    benergy[i * T * 2 + j] = 0
                # maximal energy
                Aenergy[i * T * 2 + T + j, i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (
                        j + 1) * nb_traffic_electric] = -1 / ev[i]["EFF_DC"]
                Aenergy[i * T * 2 + T + j,
                i * NX_traffic + NX_status + 2 * n_stops: i * NX_traffic + NX_status + 2 * n_stops + (
                        j + 1) * nb_traffic_electric] = ev[i]["EFF_CH"]
                Aenergy[i * T * 2 + T + j, i * NX_traffic + NX_status + 5 * n_stops + (
                        j + 1) * nb_traffic_electric - 1] = alpha_r
                if j != (T - 1):
                    benergy[i * T * 2 + T + j] = ev[i]["EMAX"] - ev[i]["E0"]
                else:
                    benergy[i * T * 2 + T + j] = 0

        A = vstack([A, Aenergy],format='lil')
        b = concatenate([b, benergy])

        # Merge the variables and constraints
        neq = Aeq.shape[0]
        nineq = Aineq.shape[0]
        vtypes = vtypes + vtypes_traffic
        lb_full = concatenate([lb, lb_traffic])
        ub_full = concatenate([ub, ub_traffic])
        Aeq_full = lil_matrix((neq + neq_traffic * nev, nx + NX_traffic * nev))
        beq_full = zeros((neq + neq_traffic * nev, 1))
        Aeq_full[0:neq, 0:nx] = Aeq
        beq_full[0:neq] = beq
        Aeq_full[neq:, nx:] = Aeq_traffic_full
        beq_full[neq:, 0] = beq_traffic_full

        c_full = concatenate([c, zeros((NX_traffic * nev, 1))])

        # Add coupling constraints between the vehicles and power networks
        for i in range(min(int(max(connection_matrix[:, 3]) / nb), T)):
            row_index = find(connection_matrix[:, 3] >= i * nb + 1)
            row_index_temp = find(connection_matrix[row_index, 3] <= (i + 1) * nb)
            row_index = row_index[row_index_temp]

            if len(row_index) != 0:
                bus_index = connection_matrix[row_index, 3] - i * nb
                charging_index = NX_status + n_stops * 2 + arange(i * nb_traffic_electric,
                                                                  (i + 1) * nb_traffic_electric)
                discharging_index = NX_status + n_stops + arange(i * nb_traffic_electric, (i + 1) * nb_traffic_electric)
                power_traffic_charging = sparse((-ones(len(bus_index)), (bus_index, charging_index)), (nb, NX_traffic))

                power_traffic_discharging = sparse((ones(len(bus_index)), (bus_index, discharging_index)),
                                                   (nb, NX_traffic))
                for j in range(nev):
                    Aeq_full[i * nb:(i + 1) * nb, nx + j * NX_traffic: nx + (j + 1) * NX_traffic] = (
                            power_traffic_discharging + power_traffic_charging).toarray()

        nineq_traffic = A.shape[0]
        Aineq_full = lil_matrix((nineq + nineq_traffic, nx + NX_traffic * nev))
        bineq_full = zeros((nineq + nineq_traffic))
        Aineq_full[0:nineq, 0:nx] = Aineq
        bineq_full[0:nineq] = bineq[:, 0]
        Aineq_full[nineq:, nx:] = A
        bineq_full[nineq:] = b
        # Add constraints on the reserve requirements
        # 1) Spinning reserve
        Aineq_full_temp = lil_matrix((T, nx + NX_traffic * nev))
        bineq_full_temp = zeros(T)
        for i in range(T):
            for j in range(ng):
                Aineq_full_temp[i, RS * ng * T + i * ng + j] = -1
            bineq_full_temp[i] = -delta * profile[i] * sum(bus[:, PD])
            for j in range(nev):
                Aineq_full_temp[i, nx + j * NX_traffic + NX_status + n_stops * 3 + arange(i * nb_traffic_electric,
                                                                                          (
                                                                                                  i + 1) * nb_traffic_electric)] = -1
        Aineq_full = vstack([Aineq_full, Aineq_full_temp],format='lil')
        bineq_full = concatenate([bineq_full, bineq_full_temp])
        # 2) Regulation up reserve
        Aineq_full_temp = lil_matrix((T, nx + NX_traffic * nev))
        bineq_full_temp = zeros(T)
        for i in range(T):
            for j in range(ng):
                Aineq_full_temp[i, RU * ng * T + i * ng + j] = -1
            bineq_full_temp[i] = -delta_r * profile[i] * sum(bus[:, PD])
            for j in range(nev):
                Aineq_full_temp[i, nx + j * NX_traffic + NX_status + n_stops * 4 + arange(i * nb_traffic_electric,
                                                                                          (
                                                                                                  i + 1) * nb_traffic_electric)] = -1
        Aineq_full = vstack([Aineq_full, Aineq_full_temp],format='lil')
        bineq_full = concatenate([bineq_full, bineq_full_temp])
        # 3) Regulation down reserve
        Aineq_full_temp = lil_matrix((T, nx + NX_traffic * nev))
        bineq_full_temp = zeros(T)
        for i in range(T):
            for j in range(ng):
                Aineq_full_temp[i, RD * ng * T + i * ng + j] = -1
            bineq_full_temp[i] = -delta_r * profile[i] * sum(bus[:, PD])
            for j in range(nev):
                Aineq_full_temp[i, nx + j * NX_traffic + NX_status + n_stops * 5 + arange(i * nb_traffic_electric,
                                                                                          (
                                                                                                  i + 1) * nb_traffic_electric)] = -1
        Aineq_full = vstack([Aineq_full, Aineq_full_temp],format='lil')
        bineq_full = concatenate([bineq_full, bineq_full_temp])

        model = {"c": c_full,
                 "lb": lb_full,
                 "ub": ub_full,
                 "A": Aineq_full,
                 "b": bineq_full,
                 "Aeq": Aeq_full,
                 "beq": beq_full,
                 "vtypes": vtypes}

        return model

    def problem_solving(self, model):
        """

        :param model: Formulated mathematical models
        :return:
        """
        (xx, obj, success) = milp(model["c"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"])
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
        NX_traffic = self.NX_traffic
        nl_traffic = self.nl_traffic
        n_stops = self.n_stops
        nev = self.nev

        alpha = zeros((ng, T))
        beta = zeros((ng, T))
        ig = zeros((ng, T))
        pg = zeros((ng, T))
        rs = zeros((ng, T))
        rug = zeros((ng, T))
        rdg = zeros((ng, T))
        theta = zeros((nb, T))
        pf = zeros((nl, T))

        # The status transition, charging status, charging rate, discharging rate, spinning reserve, up reserve, down reserve        x_stops = zeros((NX_traffic, nev))
        tsn_ev = zeros((nl_traffic, nev))
        ich_ev = zeros((n_stops, nev))
        pdc_ev = zeros((n_stops, nev))
        pch_ev = zeros((n_stops, nev))
        rs_ev = zeros((n_stops, nev))
        ru_ev = zeros((n_stops, nev))
        rd_ev = zeros((n_stops, nev))

        for i in range(T):
            for j in range(ng):
                alpha[j, i] = sol[ALPHA * ng * T + i * ng + j]
                beta[j, i] = sol[BETA * ng * T + i * ng + j]
                ig[j, i] = sol[IG * ng * T + i * ng + j]
                pg[j, i] = sol[PG * ng * T + i * ng + j]
                rs[j, i] = sol[RS * ng * T + i * ng + j]
                rug[j, i] = sol[RU * ng * T + i * ng + j]
                rdg[j, i] = sol[RD * ng * T + i * ng + j]

        for i in range(T):
            for j in range(nb):
                theta[j, i] = sol[THETA * ng * T + i * nb + j]

        for i in range(T):
            for j in range(nl):
                pf[j, i] = sol[THETA * ng * T + T * nb + i * nl + j]

        nx = NG * T * ng + nb * T + nl * T
        # The scheduling of TESSs
        for i in range(nev):
            for j in range(nl_traffic):
                tsn_ev[j, i] = sol[nx + i * NX_traffic + j]
            for j in range(n_stops):
                ich_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 0 * n_stops + j]
            for j in range(n_stops):
                pdc_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 1 * n_stops + j]
            for j in range(n_stops):
                pch_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 2 * n_stops + j]
            for j in range(n_stops):
                rs_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 3 * n_stops + j]
            for j in range(n_stops):
                ru_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 4 * n_stops + j]
            for j in range(n_stops):
                rd_ev[j, i] = sol[nx + i * NX_traffic + nl_traffic + 5 * n_stops + j]

        solution = {"ALPHA": alpha,
                    "BETA": beta,
                    "IG": ig,
                    "PG": pg,
                    "RS": rs,
                    "RUG": rug,
                    "RDG": rdg,
                    "THETA": theta,
                    "PF": pf,
                    "TSN": tsn_ev,
                    "ICH": ich_ev,
                    "PDC": pdc_ev,
                    "PCH": pch_ev,
                    "RBS": rs_ev,
                    "RBU": ru_ev,
                    "RBD": rd_ev,
                    }

        return solution


if __name__ == "__main__":
    electricity_networks = case118()  # Default test case
    ev = []
    traffic_networks = case6.transportation_network()  # Default test case
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([1, 0, 0, 0, 0, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([0, 1, 0, 0, 0, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([0, 0, 1, 0, 0, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([0, 0, 0, 1, 0, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([0, 0, 0, 0, 1, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
               "end": array([0, 0, 0, 0, 0, 1]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev = ev*6


    # traffic_networks = case3.transportation_network()  # Default test case
    # ev.append({"initial": array([1, 0, 0]),
    #            "end": array([1, 0, 0]),
    #            "PCMAX": 100,
    #            "PDMAX": 100,
    #            "EFF_CH": 0.9,
    #            "EFF_DC": 0.9,
    #            "E0": 200,
    #            "EMAX": 400,
    #            "EMIN": 100,
    #            "COST_OP": 0.01,
    #            })

    # traffic_networks = case3_modified.transportation_network()
    # ev.append({"initial": traffic_networks["initial"],
    #            "end": traffic_networks["end"],
    #            "PCMAX": 100,
    #            "PDMAX": 100,
    #            "EFF_CH": 0.9,
    #            "EFF_DC": 0.9,
    #            "E0": 200,
    #            "EMAX": 400,
    #            "EMIN": 100,
    #            "COST_OP": 0.01,
    #            })


    traffic_power_unit_commitment = TrafficPowerUnitCommitment()

    model = traffic_power_unit_commitment.run(electricity_networks=electricity_networks,
                                              traffic_networks=traffic_networks, electric_vehicles=ev)
    (sol, obj) = traffic_power_unit_commitment.problem_solving(model)
    sol = traffic_power_unit_commitment.result_check(sol)
    print(sol)
