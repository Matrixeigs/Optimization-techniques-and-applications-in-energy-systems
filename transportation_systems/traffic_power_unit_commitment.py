"""
Traffic power based unit commitment
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""
from numpy import array, zeros, ones, concatenate, shape, arange, eye, r_
from scipy.sparse import csr_matrix as sparse
from transportation_systems.test_cases import case3, TIME, LOCATION

# Import data format for electricity networks
from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A
from pypower.idx_cost import STARTUP
from pypower.idx_bus import BUS_TYPE, REF, PD, BUS_I
from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, RAMP_10, RAMP_AGC, RAMP_30

from numpy import flatnonzero as find

from unit_commitment.data_format.data_formate_bess import ALPHA, BETA, IG, PG, RS, RU, RD, THETA, PL, NG

from solvers.mixed_integer_quadratic_solver_cplex import mixed_integer_quadratic_programming as miqp
from unit_commitment.test_cases.case6 import case6 as power_network


class TrafficPowerUnitCommitment():
    """
    Jointed traffic power networks dispatch
    """

    def __init__(self):
        self.name = "Traffic power unit commitment"

    def run(self, electricity_networks, traffic_networks, electric_vehicles, profile):
        """

        :param electricity_networks:
        :param traffic_networks:
        :param electric_vehicles:
        :param load_profile:
        :return:
        """
        T = profile.shape[0]  # Dispatch horizon
        self.T = T

        nev = len(electric_vehicles)
        MIN_UP = -2
        MIN_DOWN = -3
        THETA = RD + 1
        baseMVA, bus, gen, branch, gencost = electricity_networks["baseMVA"], electricity_networks["bus"], \
                                             electricity_networks["gen"], electricity_networks["branch"], \
                                             electricity_networks["gencost"]

        # Modify the bus, gen and branch matrix
        bus[:, BUS_I] = bus[:, BUS_I] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1

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
                ub[PG * ng * T + i * ng + j] = gen[j, PMAX]
                ub[RS * ng * T + i * ng + j] = gen[j, RAMP_10]
                ub[RU * ng * T + i * ng + j] = gen[j, RAMP_AGC]
                ub[RD * ng * T + i * ng + j] = gen[j, RAMP_AGC]
                # variable types
                vtypes[IG * ng * T + i * ng + j] = "B"
        # The bus angle
        for i in range(T):
            for j in range(nb):
                lb[NG * ng * T + i * nb + j] = -360
                ub[NG * ng * T + i * nb + j] = 360
                if bus[j, BUS_TYPE] == REF:
                    lb[NG * ng * T + i * nb + j] = 0
                    ub[NG * ng * T + i * nb + j] = 0
        # The power flow
        for i in range(T):
            for j in range(nl):
                lb[NG * ng * T + T * nb + i * nl + j] = -branch[j, RATE_A]
                ub[NG * ng * T + T * nb + i * nl + j] = branch[j, RATE_A]

        c = zeros((nx, 1))
        q = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost
                c[ALPHA * ng * T + i * ng + j] = gencost[j, STARTUP]
                c[IG * ng * T + i * ng + j] = gencost[j, 6]
                c[PG * ng * T + i * ng + j] = gencost[j, 5]

                q[PG * ng * T + i * ng + j] = gencost[j, 4]
        # 2) Constraint set
        # 2.1) Power balance equation, for each node
        Aeq = zeros((T * nb, nx))
        beq = zeros((T * nb, 1))
        for i in range(T):
            # For the unit
            Aeq[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()

            Aeq[i * nb:(i + 1) * nb,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -(
                Cft.transpose()).todense()

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
            Aeq_temp[i * nl:(i + 1) * nl, THETA * ng * T + i * nb:THETA * ng * T + (i + 1) * nb] = X.dot(
                Cft.todense())

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
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1

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
                    Aineq_temp[
                        sum(UP_LIMIT[0:i]) + j - int(gencost[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
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
                Aineq_temp[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_30] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_30]

        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.2) Ramp up limitation
        Aineq_temp = zeros((ng * (T - 1), nx))
        bineq_temp = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_temp[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_temp[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_temp[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = gen[i, RAMP_30] - gen[i, PMIN]
                bineq_temp[i * (T - 1) + j] = gen[i, RAMP_30]
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.3) Rs<=Ig*RAMP_10
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_10]
                Aineq_temp[i * ng + j, RS * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.4) ru<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RU * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # 2.6.5) rd<=Ig*RAMP_AGC
        Aineq_temp = zeros((T * ng, nx))
        bineq_temp = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_temp[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, RAMP_AGC]
                Aineq_temp[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = concatenate((Aineq, Aineq_temp), axis=0)
        bineq = concatenate((bineq, bineq_temp), axis=0)
        # The transportable energy storage set
        # For each vehicle
        nb_traffic = traffic_networks["bus"].shape[0]
        nl_traffic = traffic_networks["branch"].shape[0]
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
                connection_matrix[i * (
                        2 * nl_traffic + nb_traffic) + 2 * nl_traffic + j, 3] = traffic_networks["bus"][
                                                                                    j, LOCATION] + i * nb  # Location information

        # Delete the out of range lines
        index = find(connection_matrix[:, T_BUS] < T * nb_traffic)
        connection_matrix = connection_matrix[index, :]
        weight = weight[index]

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
        connection_matrix_f = zeros((T * nb_traffic + 2, nl_traffic))
        connection_matrix_t = zeros((T * nb_traffic + 2, nl_traffic))

        for i in range(T * nb_traffic + 2):
            connection_matrix_f[i, find(connection_matrix[:, F_BUS] == i)] = 1
            connection_matrix_t[i, find(connection_matrix[:, T_BUS] == i)] = 1

        n_stops = find(connection_matrix[:, 3]).__len__()

        NX_traffic = nl_traffic + 6 * n_stops  # The status transition, charging status, charging rate, discharging rate, spinning reserve, up reserve, down reserve
        NX_status = nl_traffic
        lb_traffic = zeros((NX_traffic * nev, 1))
        ub_traffic = ones((NX_traffic * nev, 1))
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
        Aeq_temp_traffic = status_matrix
        beq_temp_traffic = ones(status_matrix.shape[0])
        Aeq_traffic = concatenate([Aeq_traffic, Aeq_temp_traffic])
        beq_traffic = concatenate([beq_traffic, beq_temp_traffic])
        neq_traffic = Aeq_traffic.shape[0]
        Aeq_traffic = concatenate([Aeq_traffic, zeros((neq_traffic, 6 * n_stops))], axis=1)

        Aeq_traffic_full = zeros((neq_traffic * nev, NX_traffic * nev))
        beq_traffic_full = zeros(neq_traffic * nev)
        for i in range(nev):
            Aeq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic, i * NX_traffic:(i + 1) * NX_traffic] = Aeq_traffic
            beq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic] = beq_traffic

        # Add constraints between the charging/discharging status and status
        index_stops = find(connection_matrix[:, 3])
        index_operation = arange(n_stops)
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, NX_status))

        Aev = zeros((5 * n_stops * nev, NX_traffic * nev))  # Charging, discharging status,RBS,rbu,rbd

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

        # Add constraints on the energy status
        Aenergy = zeros((2 * T * nev, NX_traffic * nev))
        benergy = zeros(2 * T * nev)
        for i in range(nev):
            for j in range(T):
                # minimal energy
                Aenergy[i * T * 2 + j,
                i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (j + 1) * nb_traffic] = 1
                Aenergy[i * T * 2 + j,
                i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 2 * n_stops + (
                        j + 1) * nb_traffic] = -1
                if j != (T - 1):
                    benergy[i * T * 2 + j] = ev[i]["E0"] - ev[i]["EMIN"]
                else:
                    benergy[i * T * 2 + j] = 0
                # maximal energy
                Aenergy[i * T * 2 + T + j,
                i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (j + 1) * nb_traffic] = -1
                Aenergy[i * T * 2 + T + j,
                i * NX_traffic + NX_status + 2 * n_stops:i * NX_traffic + NX_status + 2 * n_stops + (
                        j + 1) * nb_traffic] = 1
                if j != (T - 1):
                    benergy[i * T * 2 + T + j] = ev[i]["EMAX"] - ev[i]["E0"]
                else:
                    benergy[i * T * 2 + T + j] = 0

        A = concatenate([A, Aenergy])
        b = concatenate([bev, benergy])

        # Merge the variables and constraints
        neq = Aeq.shape[0]
        nineq = Aineq.shape[0]
        vtypes = vtypes + vtypes_traffic
        lb_full = concatenate([lb, lb_traffic])
        ub_full = concatenate([ub, ub_traffic])
        Aeq_full = zeros((neq + neq_traffic * nev, nx + NX_traffic * nev))
        beq_full = zeros((neq + neq_traffic * nev, 1))
        Aeq_full[0:neq, 0:nx] = Aeq
        beq_full[0:neq] = beq
        Aeq_full[neq:, nx:] = Aeq_traffic_full
        beq_full[neq:, 0] = beq_traffic_full

        c_full = concatenate([c, zeros((NX_traffic * nev, 1))])
        q_full = concatenate([q, zeros((NX_traffic * nev, 1))])

        # Add coupling constraints between the vehicles and power networks
        for i in range(int(max(connection_matrix[:, 3]) / nb)):
            row_index = find(connection_matrix[:, 3] >= i * nb + 1)
            row_index_temp = find(connection_matrix[row_index, 3] <= (i + 1) * nb)
            row_index = row_index[row_index_temp]

            if len(row_index) != 0:
                bus_index = connection_matrix[row_index, 3] - i * nb
                charging_index = NX_status + T * nb_traffic + arange(i * nb_traffic, (i + 1) * nb_traffic)
                discharging_index = NX_status + arange(i * nb_traffic, (i + 1) * nb_traffic)
                power_traffic_charging = sparse((-ones(len(bus_index)), (bus_index, charging_index)), (nb, NX_traffic))

                power_traffic_discharging = sparse((ones(len(bus_index)), (bus_index, discharging_index)),
                                                   (nb, NX_traffic))
                for j in range(nev):
                    Aeq_full[i * nb:(i + 1) * nb, nx + j * NX_traffic: nx + (j + 1) * NX_traffic] = (
                            power_traffic_discharging + power_traffic_charging).toarray()

        nineq_traffic = A.shape[0]

        Aineq_full = zeros((nineq + nineq_traffic, nx + NX_traffic * nev))
        bineq_full = zeros((nineq + nineq_traffic, 1))
        Aineq_full[0:nineq, 0:nx] = Aineq
        bineq_full[0:nineq] = bineq
        Aineq_full[nineq:, nx:] = A
        bineq_full[nineq:, 0] = b
        # Add constraints on the reserve requirement

        model = {"c": c_full,
                 "q": q_full,
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
        (xx, obj, success) = miqp(model["c"], model["q"], Aeq=model["Aeq"], beq=model["beq"],
                                  A=model["A"],
                                  b=model["b"], xmin=model["lb"], xmax=model["ub"],
                                  vtypes=model["vtypes"], objsense="min")
        xx = array(xx).reshape((len(xx), 1))
        return xx, obj


if __name__ == "__main__":
    from pypower import runopf

    electricity_networks = power_network()  # Default test case
    traffic_networks = case3.transportation_network()  # Default test case

    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00])

    ev = []

    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 2,
               "PDMAX": 2,
               "E0": 2,
               "EMAX": 4,
               "EMIN": 1,
               "COST_OP": 0.01,
               })

    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 2,
               "PDMAX": 2,
               "E0": 2,
               "EMAX": 4,
               "EMIN": 1,
               "COST_OP": 0.01,
               })

    traffic_power_unit_commitment = TrafficPowerUnitCommitment()

    model = traffic_power_unit_commitment.run(electricity_networks=electricity_networks,
                                              traffic_networks=traffic_networks, electric_vehicles=ev,
                                              profile=load_profile)
    (sol, obj) = traffic_power_unit_commitment.problem_solving(model)
