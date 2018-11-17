"""
Transportation power based unit commitment based on IEEE-118 test system

@author:Zhao Tianyang, liuxc
@e-mail:zhaoty@ntu.edu.sg
Using the sparse matrix structure to test big systems
"""
# Import data format
import cplex
from numpy import array, zeros, ones, concatenate, shape, arange, eye, r_, newaxis, asarray, tile
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import vstack, hstack, lil_matrix, block_diag
from transportation_systems.test_cases import case3, TIME, LOCATION, case6
from transportation_systems.test_cases import case3_modified
# Import data format for electricity networks
from transportation_systems.case6 import F_BUS, T_BUS, BR_X, RATE_A
from transportation_systems.case6 import GEN_BUS, COST_C, COST_B, COST_A, PG_MAX, PG_MIN, I0, MIN_DOWN, \
    MIN_UP, RUG, RDG, COLD_START
from transportation_systems.case6 import BUS_ID, PD
from transportation_systems.case6 import case6modified
from numpy import flatnonzero as find

from solvers.mixed_integer_solvers_cplex import mixed_integer_linear_programming as milp

from unit_commitment.data_format.data_format_contigency import ALPHA, BETA, IG, PG, RS, RD, RU, THETA, NG, PL
from transportation_systems import windfarm
from scipy.stats import beta


class SecondstageTest:
    """
    Jointed traffic power networks unit commitment
    """

    def __init__(self):
        self.name = "SecondstageTest"

    def run(self, electricity_networks, traffic_networks, electric_vehicles, windfarm, delta=0.05, delta_r=0.02, alpha_s=0.5,
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
        # Modify the bus, gen and branch matrix ID start from "0"
        bus[:, BUS_ID] = bus[:, BUS_ID] - 1
        gen[:, GEN_BUS] = gen[:, GEN_BUS] - 1
        branch[:, F_BUS] = branch[:, F_BUS] - 1
        branch[:, T_BUS] = branch[:, T_BUS] - 1

        # T = profile.shape[0]  # Dispatch horizon
        T = 8
        self.T = T

        ng = shape(electricity_networks['gen'])[0]  # number of schedule injections
        nl = shape(electricity_networks['branch'])[0]  # number of branches
        nb = shape(electricity_networks['bus'])[0]  # number of bus
        nWindfarm = shape(windfarm["location"])[0]  # number of windfarms
        self.ng = ng
        self.nb = nb
        self.nl = nl

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = r_[range(nl), range(nl)]  ## double set of row indices

        ## connection matrix
        #row = branch, col=bus, 1= from, -1 = to.
        Cft = sparse((r_[ones(nl), -ones(nl)], (i, r_[f, t])), (nl, nb))
        # bus-generator location
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], arange(ng))),
                    (nb, ng))
        # csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])


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

        nx = NG * T * ng + nb * T + nl * T + T * nWindfarm
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

        # Wind output bounds
        windcapacity = windfarm["capacity"]
        for i in range(nWindfarm):
            for j in range(T):
                lb[NG * ng * T + T * nb + T * nl + i * T +j] = 0
                ub[NG * ng * T + T * nb + T * nl + i * T + j] = windcapacity[i]
        c = zeros((nx, 1))
        for i in range(T):
            for j in range(ng):
                # cost cost_c=? cost_b = generation cost
                c[ALPHA * ng * T + i * ng + j] = gen[j, COLD_START]  # Start-up cost
                c[IG * ng * T + i * ng + j] = gen[j, COST_C]
                c[PG * ng * T + i * ng + j] = gen[j, COST_B]
        # 2) Constraint set
        # 2.1) Power balance equation, for each node  Pg - Pl + Pw = Pd
        AeqPowerbalance = lil_matrix((T * nb, nx))
        beqPowerbalance = zeros((T * nb, 1))
        windmean = windfarm["meanoutput"]
        windlocation = windfarm["location"] - 1
        # windoutput = zeros((T * nb, 1))
        # for i in range(T):
        #     for j in range(nWindfarm):
        #         windoutput[i * nb + windlocation[j], 0] = windmean[i] * windfarm["capacity"][j]
        for i in range(T):
            # For the unit
            AeqPowerbalance[i * nb:(i + 1) * nb, PG * ng * T + i * ng:PG * ng * T + (i + 1) * ng] = Cg.todense()

            AeqPowerbalance[i * nb:(i + 1) * nb,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -(
                Cft.transpose()).todense()
            for j in range(nWindfarm):   # wind power injection
                AeqPowerbalance[i * nb + windlocation[j], THETA * ng * T + T * nb + T * nl + j * T + i] = 1
            beqPowerbalance[i * nb:(i + 1) * nb, 0] = profile[i] * bus[:, PD]
        beqPowerbalance = beqPowerbalance

        # 2.2) Status transformation of each unit
        Aeq_UnitStatus = lil_matrix((T * ng, nx))
        beq_UnitStatus = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aeq_UnitStatus[i * ng + j, ALPHA * ng * T + i * ng + j] = -1
                Aeq_UnitStatus[i * ng + j, BETA * ng * T + i * ng + j] = 1
                Aeq_UnitStatus[i * ng + j, IG * ng * T + i * ng + j] = 1
                if i != 0:
                    Aeq_UnitStatus[i * ng + j, IG * ng * T + (i - 1) * ng + j] = -1
                else:
                    beq_UnitStatus[i * T + j] = 0

        Aeq = vstack((AeqPowerbalance, Aeq_UnitStatus),format='lil')
        beq = concatenate((beqPowerbalance, beq_UnitStatus), axis=0)

        # 2.3) Transmission line flows
        Aeq_TransFlow = lil_matrix((T * nl, nx))
        beq_TransFlow = zeros((T * nl, 1))
        X = zeros((nl, nl))
        for i in range(nl):
            X[i, i] = 1 / branch[i, BR_X]

        for i in range(T):
            # For the unit
            Aeq_TransFlow[i * nl:(i + 1) * nl,
            THETA * ng * T + T * nb + i * nl:THETA * ng * T + T * nb + (i + 1) * nl] = -eye(nl)
            Aeq_TransFlow[i * nl:(i + 1) * nl, THETA * ng * T + i * nb:THETA * ng * T + (i + 1) * nb] = X.dot(
                Cft.todense())

        Aeq = vstack((Aeq, Aeq_TransFlow),format='lil')
        beq = concatenate((beq, beq_TransFlow), axis=0)
        # 2.4) Power range limitation
        # alpha + beta <= 1
        Aineq_signal = lil_matrix((T * ng, nx))
        bineq_signal = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_signal[i * ng + j, ALPHA * ng * T + i * ng + j] = 1
                Aineq_signal[i * ng + j, BETA * ng * T + i * ng + j] = 1
                bineq_signal[i * ng + j] = 1

        Aineq_minpower = lil_matrix((T * ng, nx))
        bineq_minpower = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_minpower[i * ng + j, IG * ng * T + i * ng + j] = gen[j, PG_MIN]
                Aineq_minpower[i * ng + j, PG * ng * T + i * ng + j] = -1
                Aineq_minpower[i * ng + j, RD * ng * T + i * ng + j] = 1
        Aineq = vstack((Aineq_signal, Aineq_minpower),format='lil')
        bineq = concatenate((bineq_signal, bineq_minpower), axis=0)

        Aineq_maxpower = lil_matrix((T * ng, nx))
        bineq_maxpower = zeros((T * ng, 1))
        for i in range(T):
            for j in range(ng):
                Aineq_maxpower[i * ng + j, IG * ng * T + i * ng + j] = -gen[j, PG_MAX]
                Aineq_maxpower[i * ng + j, PG * ng * T + i * ng + j] = 1
                Aineq_maxpower[i * ng + j, RU * ng * T + i * ng + j] = 1
                Aineq_maxpower[i * ng + j, RS * ng * T + i * ng + j] = 1

        Aineq = vstack((Aineq, Aineq_maxpower))
        bineq = concatenate((bineq, bineq_maxpower), axis=0)

        # 2.5) Start up and shut down time limitation
        UP_LIMIT = [0] * ng
        DOWN_LIMIT = [0] * ng
        for i in range(ng):
            UP_LIMIT[i] = T - int(ur[i])
            DOWN_LIMIT[i] = T - int(dr[i])
        # 2.5.1) Up limit
        Aineq_Uptime = lil_matrix((sum(UP_LIMIT), nx))
        bineq_Uptime = zeros((sum(UP_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gen[i, MIN_UP]), T):
                for k in range(j - int(gen[i, MIN_UP]), j):
                    Aineq_Uptime[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), ALPHA * ng * T + k * ng + i] = 1
                Aineq_Uptime[sum(UP_LIMIT[0:i]) + j - int(gen[i, MIN_UP]), IG * ng * T + j * ng + i] = -1
        Aineq = vstack((Aineq, Aineq_Uptime),format='lil')
        bineq = concatenate((bineq, bineq_Uptime), axis=0)
        # 2.5.2) Down limit
        Aineq_Downtime = lil_matrix((sum(DOWN_LIMIT), nx))
        bineq_Downtime = ones((sum(DOWN_LIMIT), 1))
        for i in range(ng):
            for j in range(int(gen[i, MIN_DOWN]), T):
                for k in range(j - int(gen[i, MIN_DOWN]), j):
                    Aineq_Downtime[
                        sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]), BETA * ng * T + k * ng + i] = 1
                Aineq_Downtime[sum(DOWN_LIMIT[0:i]) + j - int(gen[i, MIN_DOWN]), IG * ng * T + j * ng + i] = 1
        Aineq = vstack((Aineq, Aineq_Downtime),format='lil')
        bineq = concatenate((bineq, bineq_Downtime), axis=0)
        # 2.5.3) Modify the upper and lower boundary of generation status
        for j in range(ng):
            for i in range(int(dr[j] + ur[j])):
                # lower boundary
                lb[IG * ng * T + i * ng + j] = u0[j]
                # upper boundary
                ub[IG * ng * T + i * ng + j] = u0[j]

        # 2.6) Ramp constraints:
        # 2.6.1) Ramp up limitation
        Aineq_Rampup = lil_matrix((ng * (T - 1), nx))
        bineq_Rampup = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_Rampup[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = 1
                Aineq_Rampup[i * (T - 1) + j, PG * ng * T + j * ng + i] = -1
                Aineq_Rampup[i * (T - 1) + j, IG * ng * T + j * ng + i] = -gen[i, RUG]
                Aineq_Rampup[i * (T - 1) + j, ALPHA * ng * T + (j + 1) * ng + i] = -gen[i, PG_MIN]
        Aineq = vstack((Aineq, Aineq_Rampup),format='lil')
        bineq = concatenate((bineq, bineq_Rampup), axis=0)
        # 2.6.2) Ramp down limitation
        Aineq_Rampdown = lil_matrix((ng * (T - 1), nx))
        bineq_Rampdown = zeros((ng * (T - 1), 1))
        for i in range(ng):
            for j in range(T - 1):
                Aineq_Rampdown[i * (T - 1) + j, PG * ng * T + (j + 1) * ng + i] = -1
                Aineq_Rampdown[i * (T - 1) + j, PG * ng * T + j * ng + i] = 1
                Aineq_Rampdown[i * (T - 1) + j, IG * ng * T + (j + 1) * ng + i] = -gen[i, RDG]
                Aineq_Rampdown[i * (T - 1) + j, BETA * ng * T + (j + 1) * ng + i] = -gen[i, PG_MIN]
        Aineq = vstack((Aineq, Aineq_Rampdown), format='lil')
        bineq = concatenate((bineq, bineq_Rampdown), axis=0)

        # 2.7) wind power output
        windmean = windfarm["meanoutput"]
        windcapacity = windfarm["capacity"]
        Aineq_wind = lil_matrix((nWindfarm * T, nx))
        bineq_wind = zeros((nWindfarm * T, 1))
        for i in range(nWindfarm):
            for j in range(T):
                Aineq_wind[i * T + j, THETA * ng * T + T * nb + T * nl + i * T + j] = 1
                bineq_wind[i * T + j, 0] = windmean[j] * windcapacity[i]
        Aineq = vstack((Aineq, Aineq_wind), format='lil')
        bineq = concatenate((bineq, bineq_wind), axis=0)
        # The transportable energy storage set
        # For each vehicle
        nb_traffic = traffic_networks["bus"].shape[0]
        nl_traffic = traffic_networks["branch"].shape[0]
        nb_traffic_electric = sum((traffic_networks["bus"][:, 2]) >= 0)
        # Develop the connection matrix/ compact TSN matrix
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

            # To Acrs
            for j in range(nl_traffic):
                # Add to matrix
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, F_BUS] = \
                    traffic_networks["branch"][j, T_BUS] + i * nb_traffic
                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, T_BUS] = \
                    traffic_networks["branch"][j, F_BUS] + traffic_networks["branch"][
                        j, TIME] * nb_traffic + i * nb_traffic

                connection_matrix[i * (2 * nl_traffic + nb_traffic) + j + nl_traffic, TIME] = \
                    traffic_networks["branch"][j, TIME]

            # parking arcs
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
        nl_traffic = connection_matrix.shape[0]  # nTransitArcs
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

        # number of parking arcs in the whole 24 time slot tsn
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
            # lb_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1), 0] = ev[i]["end"]
            # ub_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1), 0] = ev[i]["end"]

        vtypes_traffic = (["b"] * NX_status + ["b"] * n_stops + ["c"] * 5 * n_stops) * nev

        Aeq_traffic = connection_matrix_f - connection_matrix_t
        beq_traffic = zeros(Aeq_traffic.shape[0])
        beq_traffic[0] = 1
        beq_traffic[-1] = -1
        # statue constraints
        Aeq_temp_traffic = lil_matrix(status_matrix)
        beq_temp_traffic = ones(status_matrix.shape[0])
        Aeq_traffic = vstack([Aeq_traffic, Aeq_temp_traffic], format='lil')
        beq_traffic = concatenate([beq_traffic, beq_temp_traffic])
        neq_traffic = Aeq_traffic.shape[0]
        Aeq_traffic = hstack([Aeq_traffic, lil_matrix((neq_traffic, 6 * n_stops))], format='lil')

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

        A = vstack([A, Aenergy], format='lil')
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
            bineq_full_temp[i] = - delta_r * profile[i] * sum(bus[:, PD])
            for j in range(nev):
                Aineq_full_temp[i, nx + j * NX_traffic + NX_status + n_stops * 5 + arange(i * nb_traffic_electric,
                                                                                          (
                                                                                                  i + 1) * nb_traffic_electric)] = -1
        Aineq_full = vstack([Aineq_full, Aineq_full_temp], format='lil')
        bineq_full = concatenate([bineq_full, bineq_full_temp])


        # 2. Define auxiliary variables for distributionally robust
        # original x = alpha beta Ig pg rs ru rd theta pl pwind {Nstatus ic pc pd rsev ruev rdev} * nev
        # first stage variables: alpha, beta Ig Nstatus Ic
        # second stage: Pg Rs Ru Rd theta Pl """"{Pwind}"""" Pd Pc Rsev Ruev Rdev
        # ita, rho, lambda, pi, pim, y0, y1, y2

        # nFSVar = PG * T * ng + (nl_traffic + n_stops) * nev
        nFSpower = PG * T * ng
        nFStraffic = (nl_traffic + n_stops) * nev
        nSSVar = 4 * T * ng + nb * T + nl * T + nWindfarm * T + 5 * n_stops * nev

        # Define first stage cost function
        c_first = c_full[:nFSpower, 0]
        c_second = c_full[nFSpower:nx, 0]
        for i in range(nev):
            c_first = concatenate([c_first, c_full[nx + i * NX_traffic: nx + i * NX_traffic + (nl_traffic + n_stops), 0]])
            c_second = concatenate([c_second, c_full[nx + i * NX_traffic + (nl_traffic + n_stops): nx + (i+1) * NX_traffic , 0]])

        ###########################################
        # Define ambiguity set
        # Assume two wind farm 700MW and 800MW at bus 36 and 49
        # Lifted uncertainty set Q: {errorlb <=v <= errorub, u>=max(v - Cjrt), u<= max(max(v-cjrt))}
        # V = [vfarm1,t1 vfarm1,t2......vfarm2,T]
        # U = [u.segment1.farm1,t1 u.seg1.f1,t2, ......]
        nUncertain = nWindfarm * T
        nAuxiliary = 3 * nUncertain # gjrt(v) 3 segment function
        nita = 1
        nrho = nUncertain
        nlambda = nAuxiliary
        npi = 2 * nUncertain + 3 * nAuxiliary
        npim = npi   # with M pim, M = number of rows of Tss and Wss
        ny0 = nSSVar   #y0 nSSVar X 1 vector
        ny1 = ny0         #y1 nSSVar X nWindfarm  assume to depend on current time step
        ny2 = ny1         #y2 nSSVar X 3 * nwindfarm
        # 1) E(Gv) == mu
        G = eye(nUncertain, nUncertain)
        mu = windfarm["errormean"][:T]
        # errorstd = windfarm["errorstd"]

        # 2) Define uncertainty set Q = {Cv + Du <= e}
        errorub = windfarm["errorub"][:T]
        errorlb = windfarm["errorlb"][:T]
        #flatten
        mu = concatenate([mu, mu])
        errorub = concatenate([errorub, errorub])
        errorlb = concatenate([errorlb, errorlb])
        C1rt = zeros((nUncertain, 1))
        C2rt = errorlb / 3
        C3rt = 2 * errorlb / 3
        QC = zeros((2 * nUncertain + 3 * nAuxiliary, nUncertain))
        QD = zeros((2 * nUncertain + 3 * nAuxiliary, nAuxiliary))
        Qe = zeros((2 * nUncertain + 3 * nAuxiliary, 1))
        # errorlb <=v <= errorub
        for i in range(nUncertain):
            QC[i, i] = 1
            Qe[i] = errorub[i]
        for i in range(nUncertain):
            QC[nUncertain + i, i] = -1
            Qe[nUncertain + i] = -errorlb[i]
        # u>=max(v - Cjrt, 0)>>>>>>>>u>= v-Cirt and u>=0
        for i in range(nAuxiliary):
            QD[2 * nUncertain + i, i] = -1
            Qe[2 * nUncertain + i, 0] = 0
        for i in range(nUncertain):
            QD[2 * nUncertain + nAuxiliary + i, i] = -1
            QC[2 * nUncertain + nAuxiliary + i, i] = 1
            Qe[2 * nUncertain + nAuxiliary + i, 0] = C1rt[i]
        for i in range(nUncertain):
            QD[3 * nUncertain + nAuxiliary + i, nUncertain + i] = -1
            QC[3 * nUncertain + nAuxiliary + i, i] = 1
            Qe[3 * nUncertain + nAuxiliary + i] = C2rt[i]
        for i in range(nUncertain):
            QD[4 * nUncertain + nAuxiliary + i, 2 * nUncertain + i] = -1
            QC[4 * nUncertain + nAuxiliary + i, i] = 1
            Qe[4 * nUncertain + nAuxiliary + i] = C3rt[i]
        # u<= max(max(v-cjrt))
        # obtain max(max(Cjrt)) = errorub - Cirt
        maxC1rt = errorub - C1rt
        maxC2rt = errorub - C2rt
        maxC3rt = errorub - C3rt
        maxC = concatenate([maxC1rt, maxC2rt, maxC3rt])
        for i in range(nAuxiliary):
            QD[5 * nUncertain + nAuxiliary + i, i] = 1
            Qe[5 * nUncertain + nAuxiliary + i] = maxC[i]

        # 3) Calculate E(u)= sigma by assuming v follows Beta distribution
        # Balpha = zeros(nUncertain)
        # Bbeta = zeros(nUncertain)
        # for i in range(nUncertain)
        #     Balpha[i] = ((1 - mu[i]) * pow(mu[i], 2)) / pow(errorstd[i], 2) - mu[i]
        #     Bbeta[i] = (1 - mu[i]) * errorstd[i] / mu[i]
        #     mean, var, skew, kurt = beta.stats(Balpha[i], Bbeta[i], moments = 'mvsk')
        sigma = zeros((nAuxiliary, 1))
        for i in range(nUncertain):
            sigma[i] = max((mu[i] - C1rt[i]), 0)
            sigma[nUncertain + i] = max((mu[i] - C2rt[i]), 0)
            sigma[2 * nUncertain + i] = max((mu[i] - C3rt[i]), 0)

        ###################################################################
        # Define Second stage constaint matrices Tss for X and Wss for Y and b(v) = b0 + sum(bs * vs)
        # vs = [v1,1 v1,2,...v1,T,...v2,T]
        bineq_full = bineq_full[:, newaxis]
        Aeq_exempt = vstack([Aeq_full[0: T * nb, :], Aeq_full[(T * nb + T * ng): neq, :]], format='lil')
        beq_exempt = concatenate([beq_full[0: T * nb], beq_full[(T * nb + T * ng): neq]])
        Aineq_exempt = vstack([Aineq_full[T * ng: 3 * T * ng, :], Aineq_full[(3 * T * ng + sum(UP_LIMIT) + sum(DOWN_LIMIT)):, :]], format='lil')
        # Aineq_exempt = Aineq_full[T * ng: T * ng + 8, :]
        bineq_exempt = concatenate([bineq_full[T * ng: 3 * T * ng, :], bineq_full[(3 * T * ng + sum(UP_LIMIT) + sum(DOWN_LIMIT)):]])
        # bineq_exempt = bineq_full[T * ng: T*ng + 8]
        TAeqtemp = Aeq_exempt[:, :nFSpower]
        TAineqtemp = Aineq_exempt[:, :nFSpower]
        WAeqtemp = Aeq_exempt[:, nFSpower: nx]
        WAineqtemp = Aineq_exempt[:, nFSpower: nx]
        SSub = ub_full[nFSpower: nx]
        SSlb = lb_full[nFSpower: nx]
        for i in range(nev):
            TAeqtemp = hstack([TAeqtemp, Aeq_exempt[:, nx + i * NX_traffic: nx + i * NX_traffic + nl_traffic + n_stops]], format='lil')
            TAineqtemp = hstack([TAineqtemp, Aineq_exempt[:, nx + i * NX_traffic: nx + i * NX_traffic + nl_traffic + n_stops]], format='lil')
            WAeqtemp = hstack([WAeqtemp, Aeq_exempt[:, nx + i * NX_traffic + nl_traffic + n_stops: nx + (i+1) * NX_traffic]], format='lil')
            WAineqtemp = hstack([WAineqtemp, Aineq_exempt[:, nx + i * NX_traffic + nl_traffic + n_stops: nx + (i+1) * NX_traffic]], format='lil')
            SSub = concatenate([SSub, ub_full[nx + i * NX_traffic + nl_traffic + n_stops: nx + (i+1) * NX_traffic]])
            SSlb = concatenate([SSlb, lb_full[nx + i * NX_traffic + nl_traffic + n_stops: nx + (i+1) * NX_traffic]])
        # Tss = vstack([TAeqtemp, - TAeqtemp, TAineqtemp], format='lil')
        # Wss = vstack([WAeqtemp, - WAeqtemp, WAineqtemp], format='lil')
        Tss = TAineqtemp
        Wss = WAineqtemp
        # Tss = vstack([TAineqtemp], format='lil')
        # Wss = vstack([WAineqtemp], format='lil')
        cLength = Aineq_exempt.shape[0]
        # cLength = Aineq_exempt.shape[0]
        h0 = bineq_exempt
        # h0 = concatenate([bineq_exempt])
        hv = zeros((cLength, nUncertain))
        for i in range(nWindfarm):
            for j in range(T):
                hv[2 * ng * T + 2 * ng * (T - 1) + i * T + j, i * T + j] = 1

        # Define first stage constraints where only alpha beta Ig Nstatus Ic are incorporated
        nscenario = 50
        Aeq_first = Aeq_UnitStatus[:, :nFSpower]
        Aeq_first = hstack([Aeq_first, zeros((Aeq_first.shape[0], nFStraffic + nSSVar*nscenario))], format='lil')
        for i in range(nev):
            if i == 0:
                Aeqtemp = Aeq_traffic_full[:, :(nl_traffic + n_stops)]
                continue
            Aeqtemp = hstack([Aeqtemp, Aeq_traffic_full[:, i * NX_traffic: i * NX_traffic + (nl_traffic + n_stops)]], format='lil')
        Aeqtemp = hstack([zeros((Aeqtemp.shape[0], nFSpower)), Aeqtemp, zeros((Aeqtemp.shape[0], nSSVar * nscenario))])
        Aeq_first = vstack([Aeq_first, Aeqtemp], format='lil')
        Aineq_first = vstack([Aineq_signal[:, :nFSpower], Aineq_Uptime[:, :nFSpower], Aineq_Downtime[:, :nFSpower]], format ='lil')
        Aineq_first = hstack([Aineq_first, zeros((Aineq_first.shape[0], nFStraffic + nSSVar * nscenario))], format='lil')
        beq_first = concatenate([beq_UnitStatus, beq_traffic_full[:, newaxis]])
        bineq_first = concatenate([bineq_signal, bineq_Uptime, bineq_Downtime])

        # define scenarios for wind realization
        winderror = lil_matrix((T, nscenario))
        prepower = windfarm["meanoutput"]
        std = windfarm["errorstd"]
        b_alpha = ((1 - prepower) * prepower ** 2) / (std ** 2) - prepower
        b_beta = ((1 - prepower)/prepower) * b_alpha
        for i in range(T):
            winderror[i, :] = beta.rvs(b_alpha[i], b_beta[i], size=nscenario)
        winderror = vstack([winderror, winderror], format='lil')
        for i in range(nscenario):
            samples = hv * winderror[:, i]
            if i == 0:
                hverror = samples
                continue
            hverror = concatenate([hverror, samples])
        h0 = tile(h0, (nscenario, 1))
        bineq_all = h0 + hverror
        # define stochastic problem
        c_first = c_first[:,newaxis]
        c_second = c_second[:,newaxis]
        c_second = tile(c_second,(nscenario, 1))
        c_all = concatenate([c_first, c_second/nscenario])
        Aineq_all = Tss
        Aeq_T = TAeqtemp
        for i in range(nscenario-1):
            Aineq_all = vstack([Aineq_all, Tss], format='lil')
            Aeq_T = vstack([Aeq_T, TAeqtemp])
        W = block_diag(eval(('Wss,'*nscenario).rstrip(',')),format='lil')
        Weq = block_diag(eval(('WAeqtemp,'*nscenario).rstrip(',')),format='lil')
        Aineq_all = hstack([Aineq_all, W], format='lil')
        Aineq_all = vstack([Aineq_first, Aineq_all], format='lil')
        bineq_all = concatenate([bineq_first, bineq_all])
        TWeq = hstack([Aeq_T, Weq], format='lil')
        Aeq_all = vstack([Aeq_first, TWeq], format='lil')
        beq_exempt = tile(beq_exempt, (nscenario,1))
        beq_all = concatenate([beq_first, beq_exempt])
        lb_first = lb_full[:nFSpower, 0]
        ub_first = ub_full[:nFSpower, 0]
        for i in range(nev):
            lb_first = concatenate([lb_first, lb_full[nx + i * NX_traffic: nx + i * NX_traffic + nl_traffic + n_stops, 0]])
            ub_first = concatenate([ub_first, ub_full[nx + i * NX_traffic: nx + i * NX_traffic + nl_traffic + n_stops, 0]])
        SSlb = tile(SSlb, (nscenario, 1))
        SSub = tile(SSub, (nscenario, 1))
        lb_first = lb_first[:, newaxis]
        ub_first = ub_first[:, newaxis]
        lb_all = concatenate([lb_first, SSlb])
        ub_all = concatenate([ub_first, SSub])
        vtypes = ["c"] * (nFSpower + nFStraffic + nSSVar * nscenario)
        vtypes[0: nFSpower + nFStraffic] = ["B"] * (nFSpower + nFStraffic)
        vtypes_first = ["b"] * (nFStraffic + nFSpower)
        model = {"c": c_all,
                 "lb": lb_all,
                 "ub": ub_all,
                 "A": Aineq_all,
                 "b": bineq_all,
                 "Aeq": Aeq_all,
                 "beq": beq_all,
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
                pg[j, i] = sol[PG + i * ng + j]
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
                tsn_ev[j, i] = sol[3*ng*T + i * (nl_traffic + n_stops) + j]
            for j in range(n_stops):
                ich_ev[j, i] = sol[3*ng*T + i * (nl_traffic + n_stops) + nl_traffic + 0 * n_stops + j]
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
    electricity_networks = case6modified()  # Default test case
    windfarm = windfarm.windfarm()
    ev = []
    traffic_networks = case3.transportation_network()  # Default test case
    ev.append({"initial": array([1, 0, 0]),
               "end": array([1, 0, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 1, 0]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 100,
               "PDMAX": 100,
               "EFF_CH": 0.9,
               "EFF_DC": 0.9,
               "E0": 200,
               "EMAX": 400,
               "EMIN": 100,
               "COST_OP": 0.01,
               })
    # ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
    #            "end": array([0, 0, 0, 1, 0, 0]),
    #            "PCMAX": 100,
    #            "PDMAX": 100,
    #            "EFF_CH": 0.9,
    #            "EFF_DC": 0.9,
    #            "E0": 200,
    #            "EMAX": 400,
    #            "EMIN": 100,
    #            "COST_OP": 0.01,
    #            })
    # ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
    #            "end": array([0, 0, 0, 0, 1, 0]),
    #            "PCMAX": 100,
    #            "PDMAX": 100,
    #            "EFF_CH": 0.9,
    #            "EFF_DC": 0.9,
    #            "E0": 200,
    #            "EMAX": 400,
    #            "EMIN": 100,
    #            "COST_OP": 0.01,
    #            })
    # ev.append({"initial": array([1, 0, 0, 0, 0, 0]),
    #            "end": array([0, 0, 0, 0, 0, 1]),
    #            "PCMAX": 100,
    #            "PDMAX": 100,
    #            "EFF_CH": 0.9,
    #            "EFF_DC": 0.9,
    #            "E0": 200,
    #            "EMAX": 400,
    #            "EMIN": 100,
    #            "COST_OP": 0.01,
    #            })
    # ev = ev*3


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


    traffic_power_unit_commitment = SecondstageTest()

    model = traffic_power_unit_commitment.run(electricity_networks=electricity_networks,
                                              traffic_networks=traffic_networks, windfarm=windfarm, electric_vehicles=ev)
    (sol, obj) = traffic_power_unit_commitment.problem_solving(model)
    sol = traffic_power_unit_commitment.result_check(sol)
    print(sol)
