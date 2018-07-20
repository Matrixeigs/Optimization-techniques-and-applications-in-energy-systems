"""
Dynamic dispatch of traffic and power networks
Some notes on the jointed dispatch:
[1] Traffic networks are modelled using the time-space networks
[2] The unit commitment problem is used for the dispatch of power systems
[3] The connection matrix is used for the mapping between traffic networks and power networks
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""

from numpy import array, zeros, ones, concatenate, shape, diag, arange, eye
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, diags, vstack

from distribution_system_optimization.test_cases.case33 import case33
from transportation_systems.test_cases import case3, TIME, LOCATION

# Import data format for electricity networks
from pypower import ext2int
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, RATE_A
from pypower.idx_bus import PD, VMAX, VMIN, QD
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from numpy import flatnonzero as find

from gurobipy import *

from matplotlib import pyplot


class TrafficPowerNetworks():
    """
    Jointed traffic power networks dispatch
    """

    def __init__(self):
        self.name = "Traffic power networks"

    def run(self, electricity_networks, traffic_networks, electric_vehicles, load_profile):
        """

        :param electricity_networks:
        :param traffic_networks:
        :param electric_vehicles:
        :param load_profile:
        :return:
        """
        T = load_profile.shape[0]  # Dispatch horizon
        nev = len(electric_vehicles)

        mpc = ext2int.ext2int(electricity_networks)
        baseMVA, bus, gen, branch, gencost = mpc["baseMVA"], mpc["bus"], mpc["gen"], mpc["branch"], mpc["gencost"]

        nb = shape(mpc['bus'])[0]  ## number of buses
        nl = shape(mpc['branch'])[0]  ## number of branches
        ng = shape(mpc['gen'])[0]  ## number of dispatchable injections

        f = branch[:, F_BUS]  ## list of "from" buses
        t = branch[:, T_BUS]  ## list of "to" buses
        i = range(nl)  ## double set of row indices

        # Connection matrix
        Cf = sparse((ones(nl), (i, f)), (nl, nb))
        Ct = sparse((ones(nl), (i, t)), (nl, nb))
        Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))
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

        lx = concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l])
        ux = concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u])

        model = Model("Dynamic OPF")
        x = {}
        NX = 3 * nl + nb + 2 * ng
        nx = NX * T
        lx_full = zeros(nx)
        ux_full = zeros(nx)
        for i in range(T):
            lx_full[i * NX:(i + 1) * NX] = lx
            ux_full[i * NX:(i + 1) * NX] = ux
        vtypes_full = ["c"] * nx
        # Add system level constraints
        Aeq_p = hstack([Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng))])
        beq_p = bus[:, PD] / baseMVA
        # Add constraints for each sub system
        Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg])
        beq_q = bus[:, QD] / baseMVA
        Aeq_KVL = hstack([-2 * diags(Branch_R), -2 * diags(Branch_X),
                          diags(Branch_R ** 2) + diags(Branch_X ** 2), Cf.T - Ct.T,
                          zeros((nl, 2 * ng))])
        beq_KVL = zeros(nl)

        Aeq = vstack([Aeq_p, Aeq_q, Aeq_KVL])
        Aeq = Aeq.toarray()
        neq = Aeq.shape[0]
        Aeq_full = zeros((neq * T, nx))
        beq_full = zeros(neq * T)
        for i in range(T):
            Aeq_full[i * neq:(i + 1) * neq, i * NX:(i + 1) * NX] = Aeq
            beq_full[i * neq:(i + 1) * neq] = concatenate([beq_p * load_profile[i], beq_q * load_profile[i], beq_KVL])

        # For the transportation network
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

        NX_traffic = nl_traffic + 2 * n_stops  # The status transition, charging and discharging rate
        NX_status = nl_traffic
        lb_traffic = zeros(NX_traffic * nev)
        ub_traffic = ones(NX_traffic * nev)
        for i in range(nev):
            ub_traffic[i * NX_traffic + NX_status:i * NX_traffic + NX_status + n_stops] = \
                ev[i]["PDMAX"]
            ub_traffic[i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + 2 * n_stops] = \
                ev[i]["PCMAX"]

            lb_traffic[i * NX_traffic + find(connection_matrix[:, F_BUS] == 0)] = ev[i]["initial"]
            ub_traffic[i * NX_traffic + find(connection_matrix[:, F_BUS] == 0)] = ev[i]["initial"]
            lb_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = ev[i]["end"]
            ub_traffic[i * NX_traffic + find(connection_matrix[:, T_BUS] == T * nb_traffic + 1)] = ev[i]["end"]

        vtypes_traffic = (["b"] * status_matrix.shape[1] + ["c"] * 2 * T * nb_traffic) * nev

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
        Aeq_traffic = concatenate([Aeq_traffic, zeros((neq_traffic, 2 * n_stops))], axis=1)

        Aeq_traffic_full = zeros((neq_traffic * nev, NX_traffic * nev))
        beq_traffic_full = zeros(neq_traffic * nev)
        for i in range(nev):
            Aeq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic, i * NX_traffic:(i + 1) * NX_traffic] = Aeq_traffic
            beq_traffic_full[i * neq_traffic:(i + 1) * neq_traffic] = beq_traffic

        # Merge the variables and constraints
        vtypes = vtypes_full + vtypes_traffic
        lb = concatenate([lx_full, lb_traffic])
        ub = concatenate([ux_full, ub_traffic])
        Aeq = zeros((neq * T + neq_traffic * nev, nx + NX_traffic * nev))
        beq = zeros(neq * T + neq_traffic * nev)
        Aeq[0:neq * T, 0:nx] = Aeq_full
        beq[0:neq * T] = beq_full
        Aeq[neq * T:, nx:] = Aeq_traffic_full
        beq[neq * T:] = beq_traffic_full

        # Add coupling constraints between the vehicles and distribution networks
        for i in range(int(max(connection_matrix[:, 3]) / nb)):
            row_index = find(connection_matrix[:, 3] >= i * nb + 1)
            row_index_temp = find(connection_matrix[row_index, 3] <= (i + 1) * nb)
            row_index = row_index[row_index_temp]

            if len(row_index) != 0:
                bus_index = connection_matrix[row_index, 3] - i * nb
                charging_index = NX_status + T * nb_traffic + arange(i * nb_traffic, (i + 1) * nb_traffic)
                discharging_index = NX_status + arange(i * nb_traffic, (i + 1) * nb_traffic)
                power_traffic_charging = sparse((-ones(len(bus_index)) / baseMVA, (bus_index, charging_index)),
                                                (nb, NX_traffic))

                power_traffic_discharging = sparse((ones(len(bus_index)) / baseMVA, (bus_index, discharging_index)),
                                                   (nb, NX_traffic))
                for j in range(nev):
                    Aeq[i * neq:i * neq + nb, nx + j * NX_traffic: nx + (j + 1) * NX_traffic] = (
                            power_traffic_discharging + power_traffic_charging).toarray()

        # Add constraints between the charging/discharging status and status
        index_stops = find(connection_matrix[:, 3])
        index_operation = arange(n_stops)
        power_limit = sparse((ones(n_stops), (index_operation, index_stops)), (n_stops, NX_status))

        Aev = zeros((2 * n_stops * nev, NX_traffic * nev))

        for i in range(nev):
            Aev[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * ev[i]["PDMAX"]

            Aev[i * n_stops * 2:i * n_stops * 2 + n_stops,
            i * NX_traffic + NX_status:i * NX_traffic + NX_status + n_stops] = eye(n_stops)

            Aev[i * n_stops * 2 + n_stops:(i + 1) * n_stops * 2,
            i * NX_traffic:i * NX_traffic + NX_status] = -power_limit.toarray() * ev[i]["PCMAX"]

            Aev[i * n_stops * 2 + n_stops:(i + 1) * n_stops * 2,
            i * NX_traffic + NX_status + n_stops:(i + 1) * NX_traffic] = eye(n_stops)

        bev = zeros(2 * n_stops * nev)
        A = concatenate([zeros((2 * n_stops * nev, nx)), Aev], axis=1)

        # Add constraints on the energy status
        Aenergy = zeros((2 * T * nev, NX_traffic * nev))
        benergy = zeros(2 * T * nev)
        for i in range(nev):
            for j in range(T):
                # minimal energy
                Aenergy[i * T * 2 + j, i * NX_traffic + NX_status:i * NX_traffic + NX_status + (j + 1) * nb_traffic] = 1
                Aenergy[i * T * 2 + j,
                i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (j + 1) * nb_traffic] = -1
                if j != (T - 1):
                    benergy[i * T * 2 + j] = ev[i]["E0"] - ev[i]["EMIN"]
                else:
                    benergy[i * T * 2 + j] = 0
                # maximal energy
                Aenergy[i * T * 2 + T + j,
                i * NX_traffic + NX_status:i * NX_traffic + NX_status + (j + 1) * nb_traffic] = -1
                Aenergy[i * T * 2 + T + j,
                i * NX_traffic + NX_status + n_stops:i * NX_traffic + NX_status + n_stops + (j + 1) * nb_traffic] = 1
                if j != (T - 1):
                    benergy[i * T * 2 + T + j] = ev[i]["EMAX"] - ev[i]["E0"]
                else:
                    benergy[i * T * 2 + T + j] = 0

        Aenergy = concatenate([zeros((2 * T * nev, nx)), Aenergy], axis=1)
        A = concatenate([A, Aenergy])
        b = concatenate([bev, benergy])

        nx += NX_traffic * nev
        for i in range(nx):
            if vtypes[i] == "c":
                x[i] = model.addVar(lb=lb[i], ub=ub[i], vtype=GRB.CONTINUOUS)
            elif vtypes[i] == "b":
                x[i] = model.addVar(lb=lb[i], ub=ub[i], vtype=GRB.BINARY)

        for i in range(Aeq.shape[0]):
            expr = 0
            for j in range(nx):
                expr += x[j] * Aeq[i, j]
            model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

        for i in range(A.shape[0]):
            expr = 0
            for j in range(nx):
                expr += x[j] * A[i, j]
            model.addConstr(lhs=expr, sense=GRB.LESS_EQUAL, rhs=b[i])

        for i in range(T):
            for j in range(nl):
                model.addConstr(
                    x[i * NX + j] * x[i * NX + j] + x[i * NX + j + nl] * x[i * NX + j + nl] <= x[i * NX + j + 2 * nl] *
                    x[f[j] + i * NX + 3 * nl])
        obj = 0
        for i in range(T):
            for j in range(ng):
                obj += gencost[j, 4] * x[i * NX + j + 3 * nl + nb] * x[i * NX + j + 3 * nl + nb] * baseMVA * baseMVA + \
                       gencost[j, 5] * x[i * NX + j + 3 * nl + nb] * baseMVA + gencost[j, 6]

        # Add charging/discharging cost
        for i in range(nev):
            for j in range(2 * n_stops):
                obj += ev[i]["COST_OP"] * x[NX * T + i * NX_traffic + NX_status + j]

        model.setObjective(obj)
        model.Params.OutputFlag = 1
        model.Params.LogToConsole = 1
        model.Params.DisplayInterval = 1
        model.Params.MIPGap = 0.01
        model.Params.LogFile = ""

        model.optimize()

        xx = []
        for v in model.getVars():
            xx.append(v.x)

        obj = obj.getValue()

        # return the result
        Pij = zeros((nl, T))
        Qij = zeros((nl, T))
        Iij = zeros((nl, T))
        Vm = zeros((nb, T))
        Pg = zeros((ng, T))
        Qg = zeros((ng, T))
        Gap = zeros((nl, T))
        Routine = zeros((NX_status, nev))
        Scheduling_charging = zeros((n_stops, nev))
        Scheduling_discharging = zeros((n_stops, nev))

        for i in range(T):
            for j in range(nl):
                Pij[j, i] = xx[i * NX + j]
                Qij[j, i] = xx[i * NX + nl + j]
                Iij[j, i] = xx[i * NX + 2 * nl + j]
            for j in range(nb):
                Vm[j, i] = xx[i * NX + 3 * nl + j]
            for j in range(ng):
                Pg[j, i] = xx[i * NX + 3 * nl + nb + j]
                Qg[j, i] = xx[i * NX + 3 * nl + nb + ng + j]

        for i in range(T):
            for j in range(nl):
                Gap[j, i] = Pij[j, i] ** 2 + Qij[j, i] ** 2 - Vm[int(f[j]), i] * Iij[j, i]

        xx = array(xx).reshape((len(xx), 1))

        for i in range(nev):
            Routine[:, i] = xx[T * NX + i * NX_traffic:T * NX + i * NX_traffic + NX_status, 0]
            Scheduling_discharging[:, i] = xx[
                                           T * NX + i * NX_traffic + NX_status:T * NX + i * NX_traffic + NX_status + n_stops,
                                           0]
            Scheduling_charging[:, i] = xx[T * NX + i * NX_traffic + NX_status + n_stops:T * NX + (i + 1) * NX_traffic,
                                        0]
        # return the routine
        routine = zeros((NX_status, nev))
        for i in range(nev):
            for j in range(NX_status):
                if Routine[j, i] > 0:
                    # routine[j, i] = traffic_networks["bus"][int((connection_matrix[j, T_BUS] - 1) % nb_traffic),LOCATION]
                    routine[j, i] = int((connection_matrix[j, T_BUS] - 1) % nb_traffic)
                else:
                    try:
                        routine[j, i] = routine[j - 1, i]
                    except:
                        pass

        return obj


if __name__ == "__main__":
    from pypower import runopf

    electricity_networks = case33()  # Default test case
    traffic_networks = case3.transportation_network()  # Default test case

    load_profile = array(
        [0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95,
         0.81, 0.59, 0.35, 0.14, 0.17, 0.41, 0.63, 0.86, 0.94, 1.00])
    # load_profile = array([0.14])

    ev = []

    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 0.5,
               "PDMAX": 0.5,
               "E0": 2,
               "EMAX": 4,
               "EMIN": 1,
               "COST_OP": 0.01,
               })

    ev.append({"initial": array([1, 0, 0]),
               "end": array([0, 0, 1]),
               "PCMAX": 0.5,
               "PDMAX": 0.5,
               "E0": 2,
               "EMAX": 4,
               "EMIN": 1,
               "COST_OP": 0.01,
               })

    traffic_power_networks = TrafficPowerNetworks()

    (xx, obj, residual) = traffic_power_networks.run(electricity_networks=electricity_networks,
                                                     traffic_networks=traffic_networks, electric_vehicles=ev,
                                                     load_profile=load_profile)

    result = runopf.runopf(case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
    print(max(residual))
