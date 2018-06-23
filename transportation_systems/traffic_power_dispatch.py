"""
Dynamic dispatch of traffic and power networks
Some notes on the jointed dispatch:
[1] Traffic networks are modelled using the time-space networks
[2] The unit commitment problem is used for the dispatch of power systems
[3] The connection matrix is used for the mapping between traffic networks and power networks
@author:Zhao Tianyang
@e-mail:zhaoty@ntu.edu.sg
"""

from numpy import array, zeros, ones, concatenate, shape, repeat, diag
from transportation_systems.transportation_network_models import TransportationNetworkModel
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, diags, vstack

from distribution_system_optimization.test_cases.case33 import case33
from transportation_systems.test_cases import case3, TIME

# Import data format for electricity networks
from pypower import ext2int
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN

from gurobipy import *

from matplotlib import pyplot


# Import data format for traffic networks


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




        for i in range(nx):
            x[i] = model.addVar(lb=lx_full[i], ub=ux_full[i], vtype=GRB.CONTINUOUS)
        for i in range(neq * T):
            expr = 0
            for j in range(nx):
                expr += x[j] * Aeq_full[i, j]
            model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq_full[i])

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

        model.setObjective(obj)
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.DisplayInterval = 1
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

        return obj


if __name__ == "__main__":
    from pypower import runopf

    electricity_networks = case33()  # Default test case
    traffic_networks = case3.transportation_network()  # Default test case

    load_profile = array([0.17, 0.41, 0.63, 0.86, 0.94, 1.00, 0.95, 0.81, 0.59, 0.35, 0.14])
    # load_profile = array([0.14])

    ev = []

    ev.append({"initial": array([[1], [0], [0]]),
               "end": array([[0], [0], [1]]),
               "PCMAX": 1000,
               "PDCMAX": 1000,
               "E0": 2000,
               })

    ev.append({"initial": array([[0], [1], [0]]),
               "end": array([[1], [0], [0]]),
               "PCMAX": 1000,
               "PDCMAX": 1000,
               "E0": 2000,
               })
    traffic_power_networks = TrafficPowerNetworks()

    (xx, obj, residual) = traffic_power_networks.run(electricity_networks=electricity_networks,
                                                     traffic_networks=traffic_networks, electric_vehicles=ev,
                                                     load_profile=load_profile)

    result = runopf.runopf(case33())

    gap = 100 * (result["f"] - obj) / obj

    print(gap)
    print(max(residual))
