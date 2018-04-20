"""
Optimal power flow models for hybrid AC/DC microgrids
@author: Tianyang Zhao
@email: zhaoty@ntu.edu.sg
Something should be noted for the hypothesis.

1) The energy losses on the bi-directional converters is modelled simply as used in
[1]Concerted action on computer modeling and simulation
[2]Energy management and operation modelling of hybrid AC–DC microgrid
There are more complex modelling method for different types of converters, see the following references for details.
[1]Mathematical Efficiency Modeling of Static Power Converters
[2]Power Loss Modeling of Isolated AC/DC Converter
The variations on the mathematical modelling result in significant differences in terms of the mathematical property.

2) Even renewable energy sources are assigned with operational cost, e.g., linear in this case.
3) The power losses is ignored in the real-time operation.

@Reference:
[1]
"""

from pypower import runopf
from gurobipy import *
from numpy import zeros, c_, shape, ix_, ones, r_, arange, sum, diag, concatenate, power
from scipy.sparse import csr_matrix as sparse
from scipy.sparse import hstack, vstack, diags
from Two_stage_stochastic_optimization.power_flow_modelling import case33, case_converters
# The following cases, data formats are imported from the Pypower package.
from pypower import case6ww, case9, case30, case118, case300
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN, BUS_I, QD
from pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int


def main(Case_AC=None, Case_DC=None, Converters=None):
    """

    :param Case_AC: AC case
    :param Case_DC: DC case
    :param Converters: Bi-directional converters
    :return: Obtained solutions for hybrid AC DC networks
    """
    # 1) Problem formulation
    model_AC = AC_network_formulation(Case_AC)
    model_DC = DC_network_formulation(Case_DC)
    # 2) Solve the initial problems
    sol_AC = AC_opf_solver(model_AC)
    sol_DC = DC_opf_solver(model_DC)
    # 3) Connect two systems via the BIC networks
    model_converters = BIC_network_formulation(model_AC, model_DC, Converters)
    # 4) Solve the merged functions


def DC_network_formulation(case):
    """
    :param case:
    :return:
    """
    case = ext2int(case)
    baseMVA, bus, gen, branch, gencost = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["gencost"]

    nb = shape(case['bus'])[0]  ## number of buses
    nl = shape(case['branch'])[0]  ## number of branches
    ng = shape(case['gen'])[0]  ## number of dispatchable injections

    f = branch[:, F_BUS]  ## list of "from" buses
    t = branch[:, T_BUS]  ## list of "to" buses
    i = range(nl)  ## double set of row indices

    # Connection matrix
    Cf = sparse((ones(nl), (i, f)), (nl, nb))
    Ct = sparse((ones(nl), (i, t)), (nl, nb))
    Cg = sparse((ones(ng), (gen[:, GEN_BUS], range(ng))), (nb, ng))

    # Modify the branch resistance
    Branch_R = branch[:, BR_R]
    for i in range(nl):
        if Branch_R[i] <= 0:
            Branch_R[i] = max(Branch_R)

    Cf = Cf.T
    Ct = Ct.T
    # Obtain the boundary information
    Slmax = branch[:, RATE_A] / baseMVA
    Pij_l = -Slmax
    Iij_l = zeros(nl)
    Vm_l = power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA

    Pij_u = Slmax
    Iij_u = Slmax
    # Vm_u = [max(turn_to_power(bus[:, VMAX], 2))] * nb
    Vm_u = power(bus[:, VMAX], 2)
    Pg_u = gen[:, PMAX] / baseMVA
    # Pg_l = -Pg_u
    lx = concatenate([Pij_l, Iij_l, Vm_l, Pg_l])
    ux = concatenate([Pij_u, Iij_u, Vm_u, Pg_u])

    # KCL equation
    Aeq_p = hstack([Ct - Cf, -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg])
    beq_p = bus[:, PD] / baseMVA
    # KVL equation
    Aeq_KVL = hstack([-2 * diags(Branch_R), diags(power(Branch_R, 2)), Cf.T - Ct.T, zeros((nl, ng))])
    beq_KVL = zeros(nl)

    Aeq = vstack([Aeq_p, Aeq_KVL])
    Aeq = Aeq.todense()
    beq = concatenate([beq_p, beq_KVL])
    neq = len(beq)

    nx = 2 * nl + nb + ng

    Q = zeros((nx, nx))
    c = zeros(nx)
    c0 = zeros(nx)
    for i in range(ng):
        Q[i + 2 * nl + nb, i + 2 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
        c[i + 2 * nl + nb] = gencost[i, 5] * baseMVA
        c0[i + 2 * nl + nb] = gencost[i, 6]

    model = {"Q": Q,
             "c": c,
             "c0": c0,
             "Aeq": Aeq,
             "beq": beq,
             "lx": lx,
             "ux": ux,
             "nx": nx,
             "nb": nb,
             "nl": nl,
             "ng": ng,
             "f": f}

    return model


def AC_network_formulation(case):
    """

    :param case:
    :return:
    """
    case = ext2int(case)
    baseMVA, bus, gen, branch, gencost = case["baseMVA"], case["bus"], case["gen"], case["branch"], case["gencost"]

    nb = shape(case['bus'])[0]  ## number of buses
    nl = shape(case['branch'])[0]  ## number of branches
    ng = shape(case['gen'])[0]  ## number of dispatchable injections

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
    Vm_l = power(bus[:, VMIN], 2)
    Pg_l = gen[:, PMIN] / baseMVA
    Qg_l = gen[:, QMIN] / baseMVA

    Pij_u = Slmax
    Qij_u = Slmax
    Iij_u = Slmax
    Vm_u = power(bus[:, VMAX], 2)
    Pg_u = 2 * gen[:, PMAX] / baseMVA
    Qg_u = 2 * gen[:, QMAX] / baseMVA

    # Problem formulation
    lx = concatenate([Pij_l, Qij_l, Iij_l, Vm_l, Pg_l, Qg_l])
    ux = concatenate([Pij_u, Qij_u, Iij_u, Vm_u, Pg_u, Qg_u])

    # KCL equation, active power
    Aeq_p = hstack([Ct - Cf, zeros((nb, nl)), -diag(Ct * Branch_R) * Ct, zeros((nb, nb)), Cg, zeros((nb, ng))])
    beq_p = bus[:, PD] / baseMVA

    # KCL equation, reactive power
    Aeq_q = hstack([zeros((nb, nl)), Ct - Cf, -diag(Ct * Branch_X) * Ct, zeros((nb, nb)), zeros((nb, ng)), Cg])
    beq_q = bus[:, QD] / baseMVA

    # KVL equation
    Aeq_KVL = hstack([-2 * diags(Branch_R), -2 * diags(Branch_X),
                      diags(power(Branch_R, 2)) + diags(power(Branch_X, 2)), Cf.T - Ct.T,
                      zeros((nl, 2 * ng))])
    beq_KVL = zeros(nl)

    Aeq = vstack([Aeq_p, Aeq_q, Aeq_KVL])
    Aeq = Aeq.todense()
    beq = concatenate([beq_p, beq_q, beq_KVL])
    neq = len(beq)
    nx = 3 * nl + nb + 2 * ng

    Q = zeros((nx, nx))
    c = zeros(nx)
    c0 = zeros(nx)
    for i in range(ng):
        Q[i + 3 * nl + nb, i + 3 * nl + nb] = gencost[i, 4] * baseMVA * baseMVA
        c[i + 3 * nl + nb] = gencost[i, 5] * baseMVA
        c0[i + 3 * nl + nb] = gencost[i, 6]

    model = {"Q": Q,
             "c": c,
             "c0": c0,
             "Aeq": Aeq,
             "beq": beq,
             "lx": lx,
             "ux": ux,
             "nx": nx,
             "nb": nb,
             "nl": nl,
             "ng": ng,
             "f": f}

    return model


def AC_opf_solver(case):
    """
    Optimal power flow solver for AC networks
    :param model:
    :return: AC OPF solution
    """
    nl = case["nl"]
    nb = case["nb"]
    ng = case["ng"]
    f = case["f"]
    nx = case["nx"]
    lx = case["lx"]
    ux = case["ux"]
    Aeq = case["Aeq"]
    beq = case["beq"]
    neq = len(beq)

    Q = case["Q"]
    c = case["c"]
    c0 = case["c0"]

    model = Model("OPF")
    # Define the decision variables
    x = {}

    for i in range(nx):
        x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
    for i in range(neq):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    for i in range(nl):
        model.addConstr(x[i] * x[i] + x[i + nl] * x[i + nl] <= x[i + 2 * nl] * x[f[i] + 3 * nl])

    obj = 0
    for i in range(nx):
        obj += Q[i, i] * x[i] * x[i] + c[i] * x[i] + c0[i]

    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.optimize()

    xx = []
    for v in model.getVars():
        xx.append(v.x)

    obj = obj.getValue()

    Pij = xx[0:nl]
    Qij = xx[nl + 0:2 * nl]
    Iij = xx[2 * nl:3 * nl]
    Vi = xx[3 * nl:3 * nl + nb]
    Pg = xx[3 * nl + nb:3 * nl + nb + ng]
    Qg = xx[3 * nl + nb + ng:3 * nl + nb + 2 * ng]

    primal_residual = zeros(nl)
    for i in range(nl):
        primal_residual[i] = Pij[i] * Pij[i] + Qij[i] * Qij[i] - Iij[i] * Vi[int(f[i])]

    sol = {"Pij": Pij,
           "Qij": Qij,
           "Iij": Iij,
           "Vm": power(Vi, 0.5),
           "Pg": Pg,
           "Qg": Qg,
           "obj": obj}

    return sol, primal_residual


def DC_opf_solver(case):
    """
    Optimal power flow solver for DC networks
    :param model:
    :return: DC OPF solution
    """
    nl = case["nl"]
    nb = case["nb"]
    ng = case["ng"]
    f = case["f"]
    nx = case["nx"]
    lx = case["lx"]
    ux = case["ux"]
    Aeq = case["Aeq"]
    beq = case["beq"]
    neq = len(beq)

    Q = case["Q"]
    c = case["c"]
    c0 = case["c0"]

    model = Model("OPF_DC")
    # Define the decision variables
    x = {}
    for i in range(nx):
        x[i] = model.addVar(lb=lx[i], ub=ux[i], vtype=GRB.CONTINUOUS)
    for i in range(neq):
        expr = 0
        for j in range(nx):
            expr += x[j] * Aeq[i, j]
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=beq[i])

    for i in range(nl):
        model.addConstr(x[i] * x[i] <= x[i + nl] * x[f[i] + 2 * nl])

    obj = 0
    for i in range(nx):
        obj += Q[i, i] * x[i] * x[i] + c[i] * x[i] + c0[i]

    model.setObjective(obj)
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0
    model.Params.DisplayInterval = 1
    model.optimize()

    xx = []
    for v in model.getVars():
        xx.append(v.x)

    obj = obj.getValue()

    Pij = xx[0:nl]
    Iij = xx[nl:2 * nl]
    Vi = xx[2 * nl:2 * nl + nb]
    Pg = xx[2 * nl + nb:2 * nl + nb + ng]

    primal_residual = zeros(nl)
    for i in range(nl):
        primal_residual[i] = Pij[i] * Pij[i] - Iij[i] * Vi[int(f[i])]

    sol = {"Pij": Pij,
           "Iij": Iij,
           "Vm": power(Vi, 0.5),
           "Pg": Pg,
           "obj": obj}

    return sol, primal_residual


def BIC_network_formulation(case_AC, case_DC, case_BIC):
    """
    Merger the AC network and DC networks
    :param case_AC:
    :param case_DC:
    :param case_BIC:
    :return:
    """


if __name__ == '__main__':
    # A test hybrid AC DC network is connected via BIC networks
    caseAC = case33.case33()
    caseDC = case30.case30()
    converters = case_converters.con()

    main(Case_AC=caseAC, Case_DC=caseDC, Converters=converters)
