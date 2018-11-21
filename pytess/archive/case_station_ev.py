from numpy import array

from pypower.idx_bus import BUS_I
from pypower.idx_brch import F_BUS, T_BUS
from pypower.idx_gen import GEN_BUS

def case_station_ev():
    ppc = {"version": 2}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 1, 0, 0, 0, 0, 1, 1.05, 0, 10, 1, 1.05, 0.95],
        [2, 1, 0, 0, 0, 0, 1, 1.05, 0, 10, 1, 1.05, 0.95],
        [3, 1, 0, 0, 0, 0, 1, 1.05, 0, 10, 1, 1.05, 0.95],
        [4, 0, 0, 0, 0, 0, 1, 1.05, 0, 10, 1, 1.05, 0.95]
    ])

    # generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc['gen'] = array([
        [6, 0, 0, 0.20, -0.20, 1, 100, 1, 0.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [26, 0, 0, 0.35, -0.35, 1, 100, 1, 0.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [26, 0, 0, 0.30, -0.30, 1, 100, 1, 0.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [29, 0, 0, 0.40, -0.40, 1, 100, 1, 0.20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 1, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 2, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [1, 4, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 1, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 2, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 3, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 2, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 3, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 1, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 3, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        # [2, 0, 0, 3, 0.01, 40, 0],
        # [2, 0, 0, 3, 0.01, 40, 0],
        # [2, 0, 0, 3, 0.01, 40, 0],
        # [2, 0, 0, 3, 0.01, 40, 0],
        # [2, 0, 0, 3, 0.01, 40, 0],
    ])

    # index conversion to zero-base, if input of bus ID is one-based
    # For bus
    ppc['bus'][:, BUS_I] -= 1
    # For branch
    ppc['branch'][:, [F_BUS, T_BUS]] -= 1
    # For gen
    ppc['gen'][:, GEN_BUS] -= 1

    return ppc

