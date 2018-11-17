"""
Units data for wind hydro dispatch
Data is obtained from real test systems

"""
from numpy import array

# Transmission line data format
F_BUS = 0
T_BUS = 1
BR_X = 3
RATE_A = 5

# Generator format
GEN_BUS = 0
COST_C = 1
COST_B = 2
COST_A = 3
PG_MAX = 4
PG_MIN = 5
I0 = 6
MIN_DOWN = 7
MIN_UP = 8
RUG = 9
RDG = 10
COLD_START = 11

# Bus format
BUS_ID = 0
PD = 13


def case6modified():
    """Power flow data for real wind hydro power systems

    @return: Power flow data for jointed wind hydro power systems
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin load_distribution
    ppc["bus"] = array([
        [1, 3, 20, 23, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95, 0.00],
        [2, 2, 40, 38.5, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95, 0.00],
        [3, 2, 40, 38.5, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95, 0.20],
        [4, 2, 0, 0, 0, 0, 1, 1, 0, 220, 1, 1.05, 0.95, 0.40],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95, 0.40],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, 110, 1, 1.05, 0.95, 0.00],
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    # ppc["gen"] = array([
    #     [1, 177, 13.5, 0.00045, 220, 100, 0, 0, 200, -80, 1, 100, 1, 220, 100, 0, 0, 0, 0, 0, 0, 55 / 12, 55 / 6, 55, 0, 0],
    #     [2, 130, 40, 0.001, 100, 100, 10, 0, 70, -40, 1, 100, 1, 100, 10, 0, 0, 0, 0, 0, 0, 50 / 12, 50 / 6, 50, 0, 0],
    #     [6, 137, 17.7, 0.005, 40, 100, 10, 0, 50, -40, 1, 100, 1, 20, 10, 0, 0, 0, 0, 0, 0, 20 / 12, 20 / 6, 20, 0, 0],
    # ])
    # ppc["gen"] = array([
    #     [1, 177, 13.5, 0.00045, 220, 100, 1, 4, 4, 55, 55, 100],
    #     [2, 130, 40, 0.001, 100, 10, 1, 2, 3, 20, 20, 200],
    #     [6, 137, 17.7, 0.005, 40, 10, 1, 1, 1, 20, 20, 0],
    # ])
    ppc["gen"] = array([
        [1, 177, 13.5, 0.00045, 220, 100, 1, 1, 1, 55, 55, 100],
        [2, 130, 40, 0.001, 100, 10, 1, 1, 1, 20, 20, 200],
        [6, 137, 17.7, 0.005, 40, 10, 1, 1, 1, 20, 20, 0],
    ])
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2, 0.0050, 0.170, 0.00, 200, 300, 300, 0, 0, 1, -360, 360],
        [2, 3, 0.0000, 0.037, 0.00, 110, 300, 300, 0, 0, 1, -360, 360],
        [1, 4, 0.0030, 0.258, 0.00, 100, 300, 300, 0, 0, 1, -360, 360],
        [2, 4, 0.0070, 0.197, 0.00, 100, 300, 300, 0.69, 0, 1, -360, 360],
        [4, 5, 0.0000, 0.037, 0.00, 100, 300, 300, 0.69, 0, 1, -360, 360],
        [5, 6, 0.0020, 0.140, 0.00, 100, 130, 130, 0, 0, 1, -360, 360],
        [3, 6, 0.0000, 0.018, 0.00, 100, 130, 130, 0, 0, 0, -360, 360],
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    # ppc["gencost"] = array([
    #     [2, 10, 0, 3, 0.0004, 13.51476154, 176.9507820, 4, 4, 4],
    #     [2, 200, 0, 3, 0.001, 32.63061346, 129.9709568, 2, 3, 3],
    #     [2, 100, 0, 3, 0.005, 17.69711347, 137.4120219, 1, 1, 0],
    # ])

    ppc["Load_profile"] = array([166.4,
                                 156,
                                 150.8,
                                 145.6,
                                 145.6,
                                 150.8,
                                 166.4,
                                 197.6,
                                 226.2,
                                 247,
                                 257.4,
                                 260,
                                 257.4,
                                 260,
                                 260,
                                 252.2,
                                 249.6,
                                 249.6,
                                 241.8,
                                 239.2,
                                 239.2,
                                 241.8,
                                 226.2,
                                 187.2,
                                 5412])
    return ppc
