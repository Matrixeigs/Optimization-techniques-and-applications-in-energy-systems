"""
IEEE-24 bus test systems
"""
from numpy import array


def case14():
    """Power flow data for real wind hydro power systems

    @return: Power flow data for jointed wind hydro power systems
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 2, 108, 22, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [2, 2, 97, 20, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [3, 1, 180, 37, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [4, 1, 74, 15, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [5, 1, 71, 14, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [6, 1, 136, 28, 0, -100, 2, 1, 0, 138, 1, 1.05, 0.95],
        [7, 2, 125, 25, 0, 0, 2, 1, 0, 138, 1, 1.05, 0.95],
        [8, 1, 171, 35, 0, 0, 2, 1, 0, 138, 1, 1.05, 0.95],
        [9, 1, 175, 36, 0, 0, 1, 1, 0, 138, 1, 1.05, 0.95],
        [10, 1, 195, 40, 0, 0, 2, 1, 0, 138, 1, 1.05, 0.95],
        [11, 1, 0, 0, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [12, 1, 0, 0, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [13, 3, 265, 54, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [14, 2, 194, 39, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [15, 2, 317, 64, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [16, 2, 100, 20, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [17, 1, 0, 0, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [18, 2, 333, 68, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [19, 1, 181, 37, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [20, 1, 128, 26, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [21, 2, 0, 0, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [22, 2, 0, 0, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95],
        [23, 2, 0, 0, 0, 0, 3, 1, 0, 230, 1, 1.05, 0.95],
        [24, 1, 0, 0, 0, 0, 4, 1, 0, 230, 1, 1.05, 0.95]
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, 2, 0.0026, 0.0139, 0.4611, 175, 250, 200, 0, 0, 1, -360, 360],
        [1, 3, 0.0546, 0.2112, 0.0572, 175, 208, 220, 0, 0, 1, -360, 360],
        [1, 5, 0.0218, 0.0845, 0.0229, 175, 208, 220, 0, 0, 1, -360, 360],
        [2, 4, 0.0328, 0.1267, 0.0343, 175, 208, 220, 0, 0, 1, -360, 360],
        [2, 6, 0.0497, 0.192, 0.052, 175, 208, 220, 0, 0, 1, -360, 360],
        [3, 9, 0.0308, 0.119, 0.0322, 175, 208, 220, 0, 0, 1, -360, 360],
        [3, 24, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [4, 9, 0.0268, 0.1037, 0.0281, 175, 208, 220, 0, 0, 1, -360, 360],
        [5, 10, 0.0228, 0.0883, 0.0239, 175, 208, 220, 0, 0, 1, -360, 360],
        [6, 10, 0.0139, 0.0605, 2.459, 175, 193, 200, 0, 0, 1, -360, 360],
        [7, 8, 0.0159, 0.0614, 0.0166, 175, 208, 220, 0, 0, 1, -360, 360],
        [8, 9, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0, 0, 1, -360, 360],
        [8, 10, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0, 0, 1, -360, 360],
        [9, 11, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [9, 12, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [10, 11, 0.0023, 0.0839, 0, 400, 510, 600, 1.02, 0, 1, -360, 360],
        [10, 12, 0.0023, 0.0839, 0, 400, 510, 600, 1.02, 0, 1, -360, 360],
        [11, 13, 0.0061, 0.0476, 0.0999, 500, 600, 625, 0, 0, 1, -360, 360],
        [11, 14, 0.0054, 0.0418, 0.0879, 500, 625, 625, 0, 0, 1, -360, 360],
        [12, 13, 0.0061, 0.0476, 0.0999, 500, 625, 625, 0, 0, 1, -360, 360],
        [12, 23, 0.0124, 0.0966, 0.203, 500, 625, 625, 0, 0, 1, -360, 360],
        [13, 23, 0.0111, 0.0865, 0.1818, 500, 625, 625, 0, 0, 1, -360, 360],
        [14, 16, 0.005, 0.0389, 0.0818, 500, 625, 625, 0, 0, 1, -360, 360],
        [15, 16, 0.0022, 0.0173, 0.0364, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 21, 0.0063, 0.049, 0.103, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 21, 0.0063, 0.049, 0.103, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 24, 0.0067, 0.0519, 0.1091, 500, 600, 625, 0, 0, 1, -360, 360],
        [16, 17, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [16, 19, 0.003, 0.0231, 0.0485, 500, 600, 625, 0, 0, 1, -360, 360],
        [17, 18, 0.0018, 0.0144, 0.0303, 500, 600, 625, 0, 0, 1, -360, 360],
        [17, 22, 0.0135, 0.1053, 0.2212, 500, 600, 625, 0, 0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0, 0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0, 0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0, 0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0, 0, 1, -360, 360],
        [21, 22, 0.0087, 0.0678, 0.1424, 500, 600, 625, 0, 0, 1, -360, 360]
    ])
    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 1],
        [2, 3],
        [3, 8],
        [4, 6],
    ])
    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = array([
        [1, 2, 0.0026, 0.0139, 0.4611, 175, 250, 200, 0, 0, 1, -360, 360],
        [1, 3, 0.0546, 0.2112, 0.0572, 175, 208, 220, 0, 0, 1, -360, 360],
        [1, 5, 0.0218, 0.0845, 0.0229, 175, 208, 220, 0, 0, 1, -360, 360],
        [2, 4, 0.0328, 0.1267, 0.0343, 175, 208, 220, 0, 0, 1, -360, 360],
        [2, 6, 0.0497, 0.192, 0.052, 175, 208, 220, 0, 0, 1, -360, 360],
        [3, 9, 0.0308, 0.119, 0.0322, 175, 208, 220, 0, 0, 1, -360, 360],
        [3, 24, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [4, 9, 0.0268, 0.1037, 0.0281, 175, 208, 220, 0, 0, 1, -360, 360],
        [5, 10, 0.0228, 0.0883, 0.0239, 175, 208, 220, 0, 0, 1, -360, 360],
        [6, 10, 0.0139, 0.0605, 2.459, 175, 193, 200, 0, 0, 1, -360, 360],
        [7, 8, 0.0159, 0.0614, 0.0166, 175, 208, 220, 0, 0, 1, -360, 360],
        [8, 9, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0, 0, 1, -360, 360],
        [8, 10, 0.0427, 0.1651, 0.0447, 175, 208, 220, 0, 0, 1, -360, 360],
        [9, 11, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [9, 12, 0.0023, 0.0839, 0, 400, 510, 600, 1.03, 0, 1, -360, 360],
        [10, 11, 0.0023, 0.0839, 0, 400, 510, 600, 1.02, 0, 1, -360, 360],
        [10, 12, 0.0023, 0.0839, 0, 400, 510, 600, 1.02, 0, 1, -360, 360],
        [11, 13, 0.0061, 0.0476, 0.0999, 500, 600, 625, 0, 0, 1, -360, 360],
        [11, 14, 0.0054, 0.0418, 0.0879, 500, 625, 625, 0, 0, 1, -360, 360],
        [12, 13, 0.0061, 0.0476, 0.0999, 500, 625, 625, 0, 0, 1, -360, 360],
        [12, 23, 0.0124, 0.0966, 0.203, 500, 625, 625, 0, 0, 1, -360, 360],
        [13, 23, 0.0111, 0.0865, 0.1818, 500, 625, 625, 0, 0, 1, -360, 360],
        [14, 16, 0.005, 0.0389, 0.0818, 500, 625, 625, 0, 0, 1, -360, 360],
        [15, 16, 0.0022, 0.0173, 0.0364, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 21, 0.0063, 0.049, 0.103, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 21, 0.0063, 0.049, 0.103, 500, 600, 625, 0, 0, 1, -360, 360],
        [15, 24, 0.0067, 0.0519, 0.1091, 500, 600, 625, 0, 0, 1, -360, 360],
        [16, 17, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [16, 19, 0.003, 0.0231, 0.0485, 500, 600, 625, 0, 0, 1, -360, 360],
        [17, 18, 0.0018, 0.0144, 0.0303, 500, 600, 625, 0, 0, 1, -360, 360],
        [17, 22, 0.0135, 0.1053, 0.2212, 500, 600, 625, 0, 0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [18, 21, 0.0033, 0.0259, 0.0545, 500, 600, 625, 0, 0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0, 0, 1, -360, 360],
        [19, 20, 0.0051, 0.0396, 0.0833, 500, 600, 625, 0, 0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0, 0, 1, -360, 360],
        [20, 23, 0.0028, 0.0216, 0.0455, 500, 600, 625, 0, 0, 1, -360, 360],
        [21, 22, 0.0087, 0.0678, 0.1424, 500, 600, 625, 0, 0, 1, -360, 360]
    ])

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 1500, 0, 3, 0.01199, 37.5510, 117.7511, 0, 0, -1, 1, 0, 0.508, 1.167, 1, 20, 20, 2, 0.25, 3.0],
        # 1,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.01199, 37.5510, 117.7511, 0, 0, -1, 1, 0, 0.508, 1.167, 1, 20, 20, 2, 0.25, 3.0],
        # 1,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.00876, 13.3272, 81.1364, 3, 2, 3, 2, 1, 0.642, 1.333, 3, 50, 50, 3, 0.93, 1.2],
        # 1,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.00876, 13.3272, 81.1364, 3, 2, 3, 2, 1, 0.642, 1.333, 3, 50, 50, 3, 0.93, 1.2],
        # 1,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.01199, 37.5510, 117.7511, 0, 0, -1, 1, 0, 0.508, 1.167, 1, 20, 20, 2, 0.25, 3.0],
        # 2,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.01199, 37.5510, 117.7511, 0, 0, -1, 1, 0, 0.508, 1.167, 1, 20, 20, 2, 0.25, 3.0],
        # 2,  16,   20,   0,  10, U20
        [2, 1500, 0, 3, 0.00876, 13.3272, 81.1364, 3, 2, 3, 2, 1, 0.642, 1.333, 3, 50, 50, 3, 0.93, 1.2],
        # 2,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.00876, 13.3272, 81.1364, 3, 2, 3, 2, 1, 0.642, 1.333, 3, 50, 50, 3, 0.93, 1.2],
        # 2,  15.2, 76, -25,  30, U76
        [2, 1500, 0, 3, 0.00623, 18, 217.8952, 4, 2, -3, 2, 2, 0.850, 1.233, 3, 70, 70, 4, 0.2, 2.3],
        # 7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.00623, 18, 217.8952, 4, 2, -3, 2, 2, 0.850, 1.233, 3, 70, 70, 4, 0.2, 2.3],
        # 7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.00623, 18, 217.8952, 4, 2, -3, 2, 2, 0.850, 1.233, 3, 70, 70, 4, 0.2, 2.3],
        # 7,  25,  100,   0,  60, U100
        [2, 1500, 0, 3, 0.00259, 23, 259.1310, 5, 4, -4, 4, 2, 0.917, 1.650, 6, 200, 200, 8, 0.2, 2.3],
        # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0.00259, 23, 259.1310, 5, 4, -4, 4, 2, 0.917, 1.650, 6, 200, 200, 8, 0.2, 2.3],
        # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0.00259, 23, 259.1310, 5, 4, -4, 4, 2, 0.917, 1.650, 6, 200, 200, 8, 0.2, 2.3],
        # 13,  69,  197,   0,  80, U197
        [2, 1500, 0, 3, 0.02533, 25.5472, 24.3891, 0, 0, -1, 0, 0, 0.8, 1.00, 0, 0, 0, 1, 0.10, 2.3],  # 14 SynCond
        [2, 1500, 0, 3, 0.02649, 25.6753, 24.4110, 0, 0, -1, 0, 0, 0.8, 1.00, 0, 0, 0, 1, 0.01, 2.3],
        # 15,2.4,12,0,6, U12
        [2, 1500, 0, 3, 0.02649, 25.6753, 24.4110, 0, 0, -1, 0, 0, 0.8, 1.00, 0, 0, 0, 1, 0.01, 2.3],
        # 15,2.4,12,0,6, U12
        [2, 1500, 0, 3, 0.02649, 25.6753, 24.4110, 0, 0, -1, 0, 0, 0.8, 1.00, 0, 0, 0, 1, 0.01, 2.3],
        # 15,2.4,12,0,6, U12
        [2, 1500, 0, 3, 0.02649, 25.6753, 24.4110, 0, 0, -1, 0, 0, 0.8, 1.00, 0, 0, 0, 1, 0.01, 2.3],
        # 15,2.4,12,0,6, U12
        [2, 1500, 0, 3, 0.00473, 10.7154, 143.0288, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.15, 1.2],
        # 15, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.00473, 10.7154, 143.0288, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.15, 1.2],
        # 16, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.00481, 10.7367, 143.3179, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 18, 100,  400, -50, 200, U400
        [2, 1500, 0, 3, 0.00481, 10.7367, 143.3179, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 21, 100,  400, -50, 200, U400
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00487, 10.7583, 142.5972, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.14, 1.2],
        # 22, 10,    50, -10,  16, U50
        [2, 1500, 0, 3, 0.00473, 10.7154, 143.0288, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.15, 1.2],
        # 23, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.00473, 10.7154, 143.0288, 5, 3, 5, 3, 2, 0.917, 1.300, 5, 150, 150, 6, 1.15, 1.2],
        # 23, 54.3, 155, -50,  80, U155
        [2, 1500, 0, 3, 0.00195, 7.5031, 311.9102, 8, 5, 10, 8, 4, 0.842, 1.667, 8, 500, 500, 10, 0, 0.6],
        # 23, 140,  350, -25, 150, U350
    ])

    return ppc
