
import numpy as np
from numpy import array, dtype


def case33():
    """ Power flow data for 33 bus, 6 generator case.
    Please see L{caseformat} for details on the case file format.
    Based on data from ...
    Alsac, O. & Stott, B., I{"Optimal Load Flow with Steady State Security"},
    IEEE Transactions on Power Apparatus and Systems, Vol. PAS 93, No. 3,
    1974, pp. 745-751.
    ... with branch parameters rounded to nearest 0.01, shunt values divided
    by 100 and shunt on bus 10 moved to bus 5, load at bus 5 zeroed out.
    Generator locations, costs and limits and bus areas were taken from ...
    Ferrero, R.W., Shahidehpour, S.M., Ramesh, V.C., I{"Transaction analysis
    in deregulated power systems using game theory"}, IEEE Transactions on
    Power Systems, Vol. 12, No. 3, Aug 1997, pp. 1340-1347.
    Generator Q limits were derived from Alsac & Stott, using their Pmax
    capacities. V limits and line |S| limits taken from Alsac & Stott.
    @return: Power flow data for 30 bus, 6 generator case.
    @see: U{http://www.pserc.cornell.edu/matpower/}
    """
    from numpy import ix_
    from pypower.idx_bus import BASE_KV, BUS_I, PQ, PV, REF, BUS_TYPE
    from pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
    from pypower.idx_gen import GEN_BUS

    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [2, 1, 0.1, 0.06, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [3, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [4, 1, 0.12, 0.08, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [5, 1, 0.06, 0.03, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [6, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [7, 1, 0.2, 0.1, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [8, 1, 0.2, 0.1, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [9, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [10, 1, 0.06, 0.02, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [11, 1, 0.045, 0.03, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [12, 1, 0.06, 0.035, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [13, 1, 0.06, 0.035, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [14, 1, 0.12, 0.08, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [15, 1, 0.06, 0.01, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [16, 1, 0.06, 0.02, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [17, 1, 0.06, 0.02, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [18, 1, 0.09, 0.04, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [19, 1, 0.09, 0.04, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [20, 1, 0.09, 0.04, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [21, 1, 0.09, 0.04, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [22, 1, 0.09, 0.04, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [23, 1, 0.09, 0.05, 0, 0, 2, 1, 0, 12.66, 1, 1.05, 0.95],
        [24, 1, 0.42, 0.20, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [25, 1, 0.42, 0.2, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [26, 1, 0.06, 0.025, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [27, 1, 0.06, 0.025, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [28, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.05, 0.95],
        [29, 1, 0.12, 0.07, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [30, 1, 0.2, 0.6, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [31, 1, 0.15, 0.07, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [32, 1, 0.21, 0.1, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
        [33, 1, 0.06, 0.04, 0, 0, 3, 1, 0, 12.66, 1, 1.05, 0.95],
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    # the gen information is only of DS substation, microgrid will be added to
    # DS later
    ppc["gen"] = array([
         [1, 0, 0, 10, -10, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                     0, 0, 0]
        # [21, 0, 0, 0.35, -0.35, 1, 100, 1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                                                             0, 0, 0],
        # [25, 0, 0, 0.30, -0.30, 1, 100, 1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #                                                             0, 0, 0],

    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin,
    # angmax
    # Note that r, x, b are nominal value here and will later be converted
    # to per unit value.
    ppc["branch"] = array([
        [1, 2, 0.0922, 0.0470, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 3, 0.4930, 0.2511, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 4, 0.3660, 0.1864, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.3811, 0.1941, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [5, 6, 0.8190, 0.7070, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [6, 7, 0.1872, 0.6188, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [7, 8, 0.7114, 0.2351, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [8, 9, 1.0300, 0.7400, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [9, 10, 1.0440, 0.7400, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [10, 11, 0.1966, 0.0650, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [11, 12, 0.3744, 0.1238, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [12, 13, 1.4680, 1.1550, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [13, 14, 0.5416, 0.7129, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [14, 15, 0.5910, 0.5260, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [15, 16, 0.7463, 0.5450, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [16, 17, 1.2890, 1.7210, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [17, 18, 0.7320, 0.5740, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 19, 0.1640, 0.1565, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [19, 20, 1.5042, 1.3554, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [20, 21, 0.4095, 0.4784, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [21, 22, 0.7089, 0.9373, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [3, 23, 0.4512, 0.3083, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [23, 24, 0.8980, 0.7091, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [24, 25, 0.8960, 0.7011, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [6, 26, 0.2030, 0.1034, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [26, 27, 0.2842, 0.1447, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [27, 28, 1.0590, 0.9337, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [28, 29, 0.8042, 0.7006, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [29, 30, 0.5075, 0.2585, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [30, 31, 0.9744, 0.9630, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [31, 32, 0.3105, 0.3619, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        [32, 33, 0.3410, 0.5302, 0, 250, 250, 250, 0, 0, 1, -360, 360],
        # tie lines
        [8, 21, 2.0000, 2.0000, 0, 250, 250, 250, 0, 0, 0, -360, 360],
        [9, 15, 2.0000, 2.0000, 0, 250, 250, 250, 0, 0, 0, -360, 360],
        [12, 22, 2.0000, 2.0000, 0, 250, 250, 250, 0, 0, 0, -360, 360],
        [18, 33, 0.5000, 0.5000, 0, 250, 250, 250, 0, 0, 0, -360, 360],
        [25, 29, 0.5000, 0.5000, 0, 250, 250, 250, 0, 0, 0, -360, 360],
    ])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1, 8],
        [2, 23],
        [3, 26],
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 0, 0, 3, 0, 20, 0]
        # [2, 0, 0, 3, 0, 20, 0],
        # [2, 0, 0, 3, 0, 20, 0]
    ])

    # Update ppc['gen'] considering microgrid
    ssc = sscase()
    n_microgrid = np.sum(ssc['station']['station_type'] == 'microgrid')

    ppc['gen'] = np.tile(ppc['gen'], reps=(n_microgrid, 1))
    # Obtain microgrids' bus no.
    # +1 means the bus no. on map, later it will -1 again.
    ppc['gen'][:, GEN_BUS] = ssc['station'][ssc['station']['station_type']
                                            == 'microgrid']['bus_i'] + 1
    ppc['gencost'] = np.tile(ppc['gencost'], reps=(n_microgrid, 1))

    # Unit conversion
    vbase = ppc['bus'][0, BASE_KV] * 1e3
    sbase = ppc['baseMVA'] * 1e6
    ppc['branch'][:, [BR_R, BR_X]] = ppc['branch'][:, [BR_R, BR_X]] / (
            vbase**2 / sbase)

    # Index conversion to zero-base, if input of bus ID is one-based
    # For bus
    ppc['bus'][:, BUS_I] -= 1
    # For branch
    ppc['branch'][:, [F_BUS, T_BUS]] -= 1
    # For gen
    # if ppc['gen'].size: # CHeck if a array is empty
    ppc['gen'][:, GEN_BUS] -= 1

    # Set generator bus type to type 3 (REF bus) for use in pandapower
    # Only for use in pandapower
    # Set all bus type to PQ bus
    ppc['bus'][:, BUS_TYPE] = PQ
    # Set generator bus to REF bus
    # if ppc['gen'].size:
    ppc['bus'][ppc['gen'][:, GEN_BUS].astype(int), BUS_TYPE] = REF

    # Check if gen is matched with gencost
    if not ppc['gen'].shape[0] == ppc['gencost'].shape[0]:
        raise Exception("ppc['gen'] is not matched with ppc['gencost']")

    return ppc


def siouxfalls():
    ''' Transportation systems case
    https: // github.com / bstabler / TransportationNetworks
    :return:
    '''

    from numpy import array

    tsc = {}

    # Transportaton system case
    # node data (??? The coordinates of x, y)
    data_type = dtype({'names': ('node_i', 'x', 'y'),
                'formats': ('i4', 'f8', 'f8')
                })

    tsc['node'] = array([
        (1, 50000, 510000),
        (2, 320000, 510000),
        (3, 50000, 440000),
        (4, 130000, 440000),
        (5, 220000, 440000),
        (6, 320000, 440000),
        (7, 420000, 380000),
        (8, 320000, 380000),
        (9, 220000, 380000),
        (10, 220000, 320000),
        (11, 130000, 320000),
        (12, 50000, 320000),
        (13, 50000, 50000),
        (14, 130000, 190000),
        (15, 220000, 190000),
        (16, 320000, 320000),
        (17, 320000, 260000),
        (18, 420000, 320000),
        (19, 320000, 190000),
        (20, 320000, 50000),
        (21, 220000, 50000),
        (22, 220000, 130000),
        (23, 130000, 130000),
        (24, 130000, 50000),
    ], dtype=data_type)

    # Init node Term node 	Capacity 	Length 	Free Flow Time 	B	Power
    # Speed limit 	Toll 	Type	;
    # INIT_NODE = 0
    # TERM_NODE = 1
    data_type = dtype({'names': ('init_node', 'term_node', 'capacity', 'length',
                           'free_flow_time', 'b', 'power', 'speed_limit',
                           'toll', 'type'),
                'formats': ('i4', 'i4', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                            'f8', 'i4')
                })
    tsc['edge'] = array([
        (1, 2, 25900.20064, 6, 6, 0.15, 4, 0, 0, 1),
        (1, 3, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (2, 1, 25900.20064, 6, 6, 0.15, 4, 0, 0, 1),
        (2, 6, 4958.180928, 5, 5, 0.15, 4, 0, 0, 1),
        (3, 1, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (3, 4, 17110.52372, 4, 4, 0.15, 4, 0, 0, 1),
        (3, 12, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (4, 3, 17110.52372, 4, 4, 0.15, 4, 0, 0, 1),
        (4, 5, 17782.7941, 2, 2, 0.15, 4, 0, 0, 1),
        (4, 11, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1),
        (5, 4, 17782.7941, 2, 2, 0.15, 4, 0, 0, 1),
        (5, 6, 4947.995469, 4, 4, 0.15, 4, 0, 0, 1),
        (5, 9, 10000, 5, 5, 0.15, 4, 0, 0, 1),
        (6, 2, 4958.180928, 5, 5, 0.15, 4, 0, 0, 1),
        (6, 5, 4947.995469, 4, 4, 0.15, 4, 0, 0, 1),
        (6, 8, 4898.587646, 2, 2, 0.15, 4, 0, 0, 1),
        (7, 8, 7841.81131, 3, 3, 0.15, 4, 0, 0, 1),
        (7, 18, 23403.47319, 2, 2, 0.15, 4, 0, 0, 1),
        (8, 6, 4898.587646, 2, 2, 0.15, 4, 0, 0, 1),
        (8, 7, 7841.81131, 3, 3, 0.15, 4, 0, 0, 1),
        (8, 9, 5050.193156, 10, 10, 0.15, 4, 0, 0, 1),
        (8, 16, 5045.822583, 5, 5, 0.15, 4, 0, 0, 1),
        (9, 5, 10000, 5, 5, 0.15, 4, 0, 0, 1),
        (9, 8, 5050.193156, 10, 10, 0.15, 4, 0, 0, 1),
        (9, 10, 13915.78842, 3, 3, 0.15, 4, 0, 0, 1),
        (10, 9, 13915.78842, 3, 3, 0.15, 4, 0, 0, 1),
        (10, 11, 10000, 5, 5, 0.15, 4, 0, 0, 1),
        (10, 15, 13512.00155, 6, 6, 0.15, 4, 0, 0, 1),
        (10, 16, 4854.917717, 4, 4, 0.15, 4, 0, 0, 1),
        (10, 17, 4993.510694, 8, 8, 0.15, 4, 0, 0, 1),
        (11, 4, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1),
        (11, 10, 10000, 5, 5, 0.15, 4, 0, 0, 1),
        (11, 12, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1),
        (11, 14, 4876.508287, 4, 4, 0.15, 4, 0, 0, 1),
        (12, 3, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (12, 11, 4908.82673, 6, 6, 0.15, 4, 0, 0, 1),
        (12, 13, 25900.20064, 3, 3, 0.15, 4, 0, 0, 1),
        (13, 12, 25900.20064, 3, 3, 0.15, 4, 0, 0, 1),
        (13, 24, 5091.256152, 4, 4, 0.15, 4, 0, 0, 1),
        (14, 11, 4876.508287, 4, 4, 0.15, 4, 0, 0, 1),
        (14, 15, 5127.526119, 5, 5, 0.15, 4, 0, 0, 1),
        (14, 23, 4924.790605, 4, 4, 0.15, 4, 0, 0, 1),
        (15, 10, 13512.00155, 6, 6, 0.15, 4, 0, 0, 1),
        (15, 14, 5127.526119, 5, 5, 0.15, 4, 0, 0, 1),
        (15, 19, 14564.75315, 3, 3, 0.15, 4, 0, 0, 1),
        (15, 22, 9599.180565, 3, 3, 0.15, 4, 0, 0, 1),
        (16, 8, 5045.822583, 5, 5, 0.15, 4, 0, 0, 1),
        (16, 10, 4854.917717, 4, 4, 0.15, 4, 0, 0, 1),
        (16, 17, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1),
        (16, 18, 19679.89671, 3, 3, 0.15, 4, 0, 0, 1),
        (17, 10, 4993.510694, 8, 8, 0.15, 4, 0, 0, 1),
        (17, 16, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1),
        (17, 19, 4823.950831, 2, 2, 0.15, 4, 0, 0, 1),
        (18, 7, 23403.47319, 2, 2, 0.15, 4, 0, 0, 1),
        (18, 16, 19679.89671, 3, 3, 0.15, 4, 0, 0, 1),
        (18, 20, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (19, 15, 14564.75315, 3, 3, 0.15, 4, 0, 0, 1),
        (19, 17, 4823.950831, 2, 2, 0.15, 4, 0, 0, 1),
        (19, 20, 5002.607563, 4, 4, 0.15, 4, 0, 0, 1),
        (20, 18, 23403.47319, 4, 4, 0.15, 4, 0, 0, 1),
        (20, 19, 5002.607563, 4, 4, 0.15, 4, 0, 0, 1),
        (20, 21, 5059.91234, 6, 6, 0.15, 4, 0, 0, 1),
        (20, 22, 5075.697193, 5, 5, 0.15, 4, 0, 0, 1),
        (21, 20, 5059.91234, 6, 6, 0.15, 4, 0, 0, 1),
        (21, 22, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1),
        (21, 24, 4885.357564, 3, 3, 0.15, 4, 0, 0, 1),
        (22, 15, 9599.180565, 3, 3, 0.15, 4, 0, 0, 1),
        (22, 20, 5075.697193, 5, 5, 0.15, 4, 0, 0, 1),
        (22, 21, 5229.910063, 2, 2, 0.15, 4, 0, 0, 1),
        (22, 23, 5000, 4, 4, 0.15, 4, 0, 0, 1),
        (23, 14, 4924.790605, 4, 4, 0.15, 4, 0, 0, 1),
        (23, 22, 5000, 4, 4, 0.15, 4, 0, 0, 1),
        (23, 24, 5078.508436, 2, 2, 0.15, 4, 0, 0, 1),
        (24, 13, 5091.256152, 4, 4, 0.15, 4, 0, 0, 1),
        (24, 21, 4885.357564, 3, 3, 0.15, 4, 0, 0, 1),
        (24, 23, 5078.508436, 2, 2, 0.15, 4, 0, 0, 1),
    ], dtype=data_type)

    # Index conversion to zero-base, if input of bus ID is one-based
    # For bus
    tsc['node']['node_i'] -= 1
    # For branch
    tsc['edge']['init_node'] -= 1
    tsc['edge']['term_node'] -= 1

    return tsc


def sscase():
    '''

    :return:
    '''

    from numpy import array, dtype

    ssc = {}

    # microgrid information
    data_type = dtype({'names': ('station_type', 'bus_i', 'node_i', 'max_p_kw',
                        'min_p_kw', 'max_q_kvar', 'min_q_kvar', 'cap_e_kwh',
                        'min_r_kwh', 'cost_$/kwh', 'pload_kw', 'qload_kvar',
                        'load_type'),
               'formats': ('U16', 'i4', 'i4', 'f8',
                           'f8', 'f8', 'f8', 'f8',
                           'f8', 'f8', 'f8', 'f8',
                           'U16')
               })

    ssc['station'] = array([
        ('microgrid', 14, 24, 1600, 0, 1600, -1600, 23040, 2304, 0.5, 500, 300,
                                                                'commercial'),
        ('depot', -1, 8, 1600, 0, 1600, -1600, 23040, 2304, 0.5, 500, 300,
                                                                    'no_load'),
        ('depot', -1, 2, 1600, 0, 1600, -1600, 23040, 2304, 0.5, 500, 300,
                                                                    'no_load'),
        ('microgrid', 21, 11, 1600, 0, 1600, -1600, 23040, 2304, 0.5, 500, 300,
                                                                'residential'),
        ('microgrid', 25, 15, 1600, 0, 1600, -1600, 23040, 2304, 0.5, 500, 300,
                                                                'industrial'),
    ], dtype=data_type)

    # Index conversion to zero-base, if input of bus ID is one-based
    # For location
    # It cannot operate ['bus_i', 'node_i'] at the same time
    ssc['station']['bus_i'] -= 1
    ssc['station']['node_i'] -= 1

    return ssc


def tesscase():

    tessc = {}

    # tess information
    data_type = dtype({'names': ('init_location', 'ch_p_kw',
                        'dch_p_kw', 'cap_e_kwh',
                        'init_soc', 'max_soc', 'min_soc', 'ch_efficiency',
                        'dch_efficiency', 'avg_v_km/h',
                        'cost_power', 'cost_transportation'),
               'formats': ('i4', 'f8', 'f8', 'f8',
                           'f8', 'f8', 'f8', 'f8',
                           'f8', 'f8', 'f8', 'f8')
               })

    tessc['tess'] = array([
        (8, 200, 200, 1000, 0.5, 0.9, 0.1, 0.95, 0.95, 5, 0.20, 80),
        (8, 200, 200, 1000, 0.5, 0.9, 0.1, 0.95, 0.95, 5, 0.21, 81),
        (8, 200, 200, 1000, 0.5, 0.9, 0.1, 0.95, 0.95, 5, 0.22, 82),
        (8, 200, 200, 1000, 0.5, 0.9, 0.1, 0.95, 0.95, 5, 0.23, 83),
    ], dtype=data_type)

    # Index conversion to zero-base, if input of bus ID is one-based
    # For initial location
    tessc['tess']['init_location'] -= 1

    # To check if the tess is initially parked at a depot
    ssc = sscase()
    # tess's initial location
    tess_initial_location = tessc['tess']['init_location']
    # tess's initial location
    # selection in structure array
    depot_location = ssc['station'][
        ssc['station']['station_type'] == 'depot']['node_i']

    if not set(tess_initial_location).issubset(depot_location):
        raise Exception('tess is not initially at depot, please double check')

    return tessc



if __name__ == "__main__":
    ppc = case33()
    tsc = siouxfalls()
    mgc = sscase()
    tessc = tesscase()
    print('all')
