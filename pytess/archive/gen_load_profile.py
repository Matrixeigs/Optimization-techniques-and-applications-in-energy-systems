from numpy import array, arange, zeros, interp, nonzero, tile, unique

def gen_load_profile_ppc(bus, LOAD_TYPE):
    """
    :param bus: array that indicates base load
    :param LOAD_TYPE: column index of LOAD_TYPE
    :return: loadProfile: 1-D array, time series of load profile.
    """
    index_interval = arange(24)
    n_interval = index_interval.shape[0]

    # Industiral load profile (original data)
    industrial_load = zeros((2, 24))
    industrial_load[0, :] = arange(24)
    industrial_load[1, :] = array([50, 38, 27, 27, 26, 24, 22, 42,
                                  45, 60, 52, 50, 50, 48, 58, 50,
                                  62, 57, 82, 84, 96, 100, 87, 60])

    industrial_load = interp(index_interval, industrial_load[0, :], industrial_load[1, :]) / 100
    industrial_load = industrial_load * 1

    # Commercial load profile (original data)
    commercial_load = zeros((2, 24))
    commercial_load[0, :] = arange(24)
    commercial_load[1, :] = array([28, 27, 28, 27, 29, 30, 35, 70,
                                  85, 96, 96, 96, 90, 84, 80, 60,
                                  50, 45, 40, 40, 40, 40, 40, 35])
    commercial_load = interp(index_interval, commercial_load[0, :], commercial_load[1, :]) / 100
    commercial_load = commercial_load * 1

    # Residential load profile (original data)
    residential_load = zeros((2, 24))
    residential_load[0, :] = arange(24)
    residential_load[1, :] = array([32, 30, 28, 26, 26, 25, 25, 25,
                                   22, 23, 75, 82, 88, 78, 55, 40,
                                   65, 80, 95, 100, 90, 65, 45, 35])
    residential_load = interp(index_interval, residential_load[0, :], residential_load[1, :]) / 100

    # Generate load profile
    p_load_profile = zeros((bus.shape[0], index_interval.shape[0]))
    q_load_profile = zeros((bus.shape[0], index_interval.shape[0]))

    # For industrial load
    index_industrial_load = nonzero(bus[:, LOAD_TYPE] == 1)[0]
    p_load_profile[index_industrial_load, :] = tile(industrial_load, (index_industrial_load.shape[0], 1))
    q_load_profile[index_industrial_load, :] = tile(industrial_load, (index_industrial_load.shape[0], 1))

    # For commercial load
    index_commercial_load = nonzero(bus[:, LOAD_TYPE] == 2)[0]
    p_load_profile[index_commercial_load, :] = tile(commercial_load, (index_commercial_load.shape[0], 1))
    q_load_profile[index_commercial_load, :] = tile(commercial_load, (index_commercial_load.shape[0], 1))

    # For residential load
    index_residential_load = nonzero(bus[:, LOAD_TYPE] == 3)[0]
    p_load_profile[index_residential_load, :] = tile(residential_load, (index_residential_load.shape[0], 1))
    q_load_profile[index_residential_load, :] = tile(residential_load, (index_residential_load.shape[0], 1))

    # For non-load
    index_non_load = nonzero(bus[:, LOAD_TYPE] == 0)[0]
    p_load_profile[index_non_load, :] = 0
    q_load_profile[index_non_load, :] = 0

    # consider base load PD, QD

    return p_load_profile, q_load_profile


def gen_load_profile_net(ppnet, LOAD_TYPE='load_type'):
    """
    :param ppnet: DataFrame that indicates base load
    :param LOAD_TYPE: column name of LOAD_TYPE
    :return: pload_profile: array (nd, n_interval), time series of load profile (nd after deletion).
    """

    load = ppnet.load
    index_interval = arange(24)
    n_interval = index_interval.shape[0]
    load_profile_reference = {}


    # Industiral load profile (original data)
    industrial_load = zeros((2, 24))
    industrial_load[0, :] = arange(24)
    industrial_load[1, :] = array([50, 38, 27, 27, 26, 24, 22, 42,
                                  45, 60, 52, 50, 50, 48, 58, 50,
                                  62, 57, 82, 84, 96, 100, 87, 60])

    industrial_load = interp(index_interval, industrial_load[0, :], industrial_load[1, :]) / 100
    industrial_load = industrial_load * 1
    load_profile_reference['industrial'] = industrial_load  # (n_interval,)

    # Commercial load profile (original data)
    commercial_load = zeros((2, 24))
    commercial_load[0, :] = arange(24)
    commercial_load[1, :] = array([28, 27, 28, 27, 29, 30, 35, 70,
                                  85, 96, 96, 96, 90, 84, 80, 60,
                                  50, 45, 40, 40, 40, 40, 40, 35])
    commercial_load = interp(index_interval, commercial_load[0, :], commercial_load[1, :]) / 100
    commercial_load = commercial_load * 1
    load_profile_reference['commercial'] = commercial_load  # (n_interval,)

    # Residential load profile (original data)
    residential_load = zeros((2, 24))
    residential_load[0, :] = arange(24)
    residential_load[1, :] = array([32, 30, 28, 26, 26, 25, 25, 25,
                                   22, 23, 75, 82, 88, 78, 55, 40,
                                   65, 80, 95, 100, 90, 65, 45, 35])
    residential_load = interp(index_interval, residential_load[0, :], residential_load[1, :]) / 100
    load_profile_reference['residential'] = residential_load  # (n_interval,)

    # Generate load profile -------------------------------------------------------------------------------------------
    p_load_profile = zeros((ppnet.load.shape[0], index_interval.shape[0]))
    # p_load_profile_1 = zeros((ppnet.load.shape[0], index_interval.shape[0]))
    q_load_profile = zeros((ppnet.load.shape[0], index_interval.shape[0]))

    for e_load_type in unique(load[LOAD_TYPE]):
        n_e_load_type = (load[LOAD_TYPE] == e_load_type).nonzero()[0].size  # the number of loads for each type
        p_load_profile[load[LOAD_TYPE] == e_load_type, :] = tile(load_profile_reference[e_load_type],
                                                                 reps=(n_e_load_type, 1))
        q_load_profile[load[LOAD_TYPE] == e_load_type, :] = tile(load_profile_reference[e_load_type],
                                                                 reps=(n_e_load_type, 1))

    return p_load_profile, q_load_profile
