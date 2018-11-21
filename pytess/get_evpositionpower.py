from numpy import arange, zeros, concatenate, nonzero, delete, vstack

from pypower.loadcase import loadcase
from pypower.idx_bus import BUS_TYPE

from archive.case_station_ev import case_station_ev

def get_evpositionpower(ev_position, res):
    """

    :param ev_position:
    :param res:
    :return: ev_positionpower
    """
    # 2018.08.04

    station_ev_data = loadcase(case_station_ev())
    # get parameters
    n_ev = station_ev_data['gen'].shape[0]
    n_timespan = res['ev_arc_x'].shape[1]
    n_timepoint = n_timespan + 1
    index_station_virtual = nonzero(station_ev_data['bus'][:, BUS_TYPE] == 0)[0]

    # choosing sequence is for drawing figure in OriginlabPro
    sequence_time = arange(2*n_timepoint) // 2  # if n_timespan = 12, array([0, 0, 1, 1, 2, 2, ..., 11, 11, 12, 12])
    sequence_position = sequence_time  # if n_timespan = 12, array([0, 0, 1, 1, 2, 2, ..., 11, 11, 12, 12])
    sequence_power = sequence_time[:-2]  # if n_timespan = 12, array([0, 0, 1, 1, 2, 2, ..., 10, 10, 11, 11])

    # since each ev's array shape might be different, it's better to use a list of 2-D array rather than a 3-D array
    ev_positionpower = []

    for i_ev in range(n_ev):
        ev_positionpower.append(vstack([sequence_time,  # X axis, time point
                                                    ev_position[i_ev, sequence_position],  # Y axis: ev's position
                                                    # Z axis: output power, add 0 as the first and last element
                                                    # np.append(0, res['pev_x']), it needs to append twice
                                                    concatenate([zeros(1), res['pev_x'][i_ev, sequence_power],
                                                                 zeros(1)])])
                                )  # a list of 2-D array with (3, 2*n_timepoint)
        # find out the time point at which the ev is at virtual station
        index_deletecol = nonzero( ev_positionpower[i_ev][1, :] == index_station_virtual)[0]
        # delete associated column
        # if index_deletecol.size != 0:  # judge if index_deletecol is empty array, but there is no need, as the empty
        # array will not delete elements and not raise exception
        ev_positionpower[i_ev] = delete(ev_positionpower[i_ev], index_deletecol, axis=1)  # axis=1, deleting column, why???

        return ev_positionpower