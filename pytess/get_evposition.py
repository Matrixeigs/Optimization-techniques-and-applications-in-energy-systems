from numpy import zeros

from pypower.loadcase import loadcase
from pypower.idx_brch import T_BUS, F_BUS

from archive.case_station_ev import case_station_ev

def get_evposition(ev_position_init, res):
    """
    
    :param ev_position_init: 
    :param res: 
    :return: 
    """

    station_ev_data = loadcase(case_station_ev())

    # get parameters
    n_ev = station_ev_data['gen'].shape[0]
    n_interval = res['ev_arc_x'].shape[1]
    n_timepoint = n_interval + 1

    ev_position = zeros((n_ev, n_timepoint), dtype=int)  # for each time point strating from time point 0

    # To find ev's position at beginning of each time span (both beginning and end for the last time span)
    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            j_timepoint = j_interval  # use the beginning point to get position

            index_this_activearc = res['ev_arc_x'][i_ev, :, j_interval].nonzero()
            ev_position[i_ev, j_timepoint] = station_ev_data['branch'][index_this_activearc, F_BUS].astype(int)
            ev_position[i_ev, j_timepoint+1] = station_ev_data['branch'][index_this_activearc, T_BUS].astype(int)

    return ev_position



