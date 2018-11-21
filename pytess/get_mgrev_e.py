
from numpy import zeros


def get_mgrev_e(res, case_params):
    """
    To calculate the energy that each MG receives from all EVs.
    :param res:
    :param case_params:
    :return: mgrev_e: 2-D array, (n_mg, n_interval + 1), mg receiving energy from ev
    """

    mg_p_ev_x = res['mg_p_ev_x']  # (n_mg, n_interval), indicating each mg's net receiving power associated with ev at time span t.
    delta_t = case_params['delta_t']

    n_mg = mg_p_ev_x.shape[0]
    n_interval = mg_p_ev_x.shape[1]
    n_timepoint = n_interval + 1

    mgrev_e = zeros((n_mg, n_timepoint))

    for j_interval in range(n_interval):
        mgrev_e[:, j_interval+1] = mgrev_e[:, j_interval] + mg_p_ev_x[:, j_interval] * delta_t

    return mgrev_e
