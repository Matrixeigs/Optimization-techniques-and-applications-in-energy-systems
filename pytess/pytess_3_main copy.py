from numpy import nonzero, ones, array, sum
from scipy.sparse import csr_matrix

from pypower.idx_gen import GEN_BUS

# mpl.use('TkAgg')  # For invocation in terminal, refer to notes in Onenote
import matplotlib.pyplot as plt

from archive.add_dict_items import add_dict_items
from archive.set_model_tess_1 import set_model_cpx_1
from archive.set_model_tess_2 import set_model_cpx_3
from archive.sort_results import sort_results
from pytess.get_evposition import get_evposition
from pytess.get_evpositionpower import get_evpositionpower
from pytess.get_mgrev_e import get_mgrev_e


def cpx_test():
    '''
    pandapower-based
    :param casedata:
    :return:
    '''
    # ----------------------------------------------------------------------------------------------------------------------



    # numerical order
    # --------------------------------------------------------------------------





    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^






    # local_load_peak = array([
    #     [0.5, 2],
    #     [0.7, 3],
    #     [0.7, 1],
    #     [0.5, 2],
    #     [0.7, 3],
    #     [0.7, 1],
    #     [0.5, 2],
    #     [0.7, 3],
    #     [0.7, 1]
    # ])
    # [peak load, load type]

    # Define the lower and upper bounds
    # Upper bounds
    load_mutliple = 2

    # pd_u = load['p_kw'][:, newaxis] / MW_KW * p_load_profile * load_mutliple / sn_mva
    # qd_u = load['q_kvar'][:, newaxis] / MW_KW * q_load_profile * load_mutliple / sn_mva
    # load_qp_ratio = load['q_kvar'] / load['p_kw']


    # A very large positive number used to relax power flow constraints
    large_m = 1e6



    # Bus voltage magnitude
    # Lower bounds



    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





    # Define ev's initial position array
    # To find indices of ev's initial position
    idx_ev_position_init = array([nonzero(x == traffic_case['bus'][:, 1])[0]
                                for x in station_ev_data['gen'][:, GEN_BUS]]
                                 ).reshape(n_ev)  # convert (n_ev, 1) to (n_ev)

    # idx_ev_position_init = flatnonzero(in1d(ext_grid['name'], station_ev_data['gen'][:, GEN_BUS]))
    ev_position_init = csr_matrix((ones(n_ev), (range(n_ev), idx_ev_position_init)),
                                  shape=(n_ev, n_station),
                                  dtype=int).T.toarray()  # (transpose to (n_station, n_ev), making it easier to invoke in TSN model
    n_scenario = 1
    scenario_weight = 1 / n_scenario

    # Assignment of caseParams
    case_params = {}  # dictionary

    # Parameters for settings
    case_params = add_dict_items(case_params, n_interval=n_interval,
        delta_t=delta_t, n_timewindow_set=n_timewindow_set,
        n_timewindow_effect=n_timewindow_effect, large_m=large_m,
        n_scenario=n_scenario, scenario_weight=scenario_weight, sn_mva=sn_mva,
        MW_KW=MW_KW)

    # Parameters for ds
    case_params = add_dict_items(case_params, ppnet=ppnet,
        idx_ref_bus=idx_ref_bus, idx_pv_bus=idx_pv_bus, idx_pq_bus=idx_pq_bus,
        idx_mg_bus=idx_mg_bus, nb=nb, nl=nl, nd=nd, branch_r=branch_r,
        branch_x=branch_x, connection_ds_f=connection_ds_f,
        connection_ds_t=connection_ds_t, idx_beta_ij=idx_beta_ij,
        idx_beta_ji=idx_beta_ji, p_load_profile=p_load_profile,
        q_load_profile=q_load_profile, load_inter_cost=load_inter_cost,
        load_inter_cost_augment=load_inter_cost_augment, idx_load=idx_load,
        connection_load=connection_load, pij_u=pij_u, qij_u=qij_u, sij_u=sij_u,
        pij_l=pij_l, qij_l=qij_l, v0=v0, vm_u=vm_u, vm_l=vm_l,
        slmax=slmax, pd_u=pd_u, qd_u=qd_u, load_qp_ratio=load_qp_ratio)

    # Parameters for mg
    case_params = add_dict_items(case_params, n_mg=n_mg,
        connection_mg=connection_mg, localload_p=localload_p,
        localload_q=localload_q, pmg_u=pmg_u, qmg_u=qmg_u, pmg_l=pmg_l,
        qmg_l=qmg_l, energy_capacity_ratio=energy_capacity_ratio,
        energy_u=energy_u, energy_l=energy_l, cost_mg=cost_mg)

    # Parameters for ts
    case_params = add_dict_items(case_params, n_station=n_station,
        connection_station=connection_station, n_tsn_node=n_tsn_node,
        idx_tsn_node=idx_tsn_node, n_tsn_arc=n_tsn_arc,
        tsn_arc_table=tsn_arc_table, cut_set_tsn=cut_set_tsn,
        idx_tsn_arc_parking=idx_tsn_arc_parking,
        connection_tsn_f=connection_tsn_f, connection_tsn_t=connection_tsn_t,
        is_location_station=is_location_station)

    # Parameters for ev
    case_params = add_dict_items(case_params, n_ev=n_ev,
        ev_energy_capacity=ev_energy_capacity, soc_init=soc_init,
        soc_max=soc_max, soc_min=soc_min, ev_energy_u=ev_energy_u,
        ev_energy_l=ev_energy_l, ev_energy_init=ev_energy_init,
        ev_dch_efficiency=ev_dch_efficiency, ev_ch_efficiency=ev_ch_efficiency,
        ev_dch_u=ev_dch_u, ev_ch_u=ev_ch_u, loss_on_road=loss_on_road,
        cost_ev_transit=cost_ev_transit, cost_ev_power=cost_ev_power,
        ev_position_init=ev_position_init)

    # Non-rolling optimization process
    # Test setmodel
    # '''
    # model_x = set_model_grb_1(case_params)
    model_x = set_model_cpx_3(case_params)

    model_x.solve()

    res = sort_results(case_params, model_x)

    ev_position = get_evposition(ev_position_init, res)

    ev_positionpower = get_evpositionpower(ev_position, res)

    mgrev_e = get_mgrev_e(res, case_params)
    objective_val = model_x.solution.get_objective_value()

    # The default, axis=None, will sum(np.sum) all of the elements of the input array.
    cost_customer_interruption = sum(load_inter_cost_augment * res['pdcut_x'] * delta_t, axis=None) * MW_KW * sn_mva
    cost_mg_generation = sum(cost_mg * res['pmg_x'] * delta_t, axis=None) * MW_KW * sn_mva
    cost_transportation = sum(cost_ev_transit * res['sign_onroad_x'], axis=None)
    cost_batterymaintenance = sum(cost_ev_power * (res['pev_dch_x'] + res['pev_ch_x']) * delta_t, axis=None) * MW_KW * sn_mva
    cost_total = cost_customer_interruption + cost_mg_generation + cost_transportation + cost_batterymaintenance

    # '''
    # Rolling optimization
    # Define the model and case results list
    ls_model_x = []
    ls_case_res = []

    for j_interval in range(n_interval):
        # Update model information for the time window, the result needs to be encapsulated in a function
        # Intervals in the time window
        n_timewindow_interval = min(n_timewindow_set, n_interval - j_interval)  # exact intervals in time window in each iteration
        interval_end = j_interval + n_timewindow_interval  # the end+1 interval of the time window

        # Updates on time interval
        case_params['n_interval'] = n_timewindow_interval
        # Updates in objective function
        case_params['load_inter_cost_augment'] = load_inter_cost_augment[:, j_interval:interval_end]

        # Upper and lower bounds (only parameters associated with time)
        case_params['pmg_l'] = pmg_l[:, j_interval:interval_end]
        case_params['qmg_l'] = qmg_l[:, j_interval:interval_end]
        case_params['localload_p'] = localload_p[:, j_interval:interval_end]
        case_params['localload_q'] = localload_q[:, j_interval:interval_end]

        case_params['pmg_u'] = pmg_u[:, j_interval:interval_end]
        case_params['qmg_u'] = qmg_u[:, j_interval:interval_end]

        case_params['soc_min'] = soc_min[:, j_interval:interval_end]
        case_params['soc_max'] = soc_max[:, j_interval:interval_end]
        case_params['energy_l'] = energy_l[:, j_interval:interval_end]
        case_params['energy_u'] = energy_u[:, j_interval:interval_end]
        case_params['pij_l'] = pij_l[:, j_interval:interval_end]
        case_params['pij_u'] = pij_u[:, j_interval:interval_end]
        case_params['qij_l'] = qij_l[:, j_interval:interval_end]
        case_params['qij_u'] = qij_u[:, j_interval:interval_end]
        case_params['vm_l'] = vm_l[:, j_interval:interval_end]
        case_params['vm_u'] = vm_u[:, j_interval:interval_end]

        case_params['pd_u'] = pd_u[:, j_interval:interval_end]
        case_params['qd_u'] = qd_u[:, j_interval:interval_end]

        # Update EV's initial position at the beginning of the time window
        if j_interval:
            ev_position = get_evposition(ev_position_init, ls_case_res[j_interval-1])
            case_params['ev_position_init'] = csr_matrix((ones(n_ev), (range(n_ev), ev_position[:, n_timewindow_effect])),
                                                         shape=[n_ev, n_station], dtype=int).T.toarray()
            # Why does it need to transpose?
            # aaa = csr_matrix((ones(n_ev), (ev_position[:, n_timewindow_effect], range(n_ev))),
            #                                          shape=[n_ev, n_station], dtype=int).toarray()

            # ev_position_init = csr_matrix((ones(n_ev), (range(n_ev), index_initposition)),
            #                           shape=(n_ev, n_station), dtype=int).T.toarray()
        else:
            case_params['ev_position_init'] = ev_position_init

        # Update initial SOC at the beginning of the time window
        if j_interval:
            case_params['soc_init'] = ls_case_res[j_interval-1]['ev_soc_x'][:, :n_timewindow_effect]
        else:
            case_params['soc_init'] = soc_init

        ls_model_x.append(set_model_cpx_1(case_params))
        ls_model_x[j_interval].solve()
        ls_case_res.append(sort_results(case_params, ls_model_x[j_interval]))


    # unit conversion

    plt.plot(ev_position)
    # only on plt.show() in one session, it's generally put at the end
    plt.show()

if __name__ == "__main__":
    cpx_test()



