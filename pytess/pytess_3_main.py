from numpy import ones, tile, newaxis, hstack, array, tan, arccos, setdiff1d, \
    sum, sqrt, in1d, flatnonzero
from scipy.sparse import csr_matrix

from pypower.loadcase import loadcase

# mpl.use('TkAgg')  # For invocation in terminal, refer to notes in Onenote
import matplotlib.pyplot as plt

import pandapower as pp
import pandapower.converter as pc
import pandapower.topology as ptop

from pytess.archive.gen_load_inter_cost import gen_load_inter_cost_net
from pytess.archive.gen_load_profile import gen_load_profile_ppc, gen_load_profile_net
from pytess.archive.gen_load_type import gen_load_type_net
from pytess.archive.get_load_info import get_load_information_net
from archive.add_dict_items import add_dict_items
from archive.set_model_tess_1 import set_model_cpx_1
from archive.set_model_tess_2 import set_model_cpx_3
from archive.sort_results import sort_results
from pytess.get_evposition import get_evposition
from pytess.get_evpositionpower import get_evpositionpower
from pytess.get_mgrev_e import get_mgrev_e
from pytess.test_case import siouxfalls, sscase
from pytess.archive.case33_modified import case33_modified
from pytess.archive.case_station_ev import case_station_ev

def cpx_test():
    '''
    pandapower-based
    :param casedata:
    :return:
    '''
    # ----------------------------------------------------------------------------------------------------------------------
    ppc = loadcase(case33_modified())  # ppc's data need to be in standard type as
    # defined, p.u. and nominal value
    tsc = siouxfalls()
    mgc = sscase()

    # ppc_joint = combine_ppc(ppc, ppc, ppc)
    ppnet = pc.from_ppc(ppc)
    # it's view, not duplicate, so any changes in bus will take effect in ppnet.bus
    bus, load, ext_grid, line, gen = ppnet.bus, ppnet.load, ppnet.ext_grid, ppnet.line, ppnet.gen

    # Convert MW to kW
    MW_KW = 1000
    #
    sn_mva = ppnet.sn_kva / MW_KW  # Base value for S capacity
    vn_kv = bus.loc[0, 'vn_kv']
    in_ka = sn_mva / (sqrt(3) * vn_kv)  # Base value for current

    # Save orginal bus information to field 'name' for sorting bus number
    # 'bus' is named
    load['name'] = load['bus']  # ['load_{0}'.format(load_i) for load_i in load['bus']]  # load name
    ext_grid['name'] = ext_grid['bus']
    line['name'] = line.index

    # All power values are given in the consumer system, therefore p_kw is positive if the external grid is absorbing
    # power and negative if it is supplying power.
    # but we choose the same convention as pypower. ext_grid and gen represent output parameters.
    # todo update data structure that can be self-adaptive to expanding network
    ext_grid['max_p_kw'] = array([1.6, 1.6, 1.8]) * MW_KW
    # ext_grid['max_p_kw'] = array([[1.6, 1.6, 1.8],
    #                               [1.6, 1.6, 1.8],
    #                               [1.6, 1.6, 1.8]
    #                              ]).ravel() * MW_KW  # Use .loc for indexing if not the whole column
    ext_grid['min_p_kw'] = 0  # Need to correct min_p_kw in pandapower
    # For max_q_kw and min_q_kw
    ext_grid['max_q_kvar'] = ppnet.ext_grid['max_p_kw'] * 0.8
    ext_grid['min_q_kvar'] = -ppnet.ext_grid['max_p_kw'] * 0.8

    # Add new column at the end to indicate social benefit of supplying corresponding load.
    # convert it to 'load'
    # Add 'load_cost' into ppnet.load
    ppnet = gen_load_inter_cost_net(ppnet)  # Even passing arguments is a view?
    # Add 'load_type' into ppent.load
    ppnet = gen_load_type_net(ppnet)

    # Configure bus voltage upper and lower bound
    bus['max_vm_pu'] = 1.05
    bus['min_vm_pu'] = 0.95

    # Consolidate load restoration benefit and load type
    load_information = get_load_information_net(ppnet)

    ############################################################
    index_outage_line = [17, 3, 22]
    # index_outage_line = [24, 34, 0]
    # index_outage_line = []

    line['in_service'] = True
    line.loc[index_outage_line, 'in_service'] = False

    # Identify areas and remove isolated ones -------------------------------------------------------------------------
    # Set all isolated buses and all elements connected to isolated buses out of service.
    pp.set_isolated_areas_out_of_service(ppnet)  # isolated means that a bus is disconnected from ext_grid

    # Drops any elements not in service AND any elements connected to inactive buses.
    pp.drop_inactive_elements(ppnet)  # keep buses that connect to ext_grid both directly and indirectly.
    # Creates a continuous bus index starting at zero and replaces all references of old indices by the new ones.
    ppnet = pp.create_continuous_bus_index(ppnet, start=0)
    # Correct line index starting at zero
    line.index = range(line.shape[0])
    # Correct load index starting at zero
    load.index = range(load.shape[0])
    # Correct ext_grid index starting at zero
    ext_grid.index = range(ext_grid.shape[0])

    ppgraph = ptop.create_nxgraph(ppnet)
    # Is it necessary to split into sub-nets?
    '''
    # Converts a pandapower network into a NetworkX graph
    ppgraph = ptop.create_nxgraph(ppnet)

    ppsubnet = []
    for e_island in ptop.connected_components(ppgraph):
        print(e_island)
        ppsubnet.append(pp.select_subnet(ppnet, buses=e_island))  # Modify pp.select_subnet to sort buses in ascending
    '''  # numerical order
    # --------------------------------------------------------------------------
    # Get bus index lists of each type of bus
    idx_ref_bus = array(ext_grid['bus'], dtype=int)
    idx_pv_bus = array(gen['bus'], dtype=int)
    idx_pq_bus = setdiff1d(array(bus.index, dtype=int), hstack((idx_ref_bus, idx_pv_bus)))
    idx_mg_bus = idx_ref_bus

    nb = bus.shape[0]
    nl = line.shape[0]
    n_mg = ext_grid.shape[0]

    nd = load.shape[0]

    # check if there are transformers?
    # if NOT trafo is None:

    branch_r = (line['r_ohm_per_km'] * line['length_km'] / line['parallel']
                     / ((vn_kv * 1e3) ** 2 / (sn_mva * 1e6))).values
    branch_x = (line['x_ohm_per_km'] * line['length_km'] / line['parallel']
                     / ((vn_kv * 1e3) ** 2 / (sn_mva * 1e6))).values

    # generate load profile considering load types
    p_load_profile, q_load_profile = gen_load_profile_net(ppnet)
    # Number of time intervals
    n_interval = p_load_profile.shape[1]
    # Time step
    delta_t = 1
    # Time window
    n_timewindow_set = 12
    n_timewindow_effect = 1

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Distribution network topology
    # Connection matrix for nb corresponding to nl.
    # connection matrix for from buses & line (nb, nl)
    connection_ds_f = csr_matrix((ones(nl), (array(line['from_bus']),
                                             range(nl))),
                                 shape=(nb, nl), dtype=int)
    # connection matrix for to buses & line (nb, nl)
    connection_ds_t = csr_matrix((ones(nl), (array(line['to_bus']),
                                             range(nl))),
                                 shape=(nb, nl), dtype=int)

    # Find all the immediately neighboring nodes connecting to each MG bus
    # # The index for branches starting from MG bus
    # idx_beta_ij = []  # Set up List then convert it to csr_matrix matrix (array not csr_matrix matrix)
    # # The index for branches ending at MG bus
    # idx_beta_ji = []
    #
    # for i_ref_bus in range(idx_ref_bus.shape[0]):
    #     idx_beta_ij += nonzero(line['from_bus'] == idx_ref_bus[i_ref_bus])[0].tolist()
    #     idx_beta_ji += nonzero(line['to_bus'] == idx_ref_bus[i_ref_bus])[0].tolist()
    #
    # idx_beta_ij = array(idx_beta_ij)
    # idx_beta_ji = array(idx_beta_ji)

    # # The problem is that intersection cannot deal with duplicate rows
    # idx_beta_ij_1 = indices(line['from_bus'], intersection(line['from_bus'], idx_ref_bus))
    # idx_beta_ji_1 = indices(line['to_bus'], intersection(line['to_bus'], idx_ref_bus))

    # in1d finds out all the same elements and set corresponding position in arg1 to 1
    # The index for branches starting from MG bus
    idx_beta_ij = line['from_bus'].index[in1d(line['from_bus'], idx_ref_bus)]
    # The index for branches ending at MG bus
    idx_beta_ji = line['to_bus'].index[in1d(line['to_bus'], idx_ref_bus)]

    # Load (nd,)
    load_inter_cost = load['load_cost'].values
    # (nd, n_interval)
    load_inter_cost_augment = tile(load_inter_cost[:, newaxis], (1, n_interval))

    # Find out load bus indices
    idx_load = load['bus'].values

    # Connection matrix for bus-mg and bus-load
    # (nb, n_mg)
    connection_mg = csr_matrix(
        (ones(n_mg), (ext_grid['bus'], range(n_mg))), shape=(nb, n_mg), dtype=int)
    # (nb, nd)
    connection_load = csr_matrix((ones(nd), (load['bus'], range(nd))),
                                 shape=(nb, nd), dtype=int)



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
    local_load_peak = array([
        [0.5, 2],
        [0.7, 3],
        [0.7, 1]
    ])
    p_localload_profile, q_localload_profile = gen_load_profile_ppc(local_load_peak, 1)  # it's _ppc other than _net.
    localload_p = local_load_peak[:, 0][:,
                  newaxis] * p_localload_profile / sn_mva  # local_load_peak is broadcast, (n_mg, n_interval)
    localload_q = local_load_peak[:, 0][:, newaxis] * tan(
        arccos(0.9)) * q_localload_profile / sn_mva  # (n_mg, n_interval)
    # Define the lower and upper bounds
    # Upper bounds
    load_mutliple = 2
    slmax = line['max_i_ka'] / in_ka
    pd_u = load['p_kw'][:, newaxis] / MW_KW * p_load_profile * load_mutliple / sn_mva
    qd_u = load['q_kvar'][:, newaxis] / MW_KW * q_load_profile * load_mutliple / sn_mva
    load_qp_ratio = load['q_kvar'] / load['p_kw']
    pij_u = tile(slmax[:, newaxis], (1, n_interval))
    qij_u = tile(slmax[:, newaxis], (1, n_interval))
    sij_u = tile(slmax[:, newaxis], (1, n_interval))
    pmg_u = tile(ext_grid['max_p_kw'][:, newaxis] / MW_KW / sn_mva, (1, n_interval))
    qmg_u = tile(ext_grid['max_q_kvar'][:, newaxis] / MW_KW / sn_mva, (1, n_interval))

    # Lower bounds
    pij_l = tile(-slmax[:, newaxis], (1, n_interval))
    qij_l = tile(-slmax[:, newaxis], (1, n_interval))
    pmg_l = tile(ext_grid['min_p_kw'][:, newaxis] / MW_KW / sn_mva, (1, n_interval))  # (n_mg, n_interval)
    qmg_l = tile(ext_grid['min_q_kvar'][:, newaxis] / MW_KW / sn_mva, (1, n_interval))  # (n_mg, n_interval)

    # mgs' energy_capacity_ratio = array([0.6, 0.6, 0.6])
    energy_capacity_ratio = 0.6 * ones(n_mg)
    energy_lower_ratio = 0.1
    # energyCapacity
    # To indicate the upper and lower bound for MG energy reserve
    energy_u = ext_grid['max_p_kw'] / MW_KW * energy_capacity_ratio * n_interval / sn_mva  # 1-D array, (n_mg)
    energy_u = tile(energy_u[:, newaxis], (1, n_interval))  # (n_mg, n_interval)
    energy_l = energy_u * energy_lower_ratio  # 2-D array, (n_mg, n_interval)

    # A very large positive number used to relax power flow constraints
    large_m = 1e6
    # Voltage 1 p.u.
    v0 = 1
    # MG electricity price
    cost_mg = 0.5  # 1$/kWh

    # Bus voltage magnitude
    # Lower bounds
    vm_l = tile(bus['min_vm_pu'][:, newaxis], (1, n_interval))
    vm_l[ext_grid['bus'].astype(int), :] = v0
    # Upper bounds
    vm_u = tile(bus['max_vm_pu'][:, newaxis], (1, n_interval))
    vm_u[ext_grid['bus'].astype(int), :] = v0

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # # Time-space network for TESS dispatching
    # tsn_network = TsnModel()
    # traffic_case = case3_tsn.transportation_network(delta_t)
    # tsn_model = tsn_network.set_tsn_model(traffic_case, n_interval, ppnet)
    #
    # # Extract tsn model
    # # check if a location is a charging station
    # # a boolean vector with shape of (n_location), 1 indicating a station
    # is_location_station = tsn_model['is_location_station']
    # # number of charging station
    # n_station = tsn_model['n_station']
    # # connection matrix, (nb, n_station), indicating each charging station's
    # # position at ds bus
    # connection_station = tsn_model['connection_station']
    # # number of tsn node
    # n_tsn_node = tsn_model['n_tsn_node']
    # # idx_tsn_node (n_location, n_interval)
    # idx_tsn_node = tsn_model['idx_tsn_node']
    # # number of tsn arc
    # n_tsn_arc = tsn_model['n_tsn_arc']
    # # To set up arc table with 4 columns, [F_BUS, T_BUS, TIME, LOCATION]
    # tsn_arc_table = tsn_model['tsn_arc_table']
    # # cut_set of arcs for each interval
    # cut_set_tsn = tsn_model['cut_set_tsn']
    # # Indicate the index of parking arc in tsn arc table
    # idx_tsn_arc_parking = tsn_model['idx_tsn_arc_parking']
    # # connection matrix: (n_tsn_node, n_tsn_arc), indicate each arc's from bus
    # # and to bus
    # connection_tsn_f = tsn_model['connection_tsn_f']
    # connection_tsn_t = tsn_model['connection_tsn_t']

    # TESS configuration
    # TESS parameters
    station_ev_data = loadcase(case_station_ev())
    n_ev = station_ev_data['gen'].shape[0]  # The number of ev
    ev_energy_capacity = 1. * ones(n_ev) / sn_mva  # 1-D array
    soc_init = 0.5 * ones(n_ev)
    soc_max = 0.9 * ones((n_ev, n_interval))
    soc_min = 0.1 * ones((n_ev, n_interval))
    ev_energy_u = soc_max * ev_energy_capacity[:, newaxis]  # ev energy upper bound, (n_ev, n_interval)
    ev_energy_init = soc_init * ev_energy_capacity  # (n_ev)
    ev_energy_l = soc_min * ev_energy_capacity[:, newaxis]
    ev_dch_efficiency = 0.95  # discharging efficiency
    ev_ch_efficiency = 0.95  # charging efficiency
    ev_dch_u = 0.2 / sn_mva  # maximum discharging power
    ev_ch_u = 0.2 / sn_mva  # maximum discharging power
    loss_on_road = 0 / sn_mva  #
    cost_ev_transit = 80  # ev transportation cost
    cost_ev_power = 0.2  # ev power cost $0.2/kWh

    # # Define ev's initial position array
    # # To find indices of ev's initial position
    # idx_ev_position_init = array([nonzero(x == traffic_case['bus'][:, 1])[0]
    #                             for x in station_ev_data['gen'][:, GEN_BUS]]
    #                              ).reshape(n_ev)  # convert (n_ev, 1) to (n_ev)
    #
    # # idx_ev_position_init = flatnonzero(in1d(ext_grid['name'], station_ev_data['gen'][:, GEN_BUS]))
    # ev_position_init = csr_matrix((ones(n_ev), (range(n_ev), idx_ev_position_init)),
    #                               shape=(n_ev, n_station),
    #                               dtype=int).T.toarray()  # (transpose to (n_station, n_ev), making it easier to invoke in TSN model
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



