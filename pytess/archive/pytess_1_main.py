
from numpy import nonzero, unique, ones, transpose, exp, tile, newaxis, hstack, array, tan, arccos, setdiff1d, \
    pi, sum
from scipy.sparse import csr_matrix as sparse

from pypower.bustypes import bustypes
from pypower.loadcase import loadcase
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, TAP, SHIFT, BR_STATUS, RATE_A
from pypower.idx_bus import PQ, REF, BUS_I, BUS_TYPE, PD, QD, VMAX, VMIN
from pypower.idx_gen import GEN_BUS, PMAX, PMIN, QMAX, QMIN
from pypower.ext2int import ext2int

from pytess.case33_modified import case33_modified
from archive.case_station_ev import case_station_ev
from pytess.gen_load_inter_cost import gen_load_inter_cost_ppc
from pytess.gen_load_profile import gen_load_profile_ppc
from pytess.gen_load_type import gen_load_type_ppc
from pytess.get_load_info import get_load_information_ppc
from archive.add_dict_items import add_dict_items
from archive.set_model_tess_1 import set_model_cpx_1
from archive.sort_results import sort_results
from pytess.get_evposition import get_evposition
from pytess.get_evpositionpower import get_evpositionpower
from pytess.get_mgrev_e import get_mgrev_e

from numpy_indexed import intersection, indices

import matplotlib.pyplot as plt


def cpx_test(casedata=None):
    # Convert MV to kW
    MW_KW = 1000

    ppc = loadcase(casedata)
    # Active power
    ppc['gen'][0, PMAX] = 1.6
    ppc['gen'][1, PMAX] = 1.6
    ppc['gen'][2, PMAX] = 1.8
    # Reactive power with power factor 0.9
    ppc['gen'][:, QMAX] = ppc['gen'][:, PMAX] * 0.8
    ppc['gen'][:, QMIN] = -ppc['gen'][:, PMAX] * 0.8

     # Add new column at the end to indicate social benefit of supplying corresponding load.
    ppc['bus'] = hstack((ppc['bus'], gen_load_inter_cost_ppc()[:, newaxis]))
    LOAD_COST = ppc['bus'].shape[1] - 1  # define the column index of LOAD_COST
    # Add another new column at the very end to indicate load type
    ppc['bus'] = hstack((ppc['bus'], gen_load_type_ppc()[:, newaxis]))
    # mpc.bus(:, end + 1) = gen_load_type();
    LOAD_TYPE = ppc['bus'].shape[1] - 1  # Define the index LOAD_TYPE for the last column
    # Set bus voltage upper and lower bound
    ppc['bus'][:, VMIN] = 0.95
    ppc['bus'][:, VMAX] = 1.05
    # Consolidate load restoration benefit and load type
    load_information = get_load_information_ppc(ppc, BUS_I, LOAD_COST, LOAD_TYPE)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    index_outage_line = [25-1, 32-1, 1-1]
    index_outage_line = []  # ppc edition cannot deal with topology issue

    ppc['branch'][:, BR_STATUS] = 1
    for i in range(len(index_outage_line)):
        ppc['branch'][index_outage_line[i], BR_STATUS] = 0

    ppc['bus'][ppc['bus'][:, BUS_TYPE] == REF, BUS_TYPE] = PQ  # Change the original slack bus type

    #1) Detect how many generators we have
    # Note the difference between index *** and id***
    id_genbus = unique(ppc['gen'][:, GEN_BUS]).astype(int)

    for i in range(len(id_genbus)):
        ppc['bus'][id_genbus[i], BUS_TYPE] = REF
    # id_genbus = nonzero(ppc['bus'][:, BUS_TYPE] == REF)[0]

    # Remove the offline buses and lines
    # Make the adjacent matrix of modified matrix
    # bus_deleted = []
    # bus_map, bus_isolated = find_islands(ppc)
    # if inters

    ppc = ext2int(ppc)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    baseMVA, bus, gen, branch, gencost = ppc['baseMVA'], ppc['bus'], ppc['gen'], ppc['branch'], ppc['gencost']

    #Get bus index lists of each type of bus
    index_ref_bus, index_pv_bus, index_pq_bus = bustypes(bus, gen)
    index_genbus = gen[:, GEN_BUS].astype(int)

    nb = bus.shape[0]
    nl = branch.shape[0]
    ng = gen.shape[0]
    # n_mg = ng.copy() for integers, no need for copy() method
    n_mg = ng
    stat = branch[:, BR_STATUS]
    tap = ones((nl))
    i = transpose(nonzero(branch[:, BR_STATUS]))
    i = nonzero(branch[:, BR_STATUS])[0]
    tap[i] = branch[i, TAP]
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])
    branch_r = branch[:, BR_R] * tap
    branch_x = branch[:, BR_X] * tap

    # generate load profile considering load types
    p_load_profile, q_load_profile = gen_load_profile_ppc(bus, LOAD_TYPE)
    # Number of time spans
    n_interval = p_load_profile.shape[1]
    # Time step
    delta_t = 1
    # Time window
    n_timewindow_set = 12
    n_timewindow_effect = 1

    # Distribution network topology
    f = branch[:, F_BUS]
    t = branch[:, T_BUS]
    # Connection matrix for nb corresponding to nl.
    connection_f = sparse((ones(nl), (range(nl), f)), shape=(nl, nb), dtype=int).T  # connection matrix for from buses & line (nb, nl)
    connection_t = sparse((ones(nl), (range(nl), t)), shape=(nl, nb), dtype=int).T  # conncetion matrix for to buses & line (nb, nl)

    # del f, t, i
    load_inter_cost = bus[:, LOAD_COST]
    load_inter_cost_augment = tile(load_inter_cost[:, newaxis], (1, n_interval))

    # To find load_bus indices
    LOAD_COST_THRESHOLD = 0
    index_load = nonzero(((bus[:, PD] != 0) | (bus[:, QD] != 0)) & (bus[:, LOAD_COST] >= LOAD_COST_THRESHOLD))[0]
    nd = index_load.shape[0]
    # Connection matrix for bus-generator and bus-load
    connection_generator = sparse( (ones(ng), (gen[:, GEN_BUS], range(ng) )), shape=(nb, ng), dtype=int)  # (nb, ng)
    connection_load = sparse( (ones(nd), (index_load, range(nd) )), shape=(nb, nd), dtype=int)  # (nb, nd)

    # Find all the immediately neighboring nodes connecting to each MG bus
    # The index for branches starting from MG bus
    index_beta_ij = []  # Set up List then convert it to csr_matrix matrix (array not csr_matrix matrix)
    # The index for branches ending at MG bus
    index_beta_ji = []

    for iRefBus in range(index_ref_bus.shape[0]):
        index_beta_ij += nonzero(branch[:, F_BUS] == index_ref_bus[iRefBus])[0].tolist()
        index_beta_ji += nonzero(branch[:, T_BUS] == index_ref_bus[iRefBus])[0].tolist()

    index_beta_ij = array(index_beta_ij)
    index_beta_ji = array(index_beta_ji)

    local_load_peak = array([
                        [0.5, 2],
                        [0.7, 3],
                        [0.7, 1]
                        ])
    p_localload_profile, q_localload_profile = gen_load_profile_ppc(local_load_peak, 1)
    localload_p = local_load_peak[:, 0][:, newaxis] * p_localload_profile / baseMVA  # local_load_peak is broadcast, (n_mg, n_interval)
    localload_q = local_load_peak[:, 0][:, newaxis] * tan(arccos(0.9)) * q_localload_profile / baseMVA  #(n_mg, n_interval)
    # Define the lower and upper bounds
    # Upper bounds
    loadMutliple = 2
    slmax = branch[:, RATE_A] / 1 / baseMVA
    pd_u = bus[index_load, PD][:, newaxis] * p_load_profile[index_load, :] * loadMutliple / baseMVA
    qd_u = bus[index_load, QD][:, newaxis] * q_load_profile[index_load, :] * loadMutliple / baseMVA
    load_qp_ratio = bus[index_load, QD] / bus[index_load, PD]
    pij_u = tile(slmax[:, newaxis], (1, n_interval))
    qij_u = tile(slmax[:, newaxis], (1, n_interval))
    sij_u = tile(slmax[:, newaxis], (1, n_interval))
    pg_u = tile(gen[:, PMAX][:, newaxis]/baseMVA, (1, n_interval))
    qg_u = tile(gen[:, QMAX][:, newaxis]/baseMVA, (1, n_interval))

    # Lower bounds
    pij_l = tile(-slmax[:, newaxis], (1, n_interval))
    qij_l = tile(-slmax[:, newaxis], (1, n_interval))
    pg_l = tile(gen[:, PMIN][:, newaxis]/baseMVA, (1, n_interval))  # (ng, n_interval)
    qg_l = tile(gen[:, QMIN][:, newaxis]/baseMVA, (1, n_interval))  # (ng, n_interval)

    energy_capacity_ratio = array([0.6, 0.6, 0.6])
    energy_lower_ratio = 0.1
    # energyCapacity
    # To indicate the upper and lower bound for MG energy reserve
    energy_u = gen[:, PMAX] * energy_capacity_ratio * n_interval / baseMVA  # 1-D array, (n_mg)
    energy_u = tile(energy_u[:, newaxis], (1, n_interval)) # (n_mg, n_interval)
    energy_l = energy_u * energy_lower_ratio  # 2-D array, (n_mg, n_interval)

    # A very large positive number used to relax power flow constraints
    large_m = 1e6
    # Voltage 1 p.u.
    v0 = 1
    # MG electricity price
    cost_mg = 0.5  # 1$/kWh

    # Bus voltage magnitude
    # Lower bounds
    vm_l = tile(bus[:, VMIN][:, newaxis], (1, n_interval))
    vm_l[gen[:, GEN_BUS].astype(int), :] = v0
    # Upper bounds
    vm_u = tile(bus[:, VMAX][:, newaxis], (1, n_interval))
    vm_u[gen[:, GEN_BUS].astype(int), :] = v0

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## Time-space network for TESS dispatching
    # A dictionary indicating EV and station information
    station_ev_data = loadcase(case_station_ev())
    # index list of station, including virtual station
    index_station = station_ev_data['bus'][:, BUS_I]
    # Number of station, including virtual station
    n_station = index_station.shape[0]
    # Number of connection arcs
    n_arc = station_ev_data['branch'].shape[0]
    # Index list of connection arcs
    index_arc = range(n_arc)

    # To find parking arc and station
    # Parking station
    index_station_parking = nonzero(station_ev_data['bus'][:, BUS_TYPE] == 1)[0]
    n_station_parking = index_station_parking.shape[0]
    # parking arc
    index_arc_parking = nonzero(station_ev_data['branch'][:, F_BUS] == station_ev_data['branch'][:, T_BUS])[0]
    n_arc_parking = index_arc_parking.shape[0]

    # To find virtual station
    index_station_virtual = nonzero(station_ev_data['bus'][:, BUS_TYPE] == 0)[0]
    n_station_virtual = index_station_virtual.shape[0]

    # To find out transi arc
    index_arc_transit = setdiff1d(index_arc, index_arc_parking)
    n_arc_transit = index_arc_transit.shape[0]
    # To find transit arcs starting from or ending at virtual station
    index_arc_virtual = nonzero( station_ev_data['branch'][:, [F_BUS, T_BUS]] == index_station_virtual)[0]  # This only works when index_station_virtual is scalar

    # To find corresponding round trip arc
    temp_a = (station_ev_data['branch'][:, [F_BUS, T_BUS]])
    temp_b = (station_ev_data['branch'][:, [T_BUS, F_BUS]])
    # ia and ib indicate the indices of corresponding round trip arc
    temp_c = intersection(temp_a[index_arc_transit, :], temp_b[index_arc_transit, :])  # why temp_c is not sorted?
    roundtrip_t = indices(temp_a, temp_c)
    roundtrip_f = indices(temp_b, temp_c)

    del temp_a, temp_b, temp_c

    # Set up connection span matrix for station corresponding to each arc
    # station_connection_f = csr_matrix((n_station, n_arc))  # Set up csr_matrix matrix with zeros
    # station_connection_t = csr_matrix(n_station, n_arc)
    # No need to set up csr_matrix matrix and then use for loop
    # Since each arc only has one from bus and to bus, so we establish the csr_matrix matrix with arc as rows and then transpose it
    station_connection_f = sparse((ones(n_arc), (range(n_arc), station_ev_data['branch'][:, F_BUS])),
                                  shape=(n_arc, n_station), dtype=int).T
    station_connection_t = sparse((ones(n_arc), (range(n_arc), station_ev_data['branch'][:, T_BUS])),
                                  shape=(n_arc, n_station), dtype=int).T

    # TESS configuration
    # TESS parameters
    n_ev = station_ev_data['gen'].shape[0]  # The number of ev
    ev_energy_capacity = 1. * array([1, 1, 1, 1]) / baseMVA  # 1-D array
    soc_init = 0.5 * array([1, 1, 1, 1])
    soc_max = 0.9 * ones((n_ev, n_interval))
    soc_min = 0.1 * ones((n_ev, n_interval))
    ev_energy_u = soc_max * ev_energy_capacity[:, newaxis]  # ev energy upper bound, (n_ev, n_interval)
    ev_energy_init = soc_init * ev_energy_capacity  # (n_ev)
    ev_energy_l = soc_min * ev_energy_capacity[:, newaxis]
    ev_dch_efficiency = 0.95  # discharging efficiency
    ev_ch_efficiency = 0.95  # charging efficiency
    ev_dch_u = 0.2 / baseMVA  # maximum discharging power
    ev_ch_u = 0.2 / baseMVA  # maximum discharging power
    loss_on_road = 0 / baseMVA  #
    cost_ev_transit = 80  # ev transportation cost
    cost_ev_power = 0.2  # ev power cost $0.2/kWh

    # Define ev's initial position array
    # To find indices of ev's initial position
    index_initposition = array([nonzero(x == id_genbus)[0] for x in station_ev_data['gen'][:, GEN_BUS]]).reshape(n_ev)  # convert (n_ev, 1) to (n_ev)
    ev_position_init = sparse((ones(n_ev), (range(n_ev), index_initposition)),
                              shape=(n_ev, n_station), dtype=int).T.toarray()  # (transpose to (n_station, n_ev), making it easier to invoke in TSN model

####################################################################################
    n_scenario = 1
    scenario_weight = 1 / n_scenario

    # Assignment of caseParams
    case_params = {}  # dictionary

    case_params = add_dict_items(case_params, ppc=ppc, nb=nb, nl=nl, ng=ng, n_mg=n_mg, nd=nd, n_arc=n_arc, n_ev=n_ev,
                                 n_interval=n_interval, n_scenario=n_scenario, n_station=n_station, cost_ev_transit=cost_ev_transit,
                                 cost_ev_power=cost_ev_power, delta_t=delta_t, baseMVA=baseMVA, MW_KW=MW_KW, scenario_weight=scenario_weight,
                                 cost_mg=cost_mg, load_inter_cost=load_inter_cost, index_load=index_load, soc_min=soc_min, soc_max=soc_max)

    case_params = add_dict_items(case_params, pg_l=pg_l, qg_l=qg_l, localload_p=localload_p, localload_q=localload_q, energy_u=energy_u,
                                 energy_l=energy_l, pg_u=pg_u, qg_u=qg_u, vm_l=vm_l, vm_u=vm_u, slmax=slmax, pd_u=pd_u,
                                 qd_u=qd_u, station_connection_t=station_connection_t, station_connection_f=station_connection_f,
                                 ev_position_init=ev_position_init, n_arc_transit=n_arc_transit, roundtrip_f=roundtrip_f,
                                 roundtrip_t=roundtrip_t)

    case_params = add_dict_items(case_params, index_arc_parking=index_arc_parking, ev_ch_u=ev_ch_u, ev_dch_u=ev_dch_u,
                                 ev_energy_capacity=ev_energy_capacity, ev_energy_init=ev_energy_init, ev_energy_l=ev_energy_l,
                                 ev_energy_u=ev_energy_u, index_arc_transit=index_arc_transit, ev_ch_efficiency=ev_ch_efficiency,
                                 ev_dch_efficiency=ev_dch_efficiency, connection_f=connection_f, connection_t=connection_t,
                                 index_pq_bus=index_pq_bus, index_beta_ij=index_beta_ij, index_beta_ji=index_beta_ji,
                                 connection_generator=connection_generator, connection_load=connection_load, large_m=large_m,
                                 branch_r=branch_r, branch_x=branch_x, v0=v0, load_qp_ratio=load_qp_ratio)
    case_params = add_dict_items(case_params, soc_init=soc_init, load_inter_cost_augment=load_inter_cost_augment,
                                 n_timewindow_c=n_timewindow_effect, pij_l=pij_l, qij_l=qij_l, pij_u=pij_u, qij_u=qij_u,
                                 index_genbus=index_genbus)

    ## Non-rolling optimization process
    # Test setmodel
    # '''
    # model_x = set_model_grb_1(case_params)
    model_x = set_model_cpx_1(case_params)

    model_x.solve()

    res = sort_results(case_params, model_x)

    ev_position = get_evposition(ev_position_init, res)

    ev_positionpower = get_evpositionpower(ev_position, res)

    mgrev_e = get_mgrev_e(res, case_params)
    objective_val = model_x.solution.get_objective_value()

    # The default, axis=None, will sum(np.sum) all of the elements of the input array.
    cost_customer_interruption = sum(load_inter_cost_augment[index_load, :] * res['pdcut_x'] * delta_t, axis=None) * MW_KW * baseMVA
    cost_mg_generation = sum(cost_mg * res['pmg_x'] * delta_t, axis=None) * MW_KW * baseMVA
    cost_transportation = sum(cost_ev_transit * res['sign_onroad_x'], axis=None)
    cost_batterymaintenance = sum(cost_ev_power * (res['pev_dch_x'] + res['pev_ch_x']) * delta_t, axis=None) * MW_KW * baseMVA
    cost_total = cost_customer_interruption + cost_mg_generation + cost_transportation + cost_batterymaintenance

    # '''
    ## Rolling optimization
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
        case_params['pg_l'] = pg_l[:, j_interval:interval_end]
        case_params['qg_l'] = qg_l[:, j_interval:interval_end]
        case_params['localload_p'] = localload_p[:, j_interval:interval_end]
        case_params['localload_q'] = localload_q[:, j_interval:interval_end]

        case_params['pg_u'] = pg_u[:, j_interval:interval_end]
        case_params['qg_u'] = qg_u[:, j_interval:interval_end]

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
            case_params['ev_position_init'] = sparse((ones(n_ev), (range(n_ev), ev_position[:, n_timewindow_effect])),
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
    ppc = case33_modified()
    cpx_test(ppc)



