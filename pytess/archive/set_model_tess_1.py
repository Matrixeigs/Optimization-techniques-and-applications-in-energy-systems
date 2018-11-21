
from numpy import array, inf, full, zeros, ones, concatenate, arange, ix_, eye, diag, sqrt, nonzero, sum, isnan
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, hstack, vstack
from archive.add_dict_items import add_dict_items
import time

def set_model_cpx_1(case_params):
    import cplex as cpx
    # Extract parameters
    # dynamic allocation with locals() doesn't work in python3
    # names = locals()
    # for i in range(3):
    #     names['a'+str(i)]=i
    #     print('a'+str(i))
    #     exec("")

    # for element in case_params.items():
    #     # Execute nb = nb
    #     # exec(element[0] + '=' + 'element[1]')
    #     # Why locals() doesn't work while globals works?
    #      globals()[element[0]] = element[1]  # it's global variables, any alternatives?

    params_keys = case_params.keys()
    params_values = case_params.values()

    (ppc, nb, nl, ng, n_mg, nd, n_arc, n_ev, n_interval, n_scenario, n_station, cost_ev_transit, cost_ev_power, delta_t,
     baseMVA, MW_KW, scenario_weight, cost_mg, load_inter_cost, index_load, soc_min, soc_max, pg_l, qg_l, localload_p,
     localload_q, energy_u, energy_l, pg_u, qg_u, vm_l, vm_u, slmax, pd_u, qd_u, station_connection_t, station_connection_f,
     ev_position_init, n_arc_transit, roundtrip_f, roundtrip_t, index_arc_parking, ev_ch_u, ev_dch_u, ev_energy_capacity,
     ev_energy_init, ev_energy_l, ev_energy_u, index_arc_transit, ev_ch_efficiency, ev_dch_efficiency, connection_f,
     connection_t, index_pq_bus, index_beta_ij, index_beta_ji, connection_generator, connection_load, large_m, branch_r,
     branch_x, v0, load_qp_ratio, soc_init, load_inter_cost_augment, n_timewindow_c) = (case_params['ppc'], case_params['nb'],
            case_params['nl'], case_params['ng'], case_params['n_mg'], case_params['nd'], case_params['n_arc'],
            case_params['n_ev'], case_params['n_interval'], case_params['n_scenario'], case_params['n_station'],
            case_params['cost_ev_transit'],  case_params['cost_ev_power'], case_params['delta_t'], case_params['baseMVA'],
            case_params['MW_KW'], case_params['scenario_weight'], case_params['cost_mg'], case_params['load_inter_cost'],
            case_params['index_load'], case_params['soc_min'], case_params['soc_max'], case_params['pg_l'], case_params['qg_l'],
            case_params['localload_p'], case_params['localload_q'], case_params['energy_u'], case_params['energy_l'],
            case_params['pg_u'], case_params['qg_u'], case_params['vm_l'], case_params['vm_u'], case_params['slmax'], case_params['pd_u'],
            case_params['qd_u'], case_params['station_connection_t'], case_params['station_connection_f'], case_params['ev_position_init'],
            case_params['n_arc_transit'], case_params['roundtrip_f'], case_params['roundtrip_t'], case_params['index_arc_parking'],
            case_params['ev_ch_u'], case_params['ev_dch_u'], case_params['ev_energy_capacity'], case_params['ev_energy_init'],
            case_params['ev_energy_l'], case_params['ev_energy_u'], case_params['index_arc_transit'], case_params['ev_ch_efficiency'],
            case_params['ev_dch_efficiency'], case_params['connection_f'], case_params['connection_t'], case_params['index_pq_bus'],
            case_params['index_beta_ij'], case_params['index_beta_ji'], case_params['connection_generator'], case_params['connection_load'],
            case_params['large_m'], case_params['branch_r'], case_params['branch_x'], case_params['v0'], case_params['load_qp_ratio'],
            case_params['soc_init'], case_params['load_inter_cost_augment'], case_params['n_timewindow_c'])

    (pij_l, qij_l, pij_u, qij_u, index_genbus) = (case_params['pij_l'], case_params['qij_l'], case_params['pij_u'],
                                                  case_params['qij_u'], case_params['index_genbus'])

    # ppnet = case_params['ppnet']



    # Build model
    # Initialization of cplex
    model_x = cpx.Cplex()

    ## Define varialbles and get position array
    # prefix var_ is for variables index array
    # TESS model
    # charging power from mg to ev at time span t,
    #   3D-array, (n_ev, n_mg, n_interval)
    # var_pev2mg_ch_x = model_x.variables.add(names=['ch_ev{0}_
    # mg{1}_t{2}'.format(i_ev, j_mg, k_t)
    #                                                  for i_ev in range(n_ev)
    #                                                  for j_mg in range(n_mg)
    #                                                  for k_t in range(n_interval)])
    # var_pev2mg_ch_x = array(var_pev2mg_ch_x).reshape(n_ev, n_mg, n_interval)
    var_pev2mg_ch_x = array(model_x.variables.add(names=['ch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_t)
                                                         for i_ev in range(n_ev)
                                                         for j_mg in range(n_mg)
                                                         for k_t in range(n_interval)])
                            ).reshape(n_ev, n_mg, n_interval)

    # discharging power from ev to mg at time span t, 3D-array, (n_ev, n_mg, n_interval)
    var_pev2mg_dch_x = array(model_x.variables.add(names=['dch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_interval)
                                                          for i_ev in range(n_ev)
                                                          for j_mg in range(n_mg)
                                                          for k_interval in range(n_interval)])
                             ).reshape(n_ev, n_mg, n_interval)

    # ev's soc at time span t, 2-D array, (n_ev, n_interval)
    # lb should be 1-d array-like input with the same length as variables
    # .flatten() returns copy while ravel() generally returns view.
    var_ev_soc_x = array(model_x.variables.add(lb=soc_min.ravel(), ub=soc_max.ravel(),
                                               names=['soc_ev{0}_t{1}'.format(i_ev, j_interval)
                                                      for i_ev in range(n_ev)
                                                      for j_interval in range(n_interval)])
                         ).reshape(n_ev, n_interval)

    # charging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_ch_x = array(model_x.variables.add(types=['B'] * (n_ev*n_interval),
                                                names=['sign_ch_ev{0}_t{1}'.format(i_ev, j_interval)
                                                       for i_ev in range(n_ev)
                                                       for j_interval in range(n_interval)])
                          ).reshape(n_ev, n_interval)

    # discharging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_dch_x = array(model_x.variables.add(types=['B'] * (n_ev*n_interval),
                                                 names=['sign_dch_ev{0}_t{1}'.format(i_ev, j_interval)
                                                        for i_ev in range(n_ev)
                                                        for j_interval in range(n_interval)])
                           ).reshape(n_ev, n_interval)

    # arc status for ev at time span t, 3-D array, (n_ev, n_arc, n_interval)
    var_ev_arc_x = array(model_x.variables.add(types=['B'] * n_ev*n_arc*n_interval,
                                               names=['ev{0}_arc{1}_t{2}'.format(i_ev, j_arc, k_interval)
                                                      for i_ev in range(n_ev)
                                                      for j_arc in range(n_arc)
                                                      for k_interval in range(n_interval)])
                        ).reshape(n_ev, n_arc, n_interval)

    # Transit status for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_onroad_x = array(model_x.variables.add(types=['B'] * (n_ev*n_interval),
                                                    names=['sign_onroad_ev{0}_t{1}'.format(i_ev, j_interval)
                                                           for i_ev in range(n_ev)
                                                           for j_interval in range(n_interval)])
                              ).reshape(n_ev, n_interval)

    ## MG model
    # active power generation from MG, 2-D array, (n_mg, n_interval)
    var_pmg_x = array(model_x.variables.add(lb=pg_l.ravel(), ub=(pg_u - localload_p).ravel(),
                                            names=['p_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(n_mg)
                                                   for j_interval in range(n_interval)])
                      ).reshape(n_mg, n_interval)

    # reactive power generation from MG, 2-D array, (n_mg, n_interval)
    var_qmg_x = array(model_x.variables.add(lb=(qg_l-localload_q).ravel(), ub=(qg_u-localload_q).ravel(),
                                            names=['q_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(n_mg)
                                                   for j_interval in range(n_interval)])
                      ).reshape(n_mg, n_interval)

    # the amount of energy of MG, 2-D array, (n_mg, n_interval)
    var_emg_x = array(model_x.variables.add(lb=energy_l.ravel(), ub=energy_u.ravel(),
                                            names=['e_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(n_mg)
                                                   for j_interval in range(n_interval)])
                      ).reshape(n_mg, n_interval)

    # model DS
    # Line active power, 2-D array, (nl, n_interval)
    var_pij_x = array(model_x.variables.add(lb=pij_l.ravel(), ub=pij_u.ravel(),
                                            names=['pij_l{0}_t{1}'.format(i_l, j_interval)
                                                  for i_l in range(nl)
                                                  for j_interval in range(n_interval)])
                      ).reshape(nl, n_interval)

    # Line reactive power, 2-D array, (nl, n_interval)
    var_qij_x = array(model_x.variables.add(lb=qij_l.ravel(), ub=qij_u.ravel(),
                                            names=['qij_l{0}_t{1}'.format(i_l, j_interval)
                                                  for i_l in range(nl)
                                                  for j_interval in range(n_interval)])
                      ).reshape(nl, n_interval)

    # bus voltage, 2-D array, (nb, n_interval)
    var_vm_x = array(model_x.variables.add(lb=vm_l.ravel(), ub=vm_u.ravel(),
                                           names=['vm_l{0}_t{1}'.format(i_b, j_interval)
                                                 for i_b in range(nb)
                                                 for j_interval in range(n_interval)])
                     ).reshape(nb, n_interval)

    # aggregated active power generation at DS level, 2-D array, (ng, n_interval)
    var_pg_x = array(model_x.variables.add(names=['pg{0}_t{1}'.format(i_g, j_interval)
                                                  for i_g in range(ng)
                                                  for j_interval in range(n_interval)])
                     ).reshape(ng, n_interval)
    # aggregated reactive power generation at DS level, 2-D array, (ng, n_interval)
    # lb = full((ng, n_interval), fill_value=-inf).ravel()
    var_qg_x = array(model_x.variables.add(lb=-cpx.infinity*ones((ng, n_interval)).ravel(),
                                           names=['qg{0}_t{1}'.format(i_g, j_interval)
                                                  for i_g in range(ng)
                                                  for j_interval in range(n_interval)])
                     ).reshape(ng, n_interval)
    # var_qg_x = array(var_qg_x)

    # sign for load restoration, 2-D array, (nd, n_interval)
    var_gama_load_x = array(model_x.variables.add(types=[model_x.variables.type.binary] * (nd*n_interval),
                                                  names=['gama_load{0}_t{1}'.format(i_d, j_interval)
                                                         for i_d in range(nd)
                                                         for j_interval in range(n_interval)])
                            ).reshape(nd, n_interval)

    # Line connection status, 1-D array, (nl)
    var_alpha_branch_x = array(model_x.variables.add(types=['B'] * nl,
                                                     names=['alpha_branch{0}'.format(i_l)
                                                            for i_l in range(nl)])
                               ).reshape(nl)

    # Auxiliary variables for line status, 1-D array, (nl)
    var_betaij_x = array(model_x.variables.add(types=['B'] * nl,
                                               names=['betaij_{0}'.format(i_l)
                                                     for i_l in range(nl)])
                         ).reshape(nl)
    var_betaji_x = array(model_x.variables.add(types=['B'] * nl,
                                               names=['betaji_{0}'.format(i_l)
                                                      for i_l in range(nl)])
                         ).reshape(nl)

    # variables for scenario, 1-D array, (n_scenario)
    var_scenario = array(model_x.variables.add(types=['B'] * n_scenario,
                                               names=['scenario_{0}'.format(i_scenario)
                                                      for i_scenario in range(n_scenario)])
                         ).reshape(n_scenario)

    n_all_vars_x = var_scenario[-1] + 1

    # add var_ indices to case_params
    case_params = add_dict_items(case_params, var_pev2mg_ch_x=var_pev2mg_ch_x, var_pev2mg_dch_x=var_pev2mg_dch_x,
                                 var_ev_soc_x=var_ev_soc_x, var_sign_ch_x=var_sign_ch_x, var_sign_dch_x=var_sign_dch_x,
                                 var_ev_arc_x=var_ev_arc_x, var_sign_onroad_x=var_sign_onroad_x, var_pmg_x=var_pmg_x,
                                 var_qmg_x=var_qmg_x, var_emg_x=var_emg_x, var_pij_x=var_pij_x, var_qij_x=var_qij_x,
                                 var_vm_x=var_vm_x, var_pg_x=var_pg_x, var_qg_x=var_qg_x, var_gama_load_x=var_gama_load_x,
                                 var_alpha_branch_x=var_alpha_branch_x, var_betaij_x=var_betaij_x, var_betaji_x=var_betaji_x,
                                 var_scenario=var_scenario)
    # First stage model
    # Add Objective function

    # a = zip(var_sign_onroad_x.ravel().tolist(), [cost_ev_transit] * var_sign_onroad_x.size)
    # In comparison with a = zip(var_sign_onroad_x.tolist(), [cost_ev_transit] * var_sign_onroad_x.size)
    # transportation cost
    model_x.objective.set_linear(zip(var_sign_onroad_x.ravel().tolist(),
                                     [cost_ev_transit] * var_sign_onroad_x.size))
    # charging cost
    model_x.objective.set_linear(zip(var_pev2mg_ch_x.ravel().tolist(),
                                     [cost_ev_power * delta_t * baseMVA * MW_KW] * var_pev2mg_ch_x.size))
    # discharging cost
    model_x.objective.set_linear(zip(var_pev2mg_dch_x.ravel().tolist(),
                                     [cost_ev_power * delta_t * baseMVA * MW_KW] * var_pev2mg_dch_x.size))
    # generation cost
    model_x.objective.set_linear(zip(var_pmg_x.ravel().tolist(),
                                     [cost_mg * delta_t * baseMVA * MW_KW] * var_pmg_x.size))
    # load interruption cost
    model_x.objective.set_linear(zip(var_gama_load_x.ravel().tolist(),
                                     (-load_inter_cost_augment[index_load, :] * pd_u).ravel() * delta_t * baseMVA * MW_KW))

    # Add constraints
    time_start = time.time()  # zeros- 7s, roughly
    model_x_array_a = zeros((0, n_all_vars_x))
    model_x_rhs = []
    model_x_senses = []

    # Each EV only in one status in time span
    aeq_onestatus = zeros((n_ev*n_interval, n_all_vars_x))
    beq_onestatus = ones(n_ev*n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aeq_onestatus[i_ev*n_interval+j_interval, var_ev_arc_x[i_ev, :, j_interval]] = 1

    model_x_array_a = concatenate((model_x_array_a, aeq_onestatus), axis=0)
    model_x_rhs += beq_onestatus.tolist()
    model_x_senses += ['E'] * n_ev*n_interval

    # Constraints for EV transit flow
    aeq_transitflow = zeros((n_ev*n_interval*n_station, n_all_vars_x))
    beq_transitflow = zeros(n_ev*n_interval*n_station)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval-1):
            aeq_transitflow[ix_(i_ev*n_interval*n_station + j_interval*n_station + arange(n_station),
                                var_ev_arc_x[i_ev, :, j_interval])] = station_connection_t.toarray()
            aeq_transitflow[ix_(i_ev*n_interval*n_station + j_interval*n_station + arange(n_station),
                                var_ev_arc_x[i_ev, :, j_interval+1])] = -station_connection_f.toarray()  # station_connection_f needs
            # to be converted into array if aeq_ is array.
        # For t = 0 and intial status, but note it is put at the last.
        aeq_transitflow[ix_(((i_ev+1)*n_interval-1)*n_station + arange(n_station), var_ev_arc_x[i_ev, :, 0])] = station_connection_f.toarray()
        beq_transitflow[((i_ev+1)*n_interval-1)*n_station + arange(n_station)] = ev_position_init[:, i_ev]  # ev_position_init can only be ndarray other than csr_matrix matrix

    model_x_array_a = concatenate((model_x_array_a, aeq_transitflow), axis=0)
    model_x_rhs += beq_transitflow.tolist()
    model_x_senses += ['E'] * n_ev*n_interval*n_station

    # Constraints for the last time interval
    aeq_transitflowend = zeros((n_ev, n_all_vars_x))
    beq_transitflowend = ones(n_ev)

    for i_ev in range(n_ev):
        aeq_transitflowend[i_ev, var_ev_arc_x[i_ev, index_arc_parking, -1]] = 1

    # no need for this constraint in rolling optimization
    model_x_array_a = concatenate((model_x_array_a, aeq_transitflowend), axis=0)
    model_x_rhs += beq_transitflowend.tolist()
    model_x_senses += ['E'] * n_ev

    # EV cannot go back immediately to the same station
    aineq_samestation = zeros((n_ev*n_interval*n_arc_transit, n_all_vars_x))
    bineq_samestation = ones(n_ev*n_interval*n_arc_transit)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval-1):
            aineq_samestation[ix_(i_ev*n_interval*n_arc_transit + j_interval*n_arc_transit + arange(n_arc_transit),
                                  var_ev_arc_x[i_ev, roundtrip_t, j_interval])] = eye(n_arc_transit)
            aineq_samestation[ix_(i_ev*n_interval*n_arc_transit + j_interval*n_arc_transit + arange(n_arc_transit),
                    var_ev_arc_x[i_ev, roundtrip_f, j_interval+1])] = eye(n_arc_transit)

    model_x_array_a = concatenate((model_x_array_a, aineq_samestation), axis=0)
    model_x_rhs += bineq_samestation.tolist()
    model_x_senses += ['L'] * n_ev*n_interval*n_arc_transit

    # charging/discharging with respect to position
    aineq_pchposition = zeros((n_ev*n_interval*n_mg, n_all_vars_x))
    bineq_pchposition = zeros(n_ev*n_interval*n_mg)
    aineq_pdchposition = zeros((n_ev*n_interval*n_mg, n_all_vars_x))
    bineq_pdchposition = zeros(n_ev*n_interval*n_mg)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_pchposition[ix_(i_ev*n_interval*n_mg + j_interval*n_mg + arange(n_mg),
                                  var_pev2mg_ch_x[i_ev, :, j_interval])] = eye(n_mg)
            aineq_pchposition[ix_(i_ev*n_interval*n_mg + j_interval*n_mg + arange(n_mg),
                                  var_ev_arc_x[i_ev, index_arc_parking, j_interval])] = -ev_ch_u * eye(n_mg)

            aineq_pdchposition[ix_(i_ev*n_interval*n_mg + j_interval*n_mg + arange(n_mg),
                                   var_pev2mg_dch_x[i_ev, :, j_interval])] = eye(n_mg)
            aineq_pdchposition[ix_(i_ev*n_interval*n_mg + j_interval*n_mg + arange(n_mg),
                                   var_ev_arc_x[i_ev, index_arc_parking, j_interval])] = -ev_dch_u * eye(n_mg)

    model_x_array_a = concatenate((model_x_array_a, aineq_pchposition, aineq_pdchposition), axis=0)
    model_x_rhs += (bineq_pchposition.tolist() + bineq_pdchposition.tolist())
    model_x_senses += ['L'] * 2*n_ev*n_interval*n_mg

    # charging/discharging with respect to battery status
    aineq_pchstatus = zeros((n_ev*n_interval, n_all_vars_x))
    bineq_pchstatus = zeros(n_ev*n_interval)

    aineq_pdchstatus = zeros((n_ev*n_interval, n_all_vars_x))
    bineq_pdchstatus = zeros(n_ev*n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_pchstatus[i_ev*n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, j_interval]] = 1
            aineq_pchstatus[i_ev*n_interval + j_interval, var_sign_ch_x[i_ev, j_interval]] = -ev_ch_u

            aineq_pdchstatus[i_ev*n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, j_interval]] = 1
            aineq_pdchstatus[i_ev*n_interval + j_interval, var_sign_dch_x[i_ev, j_interval]] = -ev_dch_u

    model_x_array_a = concatenate((model_x_array_a, aineq_pchstatus, aineq_pdchstatus), axis=0)
    model_x_rhs += (bineq_pchstatus.tolist() + bineq_pdchstatus.tolist())
    model_x_senses += ['L'] * 2*n_ev*n_interval

    # charging/discharging status
    aineq_chdchstatus = zeros((n_ev*n_interval, n_all_vars_x))
    bineq_chdchstatus = zeros(n_ev*n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_chdchstatus[i_ev*n_interval + j_interval, var_sign_ch_x[i_ev, j_interval]] = 1
            aineq_chdchstatus[i_ev*n_interval + j_interval, var_ev_arc_x[i_ev, index_arc_parking, j_interval]] = -1

    model_x_array_a = concatenate((model_x_array_a, aineq_chdchstatus), axis=0)
    model_x_rhs += bineq_chdchstatus.tolist()
    model_x_senses += ['L'] * n_ev*n_interval

    # constraint for sign_onroad
    aeq_signonroad = zeros((n_ev*n_interval, n_all_vars_x))
    beq_signonroad = zeros(n_ev*n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aeq_signonroad[i_ev*n_interval + j_interval, var_sign_onroad_x[i_ev, j_interval]] = 1
            aeq_signonroad[i_ev*n_interval + j_interval, var_ev_arc_x[i_ev, index_arc_transit, j_interval]] = -1

    model_x_array_a = concatenate((model_x_array_a, aeq_signonroad), axis=0)
    model_x_rhs += beq_signonroad.tolist()
    model_x_senses += ['E'] * n_ev*n_interval

    # constraints for soc
    aeq_soc = zeros((n_ev*n_interval, n_all_vars_x))
    beq_soc = zeros(n_ev*n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval-1):
            aeq_soc[i_ev*n_interval + j_interval, var_ev_soc_x[i_ev, j_interval+1]] = ev_energy_capacity[i_ev]
            aeq_soc[i_ev*n_interval + j_interval, var_ev_soc_x[i_ev, j_interval]] = -ev_energy_capacity[i_ev]

            aeq_soc[i_ev*n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, j_interval+1]] = -delta_t * ev_ch_efficiency
            aeq_soc[i_ev*n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, j_interval+1]] = delta_t / ev_dch_efficiency

        aeq_soc[i_ev*n_interval + j_interval, var_ev_soc_x[i_ev, 0]] = ev_energy_capacity[i_ev]
        aeq_soc[i_ev*n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, 0]] = -delta_t * ev_ch_efficiency
        aeq_soc[i_ev*n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, 0]] = delta_t / ev_dch_efficiency
        beq_soc[i_ev*n_interval] = ev_energy_init[i_ev]

    model_x_array_a = concatenate((model_x_array_a, aeq_soc), axis=0)
    model_x_rhs += beq_soc.tolist()
    model_x_senses += ['E'] * n_ev*n_interval

    # constraints for generation of MG
    aeq_generationp = zeros((n_mg*n_interval, n_all_vars_x))
    beq_generationp = zeros(n_mg*n_interval)

    aeq_generationq = zeros((n_mg*n_interval, n_all_vars_x))
    beq_generationq = zeros(n_mg*n_interval)

    for i_mg in range(n_mg):
        for j_interval in range(n_interval):
            # pch
            aeq_generationp[i_mg*n_interval + j_interval, var_pev2mg_ch_x[:, i_mg, j_interval]] = 1
            # pdch
            aeq_generationp[i_mg*n_interval + j_interval, var_pev2mg_dch_x[:, i_mg, j_interval]] = -1
            # pmg
            aeq_generationp[i_mg*n_interval + j_interval, var_pmg_x[i_mg, j_interval]] = -1
            # pg ?????
            aeq_generationp[i_mg*n_interval + j_interval, var_pg_x[i_mg, j_interval]] = 1
            # qmg
            aeq_generationq[i_mg*n_interval + j_interval, var_qmg_x[i_mg, j_interval]] = -1
            # qg ?????
            aeq_generationq[i_mg*n_interval + j_interval, var_qg_x[i_mg, j_interval]] = 1

    model_x_array_a = concatenate((model_x_array_a, aeq_generationp, aeq_generationq), axis=0)
    model_x_rhs += beq_generationp.tolist() + beq_generationq.tolist()
    model_x_senses += ['E'] * 2*n_mg*n_interval

    # The amount of energy for each mg in the time point t (end point of time interval t)
    aeq_mgenergy = zeros((n_mg*n_interval, n_all_vars_x))
    beq_mgenergy = zeros(n_mg*n_interval)

    for i_mg in range(n_mg):
        for j_interval in range(n_interval-1):
            # Emg_t+1
            aeq_mgenergy[i_mg*n_interval + j_interval, var_emg_x[i_mg, j_interval+1]] = 1
            # Emg_t
            aeq_mgenergy[i_mg*n_interval + j_interval, var_emg_x[i_mg, j_interval]] = -1
            # pmg
            aeq_mgenergy[i_mg*n_interval + j_interval, var_pmg_x[i_mg, j_interval+1]] = delta_t

        aeq_mgenergy[(i_mg+1)*n_interval - 1, var_emg_x[i_mg, 0]] = 1
        aeq_mgenergy[(i_mg+1)*n_interval - 1, var_pmg_x[i_mg, 0]] = delta_t
        beq_mgenergy[(i_mg+1)*n_interval - 1] = energy_u[i_mg, 0]

    model_x_array_a = concatenate((model_x_array_a, aeq_mgenergy), axis=0)
    model_x_rhs += beq_mgenergy.tolist()
    model_x_senses += ['E'] * n_mg * n_interval

    # topology constraint, |N|-|M|
    aeq_dstree = zeros((1, n_all_vars_x))
    beq_dstree = zeros(1)

    aeq_dstree[:, var_alpha_branch_x] = 1
    beq_dstree[:] = nb - n_mg

    model_x_array_a = concatenate((model_x_array_a, aeq_dstree), axis=0)
    model_x_rhs += beq_dstree.tolist()
    model_x_senses += ['E']

    # topology constraint, bij + bji = alphaij
    aeq_dsbranchstatus = zeros((nl, n_all_vars_x))
    beq_dsbranchstatus = zeros(nl)

    aeq_dsbranchstatus[:, var_alpha_branch_x] = -eye(nl)
    aeq_dsbranchstatus[:, var_betaij_x] = eye(nl)
    aeq_dsbranchstatus[:, var_betaji_x] = eye(nl)

    model_x_array_a = concatenate((model_x_array_a, aeq_dsbranchstatus), axis=0)
    model_x_rhs += beq_dsbranchstatus.tolist()
    model_x_senses += ['E'] * nl

    # topology constraint, exact one parent for each bus other than mg bus
    aeq_dsoneparent = zeros((nb-n_mg, n_all_vars_x))
    beq_dsoneparent = ones(nb-n_mg)

    aeq_dsoneparent[:, var_betaij_x] = connection_f[index_pq_bus, :].toarray()
    aeq_dsoneparent[:, var_betaji_x] = connection_t[index_pq_bus, :].toarray()

    model_x_array_a = concatenate((model_x_array_a, aeq_dsoneparent), axis=0)
    model_x_rhs += beq_dsoneparent.tolist()
    model_x_senses += ['E'] * (nb-n_mg)

    # topology constraint, mg buses has no parent
    n_index_betaij = index_beta_ij.shape[0]
    n_index_betaji = index_beta_ji.shape[0]
    aeq_dsnoparent = zeros((n_index_betaij+n_index_betaji, n_all_vars_x))
    beq_dsnoparent = zeros(n_index_betaij+n_index_betaji)

    # index_beta_ij is different of array and csr_matrix
    aeq_dsnoparent[:n_index_betaij, var_betaij_x[index_beta_ij]] = eye(n_index_betaij)
    aeq_dsnoparent[n_index_betaij:, var_betaji_x[index_beta_ji]] = eye(n_index_betaji)

    model_x_array_a = concatenate((model_x_array_a, aeq_dsnoparent), axis=0)
    model_x_rhs += beq_dsnoparent.tolist()
    model_x_senses += ['E'] * (n_index_betaij+n_index_betaji)

    # power balance
    aeq_dskclp = zeros((n_interval*nb, n_all_vars_x))
    beq_dskclp = zeros(n_interval*nb)
    aeq_dskclq = zeros((n_interval*nb, n_all_vars_x))
    beq_dskclq = zeros(n_interval*nb)

    # aeq_dskclpcoe = concatenate(((connection_t-connection_f).toarray(), zeros((nb, nl)), zeros((nb, nb)), connection_generator.toarray(), zeros((nb, ng)) ), axis=1 )
    # aeq_dskclqcoe = concatenate((zeros((nb, nl)), (connection_t-connection_f).toarray(), zeros((nb, nb)), zeros((nb, ng)), connection_generator.toarray() ), axis=1 )

    for j_interval in range(n_interval):
        # aeq_dskclpcoevar = -connection_load * diag(pd_u[:, j_interval])
        # aeq_dskclqcoevar = -connection_load * diag(qd_u[:, j_interval])
        aeq_dskclp[ix_(j_interval*nb + arange(nb), var_pij_x[:, j_interval])] = (connection_t-connection_f).toarray()
        aeq_dskclp[ix_(j_interval*nb + arange(nb), var_pg_x[:, j_interval])] = connection_generator.toarray()
        aeq_dskclp[ix_(j_interval*nb + arange(nb), var_gama_load_x[:, j_interval])] = -connection_load * diag(pd_u[:, j_interval])

        aeq_dskclq[ix_(j_interval*nb + arange(nb), var_qij_x[:, j_interval])] = (connection_t-connection_f).toarray()
        aeq_dskclq[ix_(j_interval*nb + arange(nb), var_qg_x[:, j_interval])] = connection_generator.toarray()
        aeq_dskclq[ix_(j_interval*nb + arange(nb), var_gama_load_x[:, j_interval])] = -connection_load * diag(qd_u[:, j_interval])

    model_x_array_a = concatenate((model_x_array_a, aeq_dskclp, aeq_dskclq), axis=0)
    model_x_rhs += beq_dskclp.tolist() + beq_dskclq.tolist()
    model_x_senses += ['E'] * 2*nb*n_interval

    # KVL with branch status
    aineq_dskvl_u = zeros((nl * n_interval, n_all_vars_x))
    bineq_dskvl_u = large_m * ones(nl * n_interval)
    aineq_dskvl_l = zeros((nl * n_interval, n_all_vars_x))
    bineq_dskvl_l = large_m * ones(nl * n_interval)

    for j_interval in range(n_interval):
        # v_j^t - v_i^t <= M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_u[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = -diag(branch_r) / v0
        aineq_dskvl_u[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = -diag(branch_x) / v0
        aineq_dskvl_u[ix_(j_interval*nl + arange(nl), var_vm_x[:, j_interval])] = (connection_t.T - connection_f.T).toarray()
        aineq_dskvl_u[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = large_m * eye(nl)

        # v_j^t - v_i^t >= -M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_l[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = diag(branch_r) / v0
        aineq_dskvl_l[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = diag(branch_x) / v0
        aineq_dskvl_l[ix_(j_interval*nl + arange(nl), var_vm_x[:, j_interval])] = (connection_f.T - connection_t.T).toarray()
        aineq_dskvl_l[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = large_m * eye(nl)

    model_x_array_a = concatenate((model_x_array_a, aineq_dskvl_u, aineq_dskvl_l), axis=0)
    model_x_rhs += bineq_dskvl_u.tolist() + bineq_dskvl_l.tolist()
    model_x_senses += ['L'] * 2*nl*n_interval

    # branch power limit pij and qij, respectively
    # alphabranch * -slmax <= pij <= alphabranch * slmax
    aineq_dspij_u = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspij_u = zeros(nl * n_interval)
    aineq_dspij_l = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspij_l = zeros(nl * n_interval)
    # alphabranch * -slmax <= qij <= alphabranch * slmax
    aineq_dsqij_u = zeros((nl * n_interval, n_all_vars_x))
    bineq_dsqij_u = zeros(nl * n_interval)
    aineq_dsqij_l = zeros((nl * n_interval, n_all_vars_x))
    bineq_dsqij_l = zeros(nl * n_interval)

    for j_interval in range(n_interval):
        # pij - alphabranch * slmax <= 0
        aineq_dspij_u[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspij_u[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # -pij - alphabranch * slmax <= 0
        aineq_dspij_l[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspij_l[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # qij - alphabranch * slmax <= 0
        aineq_dsqij_u[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dsqij_u[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # -qij - alphabranch * slmax <= 0
        aineq_dsqij_l[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dsqij_l[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)

    model_x_array_a = concatenate((model_x_array_a, aineq_dspij_u, aineq_dspij_l, aineq_dsqij_u, aineq_dsqij_l), axis=0)
    model_x_rhs += bineq_dspij_u.tolist() + bineq_dspij_l.tolist() + bineq_dsqij_u.tolist() + bineq_dsqij_l.tolist()
    model_x_senses += ['L'] * 4*nl*n_interval

    # Branch power limit pij + qij and pij - qij
    # *** <= pij + qij <= ***
    aineq_dspijaddqij_u = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspijaddqij_u = zeros(nl * n_interval)
    aineq_dspijaddqij_l = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspijaddqij_l = zeros(nl * n_interval)
    # *** <= pij - qij <= ***
    aineq_dspijsubqij_u = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspijsubqij_u = zeros(nl * n_interval)
    aineq_dspijsubqij_l = zeros((nl * n_interval, n_all_vars_x))
    bineq_dspijsubqij_l = zeros(nl * n_interval)

    for j_interval in range(n_interval):
        # pij + qij <= ***
        aineq_dspijaddqij_u[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspijaddqij_u[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dspijaddqij_u[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # *** < pij + qij
        aineq_dspijaddqij_l[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspijaddqij_l[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dspijaddqij_l[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # pij - qij <= ***
        aineq_dspijsubqij_u[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspijsubqij_u[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dspijsubqij_u[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # *** < pij - qij
        aineq_dspijsubqij_l[ix_(j_interval*nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspijsubqij_l[ix_(j_interval*nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dspijsubqij_l[ix_(j_interval*nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)

    model_x_array_a = concatenate((model_x_array_a, aineq_dspijaddqij_u, aineq_dspijaddqij_l, aineq_dspijsubqij_u, aineq_dspijsubqij_l), axis=0)
    model_x_rhs += bineq_dspijaddqij_u.tolist() + bineq_dspijaddqij_l.tolist() + bineq_dspijsubqij_u.tolist() + bineq_dspijsubqij_l.tolist()
    model_x_senses += ['L'] * 4*nl*n_interval
    time_elapsed = time.time() - time_start
    # Add constraints
    # model_x_array_a = csr_matrix(model_x_array_a)
    a_rows = model_x_array_a.nonzero()[0].tolist()
    a_cols = model_x_array_a.nonzero()[1].tolist()
    # a_data = model_x_array_a.data.tolist()  # model_x_array_a is csr_matrix matrix, a_data element needs to be float
    # start = time.time()
    # a_vals = model_x_array_a[model_x_array_a.nonzero()]
    # elasped1 = time.time()-start
    # start = time.time()
    # a_vals = model_x_array_a[model_x_array_a != 0]  # boolean mask index array, it's faster than nonzero, but result is different?
                                                        # Note that boolean mask returns 1D-array
    a_vals = model_x_array_a[a_rows, a_cols].tolist()  # faster than boolean mask
    # elpased2 = time.time() - start
    model_x.linear_constraints.add(rhs=model_x_rhs, senses=model_x_senses,
                                   names=['constraint{0}'.format(i)
                                          for i in range(len(model_x_rhs))])

    model_x.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))
    # set objective sense
    model_x.objective.set_sense(model_x.objective.sense.minimize)
    # Benders  decomposition
    # Solving with automatic Benders decomposition.
    # By setting the Benders strategy parameter to Full, CPLEX will put all integer variables into the master, all
    # continuous variables into a sub-problem, and further decompose that sub-problem, if possible.
    # model_x.parameters.benders.strategy.set(model_x.parameters.benders.strategy.values.full)

    return model_x


def set_model_cpx_2(case_params):
    '''
    csr_matrix and lil_matrix substitute for nd-array. Sep. 11, 2018
    :param case_params:
    :return:
    '''

    import cplex as cpx
    # Extract parameters
    # dynamic allocation with locals() doesn't work in python3
    # names = locals()
    # for i in range(3):
    #     names['a'+str(i)]=i
    #     print('a'+str(i))
    #     exec("")

    # for element in case_params.items():
    #     # Execute nb = nb
    #     # exec(element[0] + '=' + 'element[1]')
    #     # Why locals() doesn't work while globals works?
    #      globals()[element[0]] = element[1]  # it's global variables, any alternatives?

    params_keys = case_params.keys()
    params_values = case_params.values()

    (ppc, nb, nl, ng, n_mg, nd, n_arc, n_ev, n_interval, n_scenario, n_station, cost_ev_transit, cost_ev_power, delta_t,
     sn_mva, MW_KW, scenario_weight, cost_mg, load_inter_cost, index_load, soc_min, soc_max, pg_l, qg_l, localload_p,
     localload_q, energy_u, energy_l, pg_u, qg_u, vm_l, vm_u, slmax, pd_u, qd_u, station_connection_t,
     station_connection_f,
     ev_position_init, n_arc_transit, roundtrip_f, roundtrip_t, index_arc_parking, ev_ch_u, ev_dch_u,
     ev_energy_capacity,
     ev_energy_init, ev_energy_l, ev_energy_u, index_arc_transit, ev_ch_efficiency, ev_dch_efficiency, connection_f,
     connection_t, index_pq_bus, index_beta_ij, index_beta_ji, connection_generator, connection_load, large_m, branch_r,
     branch_x, v0, load_qp_ratio, soc_init, load_inter_cost_augment, n_timewindow_c) = (
    case_params['ppc'], case_params['nb'],
    case_params['nl'], case_params['ng'], case_params['n_mg'], case_params['nd'], case_params['n_arc'],
    case_params['n_ev'], case_params['n_interval'], case_params['n_scenario'], case_params['n_station'],
    case_params['cost_ev_transit'], case_params['cost_ev_power'], case_params['delta_t'], case_params['sn_mva'],
    case_params['MW_KW'], case_params['scenario_weight'], case_params['cost_mg'], case_params['load_inter_cost'],
    case_params['index_load'], case_params['soc_min'], case_params['soc_max'], case_params['pg_l'], case_params['qg_l'],
    case_params['localload_p'], case_params['localload_q'], case_params['energy_u'], case_params['energy_l'],
    case_params['pg_u'], case_params['qg_u'], case_params['vm_l'], case_params['vm_u'], case_params['slmax'],
    case_params['pd_u'],
    case_params['qd_u'], case_params['station_connection_t'], case_params['station_connection_f'],
    case_params['ev_position_init'],
    case_params['n_arc_transit'], case_params['roundtrip_f'], case_params['roundtrip_t'],
    case_params['index_arc_parking'],
    case_params['ev_ch_u'], case_params['ev_dch_u'], case_params['ev_energy_capacity'], case_params['ev_energy_init'],
    case_params['ev_energy_l'], case_params['ev_energy_u'], case_params['index_arc_transit'],
    case_params['ev_ch_efficiency'],
    case_params['ev_dch_efficiency'], case_params['connection_f'], case_params['connection_t'],
    case_params['index_pq_bus'],
    case_params['index_beta_ij'], case_params['index_beta_ji'], case_params['connection_generator'],
    case_params['connection_load'],
    case_params['large_m'], case_params['branch_r'], case_params['branch_x'], case_params['v0'],
    case_params['load_qp_ratio'],
    case_params['soc_init'], case_params['load_inter_cost_augment'], case_params['n_timewindow_c'])

    (pij_l, qij_l, pij_u, qij_u, index_genbus) = (case_params['pij_l'], case_params['qij_l'], case_params['pij_u'],
                                                  case_params['qij_u'], case_params['index_genbus'])

    ppnet = case_params['ppnet']

    # Build model
    # Initialization of cplex
    model_x = cpx.Cplex()

    ## Define varialbles and get position array
    # prefix var_ is for variables index array
    # TESS model
    # charging power from mg to ev at time span t, 3D-array, (n_ev, n_mg, n_interval)
    # var_pev2mg_ch_x = model_x.variables.add(names=['ch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_t)
    #                                                  for i_ev in range(n_ev)
    #                                                  for j_mg in range(n_mg)
    #                                                  for k_t in range(n_interval)])
    # var_pev2mg_ch_x = array(var_pev2mg_ch_x).reshape(n_ev, n_mg, n_interval)
    var_pev2mg_ch_x = array(model_x.variables.add(names=['ch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_t)
                                                         for i_ev in range(n_ev)
                                                         for j_mg in range(n_mg)
                                                         for k_t in range(n_interval)])
                            ).reshape(n_ev, n_mg, n_interval)

    # discharging power from ev to mg at time span t, 3D-array, (n_ev, n_mg, n_interval)
    var_pev2mg_dch_x = array(model_x.variables.add(names=['dch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_interval)
                                                          for i_ev in range(n_ev)
                                                          for j_mg in range(n_mg)
                                                          for k_interval in range(n_interval)])
                             ).reshape(n_ev, n_mg, n_interval)

    # ev's soc at time span t, 2-D array, (n_ev, n_interval)
    # lb should be 1-d array-like input with the same length as variables
    # .flatten() returns copy while ravel() generally returns view.
    var_ev_soc_x = array(model_x.variables.add(lb=soc_min.ravel(), ub=soc_max.ravel(),
                                               names=['soc_ev{0}_t{1}'.format(i_ev, j_interval)
                                                      for i_ev in range(n_ev)
                                                      for j_interval in range(n_interval)])
                         ).reshape(n_ev, n_interval)

    # charging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_ch_x = array(model_x.variables.add(types=['B'] * (n_ev * n_interval),
                                                names=['sign_ch_ev{0}_t{1}'.format(i_ev, j_interval)
                                                       for i_ev in range(n_ev)
                                                       for j_interval in range(n_interval)])
                          ).reshape(n_ev, n_interval)

    # discharging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_dch_x = array(model_x.variables.add(types=['B'] * (n_ev * n_interval),
                                                 names=['sign_dch_ev{0}_t{1}'.format(i_ev, j_interval)
                                                        for i_ev in range(n_ev)
                                                        for j_interval in range(n_interval)])
                           ).reshape(n_ev, n_interval)

    # arc status for ev at time span t, 3-D array, (n_ev, n_arc, n_interval)
    var_ev_arc_x = array(model_x.variables.add(types=['B'] * n_ev * n_arc * n_interval,
                                               names=['ev{0}_arc{1}_t{2}'.format(i_ev, j_arc, k_interval)
                                                      for i_ev in range(n_ev)
                                                      for j_arc in range(n_arc)
                                                      for k_interval in range(n_interval)])
                         ).reshape(n_ev, n_arc, n_interval)

    # Transit status for ev at time span t, 2-D array, (n_ev, n_interval)
    var_sign_onroad_x = array(model_x.variables.add(types=['B'] * (n_ev * n_interval),
                                                    names=['sign_onroad_ev{0}_t{1}'.format(i_ev, j_interval)
                                                           for i_ev in range(n_ev)
                                                           for j_interval in range(n_interval)])
                              ).reshape(n_ev, n_interval)

    ## MG model
    # active power generation from MG, 2-D array, (n_mg, n_interval)
    var_pmg_x = array(model_x.variables.add(lb=pg_l.ravel(), ub=(pg_u - localload_p).ravel(),
                                            names=['p_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(ng)
                                                   for j_interval in range(n_interval)])
                      ).reshape(ng, n_interval)

    # reactive power generation from MG, 2-D array, (n_mg, n_interval)
    var_qmg_x = array(model_x.variables.add(lb=(qg_l - localload_q).ravel(), ub=(qg_u - localload_q).ravel(),
                                            names=['q_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(ng)
                                                   for j_interval in range(n_interval)])
                      ).reshape(ng, n_interval)

    # the amount of energy of MG, 2-D array, (n_mg, n_interval)
    var_emg_x = array(model_x.variables.add(lb=energy_l.ravel(), ub=energy_u.ravel(),
                                            names=['e_mg{0}_t{1}'.format(i_mg, j_interval)
                                                   for i_mg in range(ng)
                                                   for j_interval in range(n_interval)])
                      ).reshape(ng, n_interval)

    # model DS
    # Line active power, 2-D array, (nl, n_interval)
    var_pij_x = array(model_x.variables.add(lb=pij_l.ravel(), ub=pij_u.ravel(),
                                            names=['pij_l{0}_t{1}'.format(i_l, j_interval)
                                                   for i_l in range(nl)
                                                   for j_interval in range(n_interval)])
                      ).reshape(nl, n_interval)

    # Line reactive power, 2-D array, (nl, n_interval)
    var_qij_x = array(model_x.variables.add(lb=qij_l.ravel(), ub=qij_u.ravel(),
                                            names=['qij_l{0}_t{1}'.format(i_l, j_interval)
                                                   for i_l in range(nl)
                                                   for j_interval in range(n_interval)])
                      ).reshape(nl, n_interval)

    # bus voltage, 2-D array, (nb, n_interval)
    var_vm_x = array(model_x.variables.add(lb=vm_l.ravel(), ub=vm_u.ravel(),
                                           names=['vm_l{0}_t{1}'.format(i_b, j_interval)
                                                  for i_b in range(nb)
                                                  for j_interval in range(n_interval)])
                     ).reshape(nb, n_interval)

    # aggregated active power generation at DS level, 2-D array, (ng, n_interval)
    var_pg_x = array(model_x.variables.add(names=['pg{0}_t{1}'.format(i_g, j_interval)
                                                  for i_g in range(ng)
                                                  for j_interval in range(n_interval)])
                     ).reshape(ng, n_interval)
    # aggregated reactive power generation at DS level, 2-D array, (ng, n_interval)
    # lb = full((ng, n_interval), fill_value=-inf).ravel()
    var_qg_x = array(model_x.variables.add(lb=-cpx.infinity * ones((ng, n_interval)).ravel(),
                                           names=['qg{0}_t{1}'.format(i_g, j_interval)
                                                  for i_g in range(ng)
                                                  for j_interval in range(n_interval)])
                     ).reshape(ng, n_interval)
    # var_qg_x = array(var_qg_x)

    # sign for load restoration, 2-D array, (nd, n_interval)
    var_gama_load_x = array(model_x.variables.add(types=[model_x.variables.type.binary] * (nd * n_interval),
                                                  names=['gama_load{0}_t{1}'.format(i_d, j_interval)
                                                         for i_d in range(nd)
                                                         for j_interval in range(n_interval)])
                            ).reshape(nd, n_interval)

    # Line connection status, 1-D array, (nl)
    var_alpha_branch_x = array(model_x.variables.add(types=['B'] * nl,
                                                     names=['alpha_branch{0}'.format(i_l)
                                                            for i_l in range(nl)])
                               ).reshape(nl)

    # Auxiliary variables for line status, 1-D array, (nl)
    var_betaij_x = array(model_x.variables.add(types=['B'] * nl,
                                               names=['betaij_{0}'.format(i_l)
                                                      for i_l in range(nl)])
                         ).reshape(nl)
    var_betaji_x = array(model_x.variables.add(types=['B'] * nl,
                                               names=['betaji_{0}'.format(i_l)
                                                      for i_l in range(nl)])
                         ).reshape(nl)

    # variables for scenario, 1-D array, (n_scenario)
    var_scenario = array(model_x.variables.add(types=['B'] * n_scenario,
                                               names=['scenario_{0}'.format(i_scenario)
                                                      for i_scenario in range(n_scenario)])
                         ).reshape(n_scenario)

    n_all_vars_x = var_scenario[-1] + 1

    # add var_ indices to case_params
    case_params = add_dict_items(case_params, var_pev2mg_ch_x=var_pev2mg_ch_x, var_pev2mg_dch_x=var_pev2mg_dch_x,
                                 var_ev_soc_x=var_ev_soc_x, var_sign_ch_x=var_sign_ch_x, var_sign_dch_x=var_sign_dch_x,
                                 var_ev_arc_x=var_ev_arc_x, var_sign_onroad_x=var_sign_onroad_x, var_pmg_x=var_pmg_x,
                                 var_qmg_x=var_qmg_x, var_emg_x=var_emg_x, var_pij_x=var_pij_x, var_qij_x=var_qij_x,
                                 var_vm_x=var_vm_x, var_pg_x=var_pg_x, var_qg_x=var_qg_x,
                                 var_gama_load_x=var_gama_load_x,
                                 var_alpha_branch_x=var_alpha_branch_x, var_betaij_x=var_betaij_x,
                                 var_betaji_x=var_betaji_x,
                                 var_scenario=var_scenario)
    # First stage model
    # Add Objective function

    # a = zip(var_sign_onroad_x.ravel().tolist(), [cost_ev_transit] * var_sign_onroad_x.size)
    # In comparison with a = zip(var_sign_onroad_x.tolist(), [cost_ev_transit] * var_sign_onroad_x.size)
    # transportation cost
    model_x.objective.set_linear(zip(var_sign_onroad_x.ravel().tolist(),
                                     [cost_ev_transit] * var_sign_onroad_x.size))
    # charging cost
    model_x.objective.set_linear(zip(var_pev2mg_ch_x.ravel().tolist(),
                                     [cost_ev_power * delta_t * sn_mva * MW_KW] * var_pev2mg_ch_x.size))
    # discharging cost
    model_x.objective.set_linear(zip(var_pev2mg_dch_x.ravel().tolist(),
                                     [cost_ev_power * delta_t * sn_mva * MW_KW] * var_pev2mg_dch_x.size))
    # generation cost
    model_x.objective.set_linear(zip(var_pmg_x.ravel().tolist(),
                                     [cost_mg * delta_t * sn_mva * MW_KW] * var_pmg_x.size))
    # load interruption cost
    model_x.objective.set_linear(zip(var_gama_load_x.ravel().tolist(),
                                     (-load_inter_cost_augment * pd_u).ravel() * delta_t * sn_mva * MW_KW))

    # Add constraints
    # It turns to be more efficient to set up A incrementally as coo_matrix while b as list
    # aeq is ili_matrix while beq is nd-array
    time_start = time.time()  # zeros- 7s, roughly, lil_matrix- 0.7s, roughly
    model_x_matrix_a = coo_matrix((0, n_all_vars_x))
    model_x_rhs = []
    model_x_senses = []

    # Each EV only in one status in time span
    aeq_onestatus = lil_matrix((n_ev * n_interval, n_all_vars_x))
    beq_onestatus = ones((n_ev * n_interval))

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aeq_onestatus[i_ev * n_interval + j_interval, var_ev_arc_x[i_ev, :, j_interval]] = 1

    # model_x_matrix_a = concatenate((model_x_matrix_a, aeq_onestatus), axis=0)
    model_x_matrix_a = vstack([model_x_matrix_a, aeq_onestatus])
    # model_x_rhs = hstack([model_x_rhs, beq_onestatus])
    model_x_rhs += beq_onestatus.tolist()
    model_x_senses += ['E'] * n_ev * n_interval

    # Constraints for EV transit flow
    aeq_transitflow = lil_matrix((n_ev * n_interval * n_station, n_all_vars_x))
    beq_transitflow = zeros(n_ev * n_interval * n_station)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval - 1):
            aeq_transitflow[ix_(i_ev * n_interval * n_station + j_interval * n_station + arange(n_station),
                                var_ev_arc_x[i_ev, :, j_interval])] = station_connection_t
            aeq_transitflow[ix_(i_ev * n_interval * n_station + j_interval * n_station + arange(n_station),
                                var_ev_arc_x[i_ev, :,
                                j_interval + 1])] = -station_connection_f  # station_connection_f needs
            # to be converted into array if aeq_ is array.
        # For t = 0 and intial status, but note it is put at the last.
        aeq_transitflow[ix_(((i_ev + 1) * n_interval - 1) * n_station + arange(n_station),
                            var_ev_arc_x[i_ev, :, 0])] = station_connection_f
        beq_transitflow[((i_ev + 1) * n_interval - 1) * n_station + arange(n_station)] = ev_position_init[:,
                                                                                         i_ev]  # ev_position_init can only be ndarray other than csr_matrix matrix

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_transitflow])
    model_x_rhs += beq_transitflow.tolist()
    model_x_senses += ['E'] * n_ev * n_interval * n_station

    # Constraints for the last time interval
    aeq_transitflowend = lil_matrix((n_ev, n_all_vars_x))
    beq_transitflowend = ones(n_ev)

    for i_ev in range(n_ev):
        aeq_transitflowend[i_ev, var_ev_arc_x[i_ev, index_arc_parking, -1]] = 1

    # no need for this constraint in rolling optimization
    model_x_matrix_a = vstack([model_x_matrix_a, aeq_transitflowend])
    model_x_rhs += beq_transitflowend.tolist()
    model_x_senses += ['E'] * n_ev

    # EV cannot go back immediately to the same station
    aineq_samestation = lil_matrix((n_ev * n_interval * n_arc_transit, n_all_vars_x))
    bineq_samestation = ones(n_ev * n_interval * n_arc_transit)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval - 1):
            aineq_samestation[
                ix_(i_ev * n_interval * n_arc_transit + j_interval * n_arc_transit + arange(n_arc_transit),
                    var_ev_arc_x[i_ev, roundtrip_t, j_interval])] = eye(n_arc_transit)
            aineq_samestation[
                ix_(i_ev * n_interval * n_arc_transit + j_interval * n_arc_transit + arange(n_arc_transit),
                    var_ev_arc_x[i_ev, roundtrip_f, j_interval + 1])] = eye(n_arc_transit)

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_samestation])
    model_x_rhs += bineq_samestation.tolist()
    model_x_senses += ['L'] * n_ev * n_interval * n_arc_transit

    # charging/discharging with respect to position
    aineq_pchposition = lil_matrix((n_ev * n_interval * n_mg, n_all_vars_x))
    bineq_pchposition = zeros(n_ev * n_interval * n_mg)
    aineq_pdchposition = lil_matrix((n_ev * n_interval * n_mg, n_all_vars_x))
    bineq_pdchposition = zeros(n_ev * n_interval * n_mg)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_pchposition[ix_(i_ev * n_interval * n_mg + j_interval * n_mg + arange(n_mg),
                                  var_pev2mg_ch_x[i_ev, :, j_interval])] = eye(n_mg)
            aineq_pchposition[ix_(i_ev * n_interval * n_mg + j_interval * n_mg + arange(n_mg),
                                  var_ev_arc_x[i_ev, index_arc_parking, j_interval])] = -ev_ch_u * eye(n_mg)

            aineq_pdchposition[ix_(i_ev * n_interval * n_mg + j_interval * n_mg + arange(n_mg),
                                   var_pev2mg_dch_x[i_ev, :, j_interval])] = eye(n_mg)
            aineq_pdchposition[ix_(i_ev * n_interval * n_mg + j_interval * n_mg + arange(n_mg),
                                   var_ev_arc_x[i_ev, index_arc_parking, j_interval])] = -ev_dch_u * eye(n_mg)

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_pchposition, aineq_pdchposition])
    model_x_rhs += (bineq_pchposition.tolist() + bineq_pdchposition.tolist())
    model_x_senses += ['L'] * 2 * n_ev * n_interval * n_mg

    # charging/discharging with respect to battery status
    aineq_pchstatus = lil_matrix((n_ev * n_interval, n_all_vars_x))
    bineq_pchstatus = zeros(n_ev * n_interval)

    aineq_pdchstatus = lil_matrix((n_ev * n_interval, n_all_vars_x))
    bineq_pdchstatus = zeros(n_ev * n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_pchstatus[i_ev * n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, j_interval]] = 1
            aineq_pchstatus[i_ev * n_interval + j_interval, var_sign_ch_x[i_ev, j_interval]] = -ev_ch_u

            aineq_pdchstatus[i_ev * n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, j_interval]] = 1
            aineq_pdchstatus[i_ev * n_interval + j_interval, var_sign_dch_x[i_ev, j_interval]] = -ev_dch_u

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_pchstatus, aineq_pdchstatus])
    model_x_rhs += (bineq_pchstatus.tolist() + bineq_pdchstatus.tolist())
    model_x_senses += ['L'] * 2 * n_ev * n_interval

    # charging/discharging status
    aineq_chdchstatus = lil_matrix((n_ev * n_interval, n_all_vars_x))
    bineq_chdchstatus = zeros(n_ev * n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aineq_chdchstatus[i_ev * n_interval + j_interval, var_sign_ch_x[i_ev, j_interval]] = 1
            aineq_chdchstatus[i_ev * n_interval + j_interval, var_ev_arc_x[i_ev, index_arc_parking, j_interval]] = -1

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_chdchstatus])
    model_x_rhs += bineq_chdchstatus.tolist()
    model_x_senses += ['L'] * n_ev * n_interval

    # constraint for sign_onroad
    aeq_signonroad = lil_matrix((n_ev * n_interval, n_all_vars_x))
    beq_signonroad = zeros(n_ev * n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval):
            aeq_signonroad[i_ev * n_interval + j_interval, var_sign_onroad_x[i_ev, j_interval]] = 1
            aeq_signonroad[i_ev * n_interval + j_interval, var_ev_arc_x[i_ev, index_arc_transit, j_interval]] = -1

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_signonroad])
    model_x_rhs += beq_signonroad.tolist()
    model_x_senses += ['E'] * n_ev * n_interval

    # constraints for soc
    aeq_soc = lil_matrix((n_ev * n_interval, n_all_vars_x))
    beq_soc = zeros(n_ev * n_interval)

    for i_ev in range(n_ev):
        for j_interval in range(n_interval - 1):
            aeq_soc[i_ev * n_interval + j_interval, var_ev_soc_x[i_ev, j_interval + 1]] = ev_energy_capacity[i_ev]
            aeq_soc[i_ev * n_interval + j_interval, var_ev_soc_x[i_ev, j_interval]] = -ev_energy_capacity[i_ev]

            aeq_soc[
                i_ev * n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, j_interval + 1]] = -delta_t * ev_ch_efficiency
            aeq_soc[
                i_ev * n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, j_interval + 1]] = delta_t / ev_dch_efficiency

        aeq_soc[i_ev * n_interval + j_interval, var_ev_soc_x[i_ev, 0]] = ev_energy_capacity[i_ev]
        aeq_soc[i_ev * n_interval + j_interval, var_pev2mg_ch_x[i_ev, :, 0]] = -delta_t * ev_ch_efficiency
        aeq_soc[i_ev * n_interval + j_interval, var_pev2mg_dch_x[i_ev, :, 0]] = delta_t / ev_dch_efficiency
        beq_soc[i_ev * n_interval] = ev_energy_init[i_ev]

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_soc])
    model_x_rhs += beq_soc.tolist()
    model_x_senses += ['E'] * n_ev * n_interval

    # constraints for generation of MG
    # todo staion and mg need to be maped and linked
    aeq_generationp = lil_matrix((n_mg * n_interval, n_all_vars_x))
    beq_generationp = zeros(n_mg * n_interval)

    aeq_generationq = lil_matrix((n_mg * n_interval, n_all_vars_x))
    beq_generationq = zeros(n_mg * n_interval)

    for i_mg in range(n_mg):
        for j_interval in range(n_interval):
            # pch
            aeq_generationp[i_mg * n_interval + j_interval, var_pev2mg_ch_x[:, i_mg, j_interval]] = 1
            # pdch
            aeq_generationp[i_mg * n_interval + j_interval, var_pev2mg_dch_x[:, i_mg, j_interval]] = -1
            # pmg
            aeq_generationp[i_mg * n_interval + j_interval, var_pmg_x[i_mg, j_interval]] = -1
            # pg ?????
            aeq_generationp[i_mg * n_interval + j_interval, var_pg_x[i_mg, j_interval]] = 1
            # qmg
            aeq_generationq[i_mg * n_interval + j_interval, var_qmg_x[i_mg, j_interval]] = -1
            # qg ?????
            aeq_generationq[i_mg * n_interval + j_interval, var_qg_x[i_mg, j_interval]] = 1

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_generationp, aeq_generationq])
    model_x_rhs += beq_generationp.tolist() + beq_generationq.tolist()
    model_x_senses += ['E'] * 2 * n_mg * n_interval

    # The amount of energy for each mg in the time point t (end point of time interval t)
    aeq_mgenergy = lil_matrix((ng * n_interval, n_all_vars_x))
    beq_mgenergy = zeros(ng * n_interval)

    for i_mg in range(ng):
        for j_interval in range(n_interval - 1):
            # Emg_t+1
            aeq_mgenergy[i_mg * n_interval + j_interval, var_emg_x[i_mg, j_interval + 1]] = 1
            # Emg_t
            aeq_mgenergy[i_mg * n_interval + j_interval, var_emg_x[i_mg, j_interval]] = -1
            # pmg
            aeq_mgenergy[i_mg * n_interval + j_interval, var_pmg_x[i_mg, j_interval + 1]] = delta_t

        aeq_mgenergy[(i_mg + 1) * n_interval - 1, var_emg_x[i_mg, 0]] = 1
        aeq_mgenergy[(i_mg + 1) * n_interval - 1, var_pmg_x[i_mg, 0]] = delta_t
        beq_mgenergy[(i_mg + 1) * n_interval - 1] = energy_u[i_mg, 0]

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_mgenergy])  # error because of indentation
    model_x_rhs += beq_mgenergy.tolist()
    model_x_senses += ['E'] * ng * n_interval

    # topology constraint, |N|-|M|
    aeq_dstree = lil_matrix((1, n_all_vars_x))
    beq_dstree = zeros(1)

    aeq_dstree[:, var_alpha_branch_x] = 1
    beq_dstree[:] = nb - ng

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_dstree])
    model_x_rhs += beq_dstree.tolist()
    model_x_senses += ['E']

    # topology constraint, bij + bji = alphaij
    aeq_dsbranchstatus = lil_matrix((nl, n_all_vars_x))
    beq_dsbranchstatus = zeros(nl)

    aeq_dsbranchstatus[:, var_alpha_branch_x] = -eye(nl)
    aeq_dsbranchstatus[:, var_betaij_x] = eye(nl)
    aeq_dsbranchstatus[:, var_betaji_x] = eye(nl)

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_dsbranchstatus])
    model_x_rhs += beq_dsbranchstatus.tolist()
    model_x_senses += ['E'] * nl

    # topology constraint, exact one parent for each bus other than mg bus
    aeq_dsoneparent = lil_matrix((nb - ng, n_all_vars_x))
    beq_dsoneparent = ones(nb - ng)

    aeq_dsoneparent[:, var_betaij_x] = connection_f[index_pq_bus, :].toarray()
    aeq_dsoneparent[:, var_betaji_x] = connection_t[index_pq_bus, :].toarray()

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_dsoneparent])
    model_x_rhs += beq_dsoneparent.tolist()
    model_x_senses += ['E'] * (nb - ng)

    # topology constraint, mg buses has no parent
    n_index_betaij = index_beta_ij.shape[0]
    n_index_betaji = index_beta_ji.shape[0]
    aeq_dsnoparent = lil_matrix((n_index_betaij + n_index_betaji, n_all_vars_x))
    beq_dsnoparent = zeros(n_index_betaij + n_index_betaji)

    # index_beta_ij is different of array and csr_matrix
    aeq_dsnoparent[:n_index_betaij, var_betaij_x[index_beta_ij]] = eye(n_index_betaij)
    aeq_dsnoparent[n_index_betaij:, var_betaji_x[index_beta_ji]] = eye(n_index_betaji)

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_dsnoparent])
    model_x_rhs += beq_dsnoparent.tolist()
    model_x_senses += ['E'] * (n_index_betaij + n_index_betaji)

    # power balance
    aeq_dskclp = lil_matrix((n_interval * nb, n_all_vars_x))
    beq_dskclp = zeros(n_interval * nb)
    aeq_dskclq = lil_matrix((n_interval * nb, n_all_vars_x))
    beq_dskclq = zeros(n_interval * nb)

    # aeq_dskclpcoe = vstack(((connection_t-connection_f).toarray(), zeros((nb, nl)), zeros((nb, nb)), connection_generator.toarray(), zeros((nb, ng)) ), axis=1 )
    # aeq_dskclqcoe = vstack((zeros((nb, nl)), (connection_t-connection_f).toarray(), zeros((nb, nb)), zeros((nb, ng)), connection_generator.toarray() ), axis=1 )

    for j_interval in range(n_interval):
        # aeq_dskclpcoevar = -connection_load * diag(pd_u[:, j_interval])
        # aeq_dskclqcoevar = -connection_load * diag(qd_u[:, j_interval])
        aeq_dskclp[ix_(j_interval * nb + arange(nb), var_pij_x[:, j_interval])] = (
                    connection_t - connection_f).toarray()
        aeq_dskclp[ix_(j_interval * nb + arange(nb), var_pg_x[:, j_interval])] = connection_generator.toarray()
        aeq_dskclp[ix_(j_interval * nb + arange(nb), var_gama_load_x[:, j_interval])] = -connection_load * diag(
            pd_u[:, j_interval])

        aeq_dskclq[ix_(j_interval * nb + arange(nb), var_qij_x[:, j_interval])] = (
                    connection_t - connection_f).toarray()
        aeq_dskclq[ix_(j_interval * nb + arange(nb), var_qg_x[:, j_interval])] = connection_generator.toarray()
        aeq_dskclq[ix_(j_interval * nb + arange(nb), var_gama_load_x[:, j_interval])] = -connection_load * diag(
            qd_u[:, j_interval])

    model_x_matrix_a = vstack([model_x_matrix_a, aeq_dskclp, aeq_dskclq])
    model_x_rhs += beq_dskclp.tolist() + beq_dskclq.tolist()
    model_x_senses += ['E'] * 2 * nb * n_interval

    # KVL with branch status
    aineq_dskvl_u = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dskvl_u = large_m * ones(nl * n_interval)
    aineq_dskvl_l = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dskvl_l = large_m * ones(nl * n_interval)

    for j_interval in range(n_interval):
        # v_j^t - v_i^t <= M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_u[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = -diag(branch_r) / v0
        aineq_dskvl_u[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = -diag(branch_x) / v0
        aineq_dskvl_u[ix_(j_interval * nl + arange(nl), var_vm_x[:, j_interval])] = (
                    connection_t.T - connection_f.T).toarray()
        aineq_dskvl_u[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = large_m * eye(nl)

        # v_j^t - v_i^t >= -M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_l[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = diag(branch_r) / v0
        aineq_dskvl_l[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = diag(branch_x) / v0
        aineq_dskvl_l[ix_(j_interval * nl + arange(nl), var_vm_x[:, j_interval])] = (
                    connection_f.T - connection_t.T).toarray()
        aineq_dskvl_l[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = large_m * eye(nl)

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_dskvl_u, aineq_dskvl_l])
    model_x_rhs += bineq_dskvl_u.tolist() + bineq_dskvl_l.tolist()
    model_x_senses += ['L'] * 2 * nl * n_interval

    # branch power limit pij and qij, respectively
    # alphabranch * -slmax <= pij <= alphabranch * slmax
    aineq_dspij_u = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspij_u = zeros(nl * n_interval)
    aineq_dspij_l = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspij_l = zeros(nl * n_interval)
    # alphabranch * -slmax <= qij <= alphabranch * slmax
    aineq_dsqij_u = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dsqij_u = zeros(nl * n_interval)
    aineq_dsqij_l = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dsqij_l = zeros(nl * n_interval)

    for j_interval in range(n_interval):
        # pij - alphabranch * slmax <= 0
        aineq_dspij_u[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspij_u[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # -pij - alphabranch * slmax <= 0
        aineq_dspij_l[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspij_l[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # qij - alphabranch * slmax <= 0
        aineq_dsqij_u[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dsqij_u[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)
        # -qij - alphabranch * slmax <= 0
        aineq_dsqij_l[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dsqij_l[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -diag(slmax)

    model_x_matrix_a = vstack([model_x_matrix_a, aineq_dspij_u, aineq_dspij_l, aineq_dsqij_u, aineq_dsqij_l])
    model_x_rhs += bineq_dspij_u.tolist() + bineq_dspij_l.tolist() + bineq_dsqij_u.tolist() + bineq_dsqij_l.tolist()
    model_x_senses += ['L'] * 4 * nl * n_interval

    # Branch power limit pij + qij and pij - qij
    # *** <= pij + qij <= ***
    aineq_dspijaddqij_u = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspijaddqij_u = zeros(nl * n_interval)
    aineq_dspijaddqij_l = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspijaddqij_l = zeros(nl * n_interval)
    # *** <= pij - qij <= ***
    aineq_dspijsubqij_u = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspijsubqij_u = zeros(nl * n_interval)
    aineq_dspijsubqij_l = lil_matrix((nl * n_interval, n_all_vars_x))
    bineq_dspijsubqij_l = zeros(nl * n_interval)

    for j_interval in range(n_interval):
        # pij + qij <= ***
        aineq_dspijaddqij_u[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspijaddqij_u[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dspijaddqij_u[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # *** < pij + qij
        aineq_dspijaddqij_l[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspijaddqij_l[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dspijaddqij_l[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # pij - qij <= ***
        aineq_dspijsubqij_u[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = eye(nl)
        aineq_dspijsubqij_u[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = -eye(nl)
        aineq_dspijsubqij_u[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)
        # *** < pij - qij
        aineq_dspijsubqij_l[ix_(j_interval * nl + arange(nl), var_pij_x[:, j_interval])] = -eye(nl)
        aineq_dspijsubqij_l[ix_(j_interval * nl + arange(nl), var_qij_x[:, j_interval])] = eye(nl)
        aineq_dspijsubqij_l[ix_(j_interval * nl + arange(nl), var_alpha_branch_x)] = -sqrt(2) * diag(slmax)

    model_x_matrix_a = vstack(
        [model_x_matrix_a, aineq_dspijaddqij_u, aineq_dspijaddqij_l, aineq_dspijsubqij_u, aineq_dspijsubqij_l])
    model_x_rhs += bineq_dspijaddqij_u.tolist() + bineq_dspijaddqij_l.tolist() + bineq_dspijsubqij_u.tolist() + bineq_dspijsubqij_l.tolist()
    model_x_senses += ['L'] * 4 * nl * n_interval
    time_elapsed = time.time() - time_start
    # model_x_matrix_a.nbytes()
    # Add constraints
    # model_x_matrix_a = csr_matrix(model_x_matrix_a)
    # a_rows = model_x_matrix_a.nonzero()[0].tolist()
    a_rows = model_x_matrix_a.row.tolist()  # No computation, only query of attributes, faster than nonzero.
    # a_cols = model_x_matrix_a.nonzero()[1].tolist()
    a_cols = model_x_matrix_a.col.tolist()  # tolist() is for 'non-integral value in input sequence'
    # a_data = model_x_matrix_a.data.tolist()  # model_x_matrix_a is csr_matrix matrix, a_data element needs to be float
    # start = time.time()
    # a_vals = model_x_matrix_a[model_x_matrix_a.nonzero()]
    # elasped1 = time.time()-start
    # start = time.time()
    # a_vals = model_x_matrix_a[model_x_matrix_a != 0]  # boolean mask index array, it's faster than nonzero, but result is different?
    # Note that boolean mask returns 1D-array
    # a_vals = model_x_matrix_a[a_rows, a_cols].tolist()  # faster than boolean mask
    a_vals = model_x_matrix_a.data
    # elpased2 = time.time() - start
    model_x.linear_constraints.add(rhs=model_x_rhs, senses=model_x_senses,
                                   names=['constraint{0}'.format(i)
                                          for i in range(len(model_x_rhs))])

    model_x.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))
    # set objective sense
    model_x.objective.set_sense(model_x.objective.sense.minimize)
    # Benders  decomposition
    # Solving with automatic Benders decomposition.
    # By setting the Benders strategy parameter to Full, CPLEX will put all integer variables into the master, all
    # continuous variables into a sub-problem, and further decompose that sub-problem, if possible.
    # model_x.parameters.benders.strategy.set(model_x.parameters.benders.strategy.values.full)

    return model_x

def set_model_grb_1(case_params):
    import gurobipy as grb
    # Extract parameters
    # dynamic allocation with locals() doesn't work in python3
    # names = locals()
    # for i in range(3):
    #     names['a'+str(i)]=i
    #     print('a'+str(i))
    #     exec("")

    # for element in case_params.items():
    #     # Execute nb = nb
    #     # exec(element[0] + '=' + 'element[1]')
    #     # Why locals() doesn't work while globals works?
    #      globals()[element[0]] = element[1]  # it's global variables, any alternatives?

    params_keys = case_params.keys()
    params_values = case_params.values()

    (ppc, nb, nl, ng, n_mg, nd, n_arc, n_ev, n_interval, n_scenario, n_station, cost_ev_transit, cost_ev_power, delta_t,
     sn_mva, MW_KW, scenario_weight, cost_mg, load_inter_cost, index_load, soc_min, soc_max, pg_l, qg_l, localload_p,
     localload_q, energy_u, energy_l, pg_u, qg_u, vm_l, vm_u, slmax, pd_u, qd_u, station_connection_t, station_connection_f,
     ev_position_init, n_arc_transit, roundtrip_f, roundtrip_t, index_arc_parking, ev_ch_u, ev_dch_u, ev_energy_capacity,
     ev_energy_init, ev_energy_l, ev_energy_u, index_arc_transit, ev_ch_efficiency, ev_dch_efficiency, connection_f,
     connection_t, index_pq_bus, index_beta_ij, index_beta_ji, connection_generator, connection_load, large_m, branch_r,
     branch_x, v0, load_qp_ratio, soc_init, load_inter_cost_augment, n_timewindow_c) = (case_params['ppc'], case_params['nb'],
            case_params['nl'], case_params['ng'], case_params['n_mg'], case_params['nd'], case_params['n_arc'],
            case_params['n_ev'], case_params['n_interval'], case_params['n_scenario'], case_params['n_station'],
            case_params['cost_ev_transit'],  case_params['cost_ev_power'], case_params['delta_t'], case_params['sn_mva'],
            case_params['MW_KW'], case_params['scenario_weight'], case_params['cost_mg'], case_params['load_inter_cost'],
            case_params['index_load'], case_params['soc_min'], case_params['soc_max'], case_params['pg_l'], case_params['qg_l'],
            case_params['localload_p'], case_params['localload_q'], case_params['energy_u'], case_params['energy_l'],
            case_params['pg_u'], case_params['qg_u'], case_params['vm_l'], case_params['vm_u'], case_params['slmax'], case_params['pd_u'],
            case_params['qd_u'], case_params['station_connection_t'], case_params['station_connection_f'], case_params['ev_position_init'],
            case_params['n_arc_transit'], case_params['roundtrip_f'], case_params['roundtrip_t'], case_params['index_arc_parking'],
            case_params['ev_ch_u'], case_params['ev_dch_u'], case_params['ev_energy_capacity'], case_params['ev_energy_init'],
            case_params['ev_energy_l'], case_params['ev_energy_u'], case_params['index_arc_transit'], case_params['ev_ch_efficiency'],
            case_params['ev_dch_efficiency'], case_params['connection_f'], case_params['connection_t'], case_params['index_pq_bus'],
            case_params['index_beta_ij'], case_params['index_beta_ji'], case_params['connection_generator'], case_params['connection_load'],
            case_params['large_m'], case_params['branch_r'], case_params['branch_x'], case_params['v0'], case_params['load_qp_ratio'],
            case_params['soc_init'], case_params['load_inter_cost_augment'], case_params['n_timewindow_c'])

    (pij_l, qij_l, pij_u, qij_u, index_genbus) = (case_params['pij_l'], case_params['qij_l'], case_params['pij_u'],
                                                  case_params['qij_u'], case_params['index_genbus'])

    # Build model
    # Initialization of gurobi
    model_x = grb.Model('first stage model')

    ## Define variables
    # prefix var_ is for variables index array
    # TESS model
    # charging power from mg to ev at time interval t, 3D-array, (n_ev, n_mg, n_interval), each element is gurobi variable object
    var_pev2mg_ch_x = array([model_x.addVar(name='ch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_interval))
                             for i_ev in range(n_ev)
                             for j_mg in range(n_mg)
                             for k_interval in range(n_interval)],
                            dtype='object').reshape(n_ev, n_mg, n_interval)
    
    # discharging power from ev to mg at time interval t, 3D-array, (n_ev, n_mg, n_interval)
    var_pev2mg_dch_x = array([model_x.addVar(name='dch_ev{0}_mg{1}_t{2}'.format(i_ev, j_mg, k_interval))
                              for i_ev in range(n_ev)
                              for j_mg in range(n_mg)
                              for k_interval in range(n_interval)],
                            dtype='object').reshape(n_ev, n_mg, n_interval)

    # ev's soc at time interval t, 2-D array, (n_ev, n_interval)
    # lb should be 1-d array-like input with the same length as variables
    # .flatten() returns copy while ravel() generally returns view.
    var_ev_soc_x = array([model_x.addVar(lb=soc_min[i_ev, j_interval], ub=soc_max[i_ev, j_interval],
                                         name='soc_ev{0}_t{1}'.format(i_ev, j_interval))
                          for i_ev in range(n_ev)
                          for j_interval in range(n_interval)],
                         dtype='object').reshape(n_ev, n_interval)

    # charging sign for ev at time interval t, 2-D array, (n_ev, n_interval)
    var_sign_ch_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                          name='sign_ch_ev{0}_t{1}'.format(i_ev, j_interval))
                           for i_ev in range(n_ev)
                           for j_interval in range(n_interval)],
                           dtype='object').reshape(n_ev, n_interval)

    # discharging sign for ev at time interval t, 2-D array, (n_ev, n_interval)
    var_sign_dch_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                           name='sign_dch_ev{0}_t{1}'.format(i_ev, j_interval))
                            for i_ev in range(n_ev)
                            for j_interval in range(n_interval)],
                           dtype='object').reshape(n_ev, n_interval)

    # arc status for ev at time interval t, 3-D array, (n_ev, n_arc, n_interval)
    var_ev_arc_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                         name='ev{0}_arc{1}_t{2}'.format(i_ev, j_arc, k_interval))
                          for i_ev in range(n_ev)
                          for j_arc in range(n_arc)
                          for k_interval in range(n_interval)],
                         dtype='object').reshape(n_ev, n_arc, n_interval)

    # Transit status for ev at time interval t, 2-D array, (n_ev, n_interval)
    var_sign_onroad_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                              name='sign_onroad_ev{0}_t{1}'.format(i_ev, j_interval))
                               for i_ev in range(n_ev)
                               for j_interval in range(n_interval)],
                              dtype='object').reshape(n_ev, n_interval)

    ## MG model
    # active power generation from MG, 2-D array, (n_mg, n_interval)
    var_pmg_x = array([model_x.addVar(lb=pg_l[i_mg, j_interval], ub=(pg_u-localload_p)[i_mg, j_interval],
                                      name='p_mg{0}_t{1}'.format(i_mg, j_interval))
                       for i_mg in range(n_mg)
                       for j_interval in range(n_interval)],
                      dtype='object').reshape(n_mg, n_interval)

    # reactive power generation from MG, 2-D array, (n_mg, n_interval)
    var_qmg_x = array([model_x.addVar(lb=(qg_l-localload_q)[i_mg, j_interval], ub=(qg_u-localload_q)[i_mg, j_interval],
                                      name='q_mg{0}_t{1}'.format(i_mg, j_interval))
                       for i_mg in range(n_mg)
                       for j_interval in range(n_interval)],
                      dtype='object').reshape(n_mg, n_interval)

    # the amount of energy of MG, 2-D array, (n_mg, n_interval)
    var_emg_x = array([model_x.addVar(lb=energy_l[i_mg, j_interval], ub=energy_u[i_mg,j_interval],
                                      name='e_mg{0}_t{1}'.format(i_mg, j_interval))
                       for i_mg in range(n_mg)
                       for j_interval in range(n_interval)],
                      dtype='object').reshape(n_mg, n_interval)

    # model DS
    # Line active power, 2-D array, (nl, n_interval)
    var_pij_x = array([model_x.addVar(lb=pij_l[i_l, j_interval], ub=pij_u[i_l, j_interval],
                                      name='pij_l{0}_t{1}'.format(i_l, j_interval))
                       for i_l in range(nl)
                       for j_interval in range(n_interval)],
                      dtype='object').reshape(nl, n_interval)

    # Line reactive power, 2-D array, (nl, n_interval)
    var_qij_x = array([model_x.addVar(lb=qij_l[i_l, j_interval], ub=qij_u[i_l, j_interval],
                                      name='qij_l{0}_t{1}'.format(i_l, j_interval))
                      for i_l in range(nl)
                      for j_interval in range(n_interval)],
                      dtype='object').reshape(nl, n_interval)

    # bus voltage, 2-D array, (nb, n_interval)
    var_vm_x = array([model_x.addVar(lb=vm_l[i_b, j_interval], ub=vm_u[i_b, j_interval],
                                     name='vm_l{0}_t{1}'.format(i_b, j_interval))
                     for i_b in range(nb)
                     for j_interval in range(n_interval)],
                     dtype='object').reshape(nb, n_interval)

    # aggregated active power generation at DS level, 2-D array, (ng, n_interval)
    var_pg_x = array([model_x.addVar(name='pg{0}_t{1}'.format(i_g, j_interval))
                      for i_g in range(ng)
                      for j_interval in range(n_interval)],
                     dtype='object').reshape(ng, n_interval)

    # aggregated reactive power generation at DS level, 2-D array, (ng, n_interval)
    # lb = full((ng, n_interval), fill_value=-inf).ravel()
    var_qg_x = array([model_x.addVar(lb=-grb.GRB.INFINITY,
                                     name='qg{0}_t{1}'.format(i_g, j_interval))
                      for i_g in range(ng)
                      for j_interval in range(n_interval)],
                     dtype='object').reshape(ng, n_interval)

    # sign for load restoration, 2-D array, (nd, n_interval)
    var_gama_load_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                            name='gama_load{0}_t{1}'.format(i_d, j_interval))
                             for i_d in range(nd)
                             for j_interval in range(n_interval)],
                            dtype='object').reshape(nd, n_interval)

    # Line connection status, 1-D array, (nl)
    var_alpha_branch_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                               name='alpha_branch{0}'.format(i_l))
                                for i_l in range(nl)],
                               dtype='object').reshape(nl)

    # Auxiliary variables for line status, 1-D array, (nl)
    var_betaij_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                         name='betaij_{0}'.format(i_l))
                         for i_l in range(nl)],
                         dtype='object').reshape(nl)

    var_betaji_x = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                         name='betaji_{0}'.format(i_l))
                          for i_l in range(nl)],
                         dtype='object').reshape(nl)

    # variables for scenario, 1-D array, (n_scenario)
    var_scenario = array([model_x.addVar(vtype=grb.GRB.BINARY,
                                         name='scenario_{0}'.format(i_scenario))
                          for i_scenario in range(n_scenario)],
                         dtype='object').reshape(n_scenario)
    # It is compulsory for this update()
    model_x.update()

    # Add constraints

    # constraints sigma(**) == 1
    lhs = sum(var_ev_arc_x, axis=1)
    rhs = 1

    model_x.addConstrs(lhs[i_ev, j_interval] == rhs
                       for i_ev in range(n_ev)
                       for j_interval in range(n_interval))

    # constraints sigma(z-, t) == sigma(z+, t+1)

    # Objective function
    obj = grb.LinExpr()
    # Note that while quicksum is much faster than sum, it isn’t the fastest approach for building a large expression.Use addTerms or the LinExpr() constructor if you want the quickest possible expression construction.
    for i in range(ng):
        obj += 1

    model_x.setObjective(obj)

    model_x.optimize()

    #
    objval = model_x.objVal

    # solutions
    ev_arc_x = array([var_ev_arc_x[i_ev, j_arc, k_interval].x
           for i_ev in range(n_ev)
           for j_arc in range(n_arc)
           for k_interval in range(n_interval)]).reshape(n_ev, n_arc, n_interval)

    a = 1