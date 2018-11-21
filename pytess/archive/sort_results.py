from numpy import array, sum

def sort_results(case_params, model_x):
    """
    Sort optimization solutions and return dictionary res that store results.

    :param case_params: dictionary, case parameters
    :param model_x: Cplex instance, optimization model with solutions
    :return: res: dictionary, store results
    """
    # 2018.08.02
    # extract case parameters
    params_keys = case_params.keys()
    params_values = case_params.values()

    (var_pev2mg_ch_x, var_pev2mg_dch_x, var_ev_soc_x, var_sign_ch_x, var_sign_dch_x, var_ev_arc_x, var_sign_onroad_x,
     var_pmg_x, var_qmg_x, var_emg_x, var_pij_x, var_qij_x, var_vm_x, var_pg_x, var_qg_x, var_gama_load_x,
     var_alpha_branch_x, var_betaij_x, var_betaji_x, var_scenario) = (case_params['var_pev2mg_ch_x'],
                case_params['var_pev2mg_dch_x'], case_params['var_ev_soc_x'], case_params['var_sign_ch_x'],
                case_params['var_sign_dch_x'], case_params['var_ev_arc_x'], case_params['var_sign_onroad_x'],
                case_params['var_pmg_x'], case_params['var_qmg_x'], case_params['var_emg_x'], case_params['var_pij_x'],
                case_params['var_qij_x'], case_params['var_vm_x'], case_params['var_pg_x'], case_params['var_qg_x'],
                case_params['var_gama_load_x'], case_params['var_alpha_branch_x'], case_params['var_betaij_x'],
                case_params['var_betaji_x'], case_params['var_scenario'])

    (pd_u, qd_u) = (case_params['pd_u'], case_params['qd_u'])

    # acquire and sort results
    res = {}

    res['status'] = model_x.solution.status[model_x.solution.get_status()]
    res['goal'] = model_x.solution.get_objective_value()

    # coloumn vector for solutions
    values = array(model_x.solution.get_values())
    # TESS model
    # charging power from mg to ev at time span t, 3D-array, (n_ev, n_mg, n_interval)
    res['pev2mg_ch_x'] = values[var_pev2mg_ch_x]
    res['pev_ch_x'] = res['pev2mg_ch_x'].sum(axis=1)  # (n_ev, n_interval)
    res['mg_se_ev_x'] = res['pev2mg_ch_x'].sum(axis=0) # (n_mg, n_interval), indicating each mg's sending power at time span t.
    # discharging power from ev to mg at time span t, 3D-array, (n_ev, n_mg, n_interval)
    res['pev2mg_dch_x'] = values[var_pev2mg_dch_x]
    res['pev_dch_x'] = res['pev2mg_dch_x'].sum(axis=1)  # (n_ev, n_interval)
    res['mg_re_ev_x'] = res['pev2mg_dch_x'].sum(axis=0)  # (n_mg, n_interval), indicating each mg's receiving power at time span t.
    res['pev_x'] = res['pev_dch_x'] - res['pev_ch_x']  # (n_ev, n_interval), ev's net output power to mg
    res['mg_p_ev_x'] = res['mg_re_ev_x'] - res['mg_se_ev_x']  # (n_mg, n_interval), indicating each mg's net receiving power associated with ev at time span t.
    # ev's soc at time span t, 2-D array, (n_ev, n_interval)
    res['ev_soc_x'] = values[var_ev_soc_x]
    # charging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    res['sign_ch_x'] = values[var_sign_ch_x]
    # discharging sign for ev at time span t, 2-D array, (n_ev, n_interval)
    res['sign_dch_x'] = values[var_sign_dch_x]
    # arc status for ev at time span t, 3-D array, (n_ev, n_arc, n_interval)
    res['ev_arc_x'] = values[var_ev_arc_x]
    # Transit status for ev at time span t, 2-D array, (n_ev, n_interval)
    res['sign_onroad_x'] = values[var_sign_onroad_x]

    # MG model
    # active power generation from MG, 2-D array, (n_mg, n_interval)
    res['pmg_x'] = values[var_pmg_x]
    # reactive power generation from MG, 2-D array, (n_mg, n_interval)
    res['qmg_x'] = values[var_qmg_x]
    # the amount of energy of MG, 2-D array, (n_mg, n_interval)
    res['emg_x'] = values[var_emg_x]

    # DS model
    # Line active power, 2-D array, (nl, n_interval)
    res['pij_x'] = values[var_pij_x]
    # Line reactive power, 2-D array, (nl, n_interval)
    res['qij_x'] = values[var_qij_x]
    # bus voltage, 2-D array, (nb, n_interval)
    res['vm_x'] = values[var_vm_x]
    # aggregated active power generation at DS level, 2-D array, (ng, n_interval)
    res['pg_x'] = values[var_pg_x]
    # aggregated reactive power generation at DS level, 2-D array, (ng, n_interval)
    res['qg_x'] = values[var_qg_x]
    # sign for load restoration, 2-D array, (nd, n_interval)
    res['gama_load_x'] = values[var_gama_load_x]
    # load restoration
    res['pd_x'] = res['gama_load_x'] * pd_u  # (nd, n_interval)
    res['qd_x'] = res['gama_load_x'] * qd_u  # (nd, n_interval)
    res['pdcut_x'] = pd_u - res['pd_x']
    res['qdcut_x'] = qd_u - res['qd_x']
    # Line connection status, 1-D array, (nl)
    res['alpha_branch_x'] = values[var_alpha_branch_x]
    # Auxiliary variables for line status, 1-D array, (nl)
    res['betaij_x'] = values[var_betaij_x]
    res['betaji_x'] = values[var_betaji_x]
    # variables for scenario, 1-D array, (n_scenario)
    res['var_scenario'] = values[var_scenario]

    return res

# def sort_results_rolling(case_params, ls_model_x):