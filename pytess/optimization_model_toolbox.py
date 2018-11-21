"""
-------------------------------------------------
   File Name：     model_box
   Description :
   Author :       yaoshuhan
   date：          19/11/18
-------------------------------------------------
   Change Activity:
                   19/11/18:
-------------------------------------------------
"""


import numpy as np
from numpy import array
import cplex as cpx

def add_variables_cpx(self, dsnet, ssnet, tessnet, tsnnet):
    ## Define varialbles and get position array
    # prefix var_ is for variables index array
    # TESS model
    # charging power from station to tess at interval t, 3D-array,
    # (n_tess, n_station, n_interval)
    n_tess = tessnet.n_tess
    n_station = ssnet.n_station
    n_interval = dsnet.n_interval

    self.var_tess2st_pch_x = array(self.variables.add(
        names=['ch_tess{0}_st{1}_t{2}'.format(i_tess, j_station, k_interval)
               for i_tess in range(n_tess)
               for j_station in range(n_station)
               for k_interval in range(n_interval)])
    ).reshape(n_tess, n_station, n_interval)

    # discharging power from tess to station at interval t, 3D-array,
    # (n_tess, n_station, n_interval)
    self.var_tess2st_pdch_x = array(self.variables.add(
        names=['dch_tess{0}_st{1}_t{2}'.format(i_tess, j_station, k_interval)
               for i_tess in range(n_tess)
               for j_station in range(n_station)
               for k_interval in range(n_interval)])
    ).reshape(n_tess, n_station, n_interval)

    # tess's energy at interval t, (n_tess, n_interval)
    # lb should be 1-d array-like input with the same length as variables
    # .flatten() returns copy while ravel() generally returns view.
    self.var_tess_e_x = array(self.variables.add(
        lb=tessnet.tess_e_l.ravel(), ub=tessnet.tess_e_u.ravel(),
        names=['e_tess{0}_t{1}'.format(i_tess, j_interval)
               for i_tess in range(n_tess)
               for j_interval in range(n_interval)])
    ).reshape(n_tess, n_interval)

    # charging sign for tess at interval t, (n_tess, n_interval)
    self.var_sign_ch_x = array(self.variables.add(
        types=['B'] * (n_tess * n_interval),
        names=['sign_ch_tess{0}_t{1}'.format(i_tess, j_interval)
               for i_tess in range(n_tess)
               for j_interval in range(n_interval)])
    ).reshape(n_tess, n_interval)

    # discharging sign for ev at interval t, (n_tess, n_interval)
    self.var_sign_dch_x = array(self.variables.add(
        types=['B'] * (n_tess * n_interval),
        names=['sign_dch_tess{0}_t{1}'.format(i_tess, j_interval)
               for i_tess in range(n_tess)
               for j_interval in range(n_interval)])
    ).reshape(n_tess, n_interval)

    # arc status for tess at interval t, (n_tess, n_tsn_arc)
    # modify it to fit into new tsn model
    n_tsn_arc = tsnnet.n_tsn_arc

    self.var_tess_arc_x = array(self.variables.add(
        types=['B'] * n_tess * n_tsn_arc,
        names=['tess{0}_arc{1}'.format(i_tess, j_arc)
               for i_tess in range(n_tess)
               for j_arc in range(n_tsn_arc)])
    ).reshape(n_tess, n_tsn_arc)

    # Transit status for tess at time span t, (n_tess, n_interval)
    self.var_sign_onroad_x = array(self.variables.add(
        types=['B'] * (n_tess * n_interval),
        names=['sign_onroad_tess{0}_t{1}'.format(i_tess, j_interval)
               for i_tess in range(n_tess)
               for j_interval in range(n_interval)])
    ).reshape(n_tess, n_interval)

    ## MG model
    # active power output of station, (n_station, n_interval)
    self.var_station_p_x = array(self.variables.add(
        lb=ssnet.station_p_l.ravel(),
        ub=(ssnet.station_p_u - ssnet.p_localload).ravel(),
        names=['p_station{0}_t{1}'.format(i_station, j_interval)
               for i_station in range(n_station)
               for j_interval in range(n_interval)])
    ).reshape(n_station, n_interval)

    # reactive power output of station, (n_station, n_interval)
    self.var_station_q_x = array(self.variables.add(
        lb=(ssnet.station_q_l - ssnet.q_localload).ravel(),
        ub=(ssnet.station_q_u - ssnet.q_localload).ravel(),
        names=['q_station{0}_t{1}'.format(i_station, j_interval)
               for i_station in range(n_station)
               for j_interval in range(n_interval)])
    ).reshape(n_station, n_interval)

    # the amount of energy of station, (n_station, n_interval)
    self.var_station_e_x = array(self.variables.add(
        lb=ssnet.station_e_l.ravel(), ub=ssnet.station_e_u.ravel(),
        names=['e_station{0}_t{1}'.format(i_station, j_interval)
               for i_station in range(n_station)
               for j_interval in range(n_interval)])
    ).reshape(n_station, n_interval)

    # model distribution system
    n_line = dsnet.n_line
    n_bus = dsnet.n_bus
    # Line active power, (n_line, n_interval)
    self.var_pij_x = array(self.variables.add(
        lb=dsnet.pij_l.ravel(), ub=dsnet.pij_u.ravel(),
        names=['pij_l{0}_t{1}'.format(i_line, j_interval)
               for i_line in range(n_line)
               for j_interval in range(n_interval)])
    ).reshape(n_line, n_interval)

    # Line reactive power,  (n_line, n_interval)
    self.var_qij_x = array(self.variables.add(
        lb=dsnet.qij_l.ravel(), ub=dsnet.qij_u.ravel(),
        names=['qij_l{0}_t{1}'.format(i_line, j_interval)
               for i_line in range(n_line)
               for j_interval in range(n_interval)])
    ).reshape(n_line, n_interval)

    # bus voltage, (n_bus, n_interval)
    self.var_vm_x = array(self.variables.add(
        lb=dsnet.vm_l.ravel(), ub=dsnet.vm_u.ravel(),
        names=['vm_l{0}_t{1}'.format(i_bus, j_interval)
               for i_bus in range(n_bus)
               for j_interval in range(n_interval)])
    ).reshape(n_bus, n_interval)

    # aggregated active power generation at DS level,(n_station, n_interval)
    self.var_aggregate_pg_x = array(self.variables.add(
        lb=-cpx.infinity * np.ones((n_station, n_interval)).ravel(),
        ub=cpx.infinity * np.ones((n_station, n_interval)).ravel(),
        names=['aggregate_pg{0}_t{1}'.format(i_station, j_interval)
               for i_station in range(n_station)
               for j_interval in range(n_interval)])
    ).reshape(n_station, n_interval)
    # aggregated reactive power generation at DS level, (n_station, n_interval)
    # lb = full((n_station, n_interval), fill_value=-inf).ravel()
    self.var_aggregate_qg_x = array(self.variables.add(
        lb=-cpx.infinity * np.ones((n_station, n_interval)).ravel(),
        ub=cpx.infinity * np.ones((n_station, n_interval)).ravel(),
        names=['aggregate_qg{0}_t{1}'.format(i_station, j_interval)
               for i_station in range(n_station)
               for j_interval in range(n_interval)])
    ).reshape(n_station, n_interval)

    # sign for load restoration, (n_load, n_interval)
    # self.variables.type.binary is equivalent to 'B'
    n_load = dsnet.n_load
    self.var_gama_load_x = array(self.variables.add(
        types=['B'] * (n_load * n_interval),
        names=['gama_load{0}_t{1}'.format(i_d, j_interval)
               for i_d in range(n_load)
               for j_interval in range(n_interval)])
    ).reshape(n_load, n_interval)

    # Line connection status, 1-D array, (n_line)
    self.var_alpha_branch_x = array(self.variables.add(
        types=['B'] * n_line,
        names=['alpha_branch{0}'.format(i_line)
               for i_line in range(n_line)])
    ).reshape(n_line)

    # Auxiliary variables for line status, 1-D array, (n_line)
    self.var_betaij_x = array(self.variables.add(
        types=['B'] * n_line,
        names=['betaij_{0}'.format(i_line)
               for i_line in range(n_line)])
    ).reshape(n_line)

    self.var_betaji_x = array(self.variables.add(
        types=['B'] * n_line,
        names=['betaji_{0}'.format(i_line)
               for i_line in range(n_line)])
    ).reshape(n_line)

    # variables for scenario, 1-D array, (n_scenario)
    # var_scenario = array(model_x.variables.add(
    #     types=['B'] * n_scenario,
    #     names=['scenario_{0}'.format(i_scenario)
    #            for i_scenario in range(n_scenario)])
    # ).reshape(n_scenario)

    #  The total number of variables
    # self.variables.get_num()

def add_constraints_cpx(self, dsnet, ssnet, tessnet, tsnnet):
    # Form matrix A, vector b and sense
    # It turns to be more efficient to set up A incrementally as coo_matrix
    # while b as list
    # aeq is ili_matrix while beq is nd-array
    # time_start = time.time()  # zeros- 7s, roughly, lil_matrix- 0.7s, roughly
    import scipy.sparse as spar
    from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
    # Extract parameters
    n_all_vars_x = self.variables.get_num()
    n_interval = dsnet.n_interval
    n_tess = tessnet.n_tess
    # Initialization of matrix A
    model_x_matrix_a = coo_matrix((0, n_all_vars_x))
    model_x_matrix_a = coo_matrix((0, n_all_vars_x))
    model_x_rhs = []
    model_x_senses = []

    # --------------------------------------------------------------------------
    # Each tess only in one status in each interval
    aeq_onestatus = lil_matrix((n_tess * n_interval, n_all_vars_x))
    beq_onestatus = np.ones((n_tess * n_interval))

    # retrieve parameters
    tsn_cut_set = tsnnet.tsn_cut_set

    for i_tess in range(n_tess):
        for j_interval in range(n_interval):
            aeq_onestatus[i_tess * n_interval + j_interval,
                          self.var_tess_arc_x[
                              i_tess, tsn_cut_set[:, j_interval]]] = 1

    # model_x_matrix_a = concatenate((model_x_matrix_a, aeq_onestatus), axis=0)
    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_onestatus])
    # # model_x_rhs = hstack([model_x_rhs, beq_onestatus])
    model_x_rhs += beq_onestatus.tolist()
    model_x_senses += ['E'] * n_tess * n_interval

    # --------------------------------------------------------------------------
    # Constraints for tess transit flow
    tsn_node = tsnnet.tsn_node
    # the actual no. of tsn_node
    n_tsn_node = tsnnet.n_tsn_node

    aeq_transitflow = lil_matrix((n_tess * n_tsn_node, n_all_vars_x))
    beq_transitflow = np.zeros(n_tess * n_tsn_node)

    for i_tess in range(n_tess):
        # For the tsn source nodes that reside in the time point 0
        tsn_node_source = tsn_node[ssnet.idx_depot, 0]
        mapping_tsn_f2arc = tsnnet.mapping_tsn_f2arc
        mapping_tess_init2tsnode = tessnet.mapping_tess_init2tsnode

        aeq_transitflow[np.ix_(i_tess * n_tsn_node + tsn_node_source,
                               self.var_tess_arc_x[i_tess,
                               :])] = mapping_tsn_f2arc[
                                      tsn_node_source, :]
        # Initial location
        beq_transitflow[i_tess * n_tsn_node +
                        tsn_node_source] = mapping_tess_init2tsnode[
            i_tess, ssnet.idx_depot].toarray()
        # For the tsn nodes that reside from the 2nd time point to the
        # last-to-second time point
        mapping_tsn_t2arc = tsnnet.mapping_tsn_t2arc
        tsn_node_microgrid = tsn_node[
                             ssnet.idx_microgrid, 1:-1].ravel(order='F')

        aeq_transitflow[np.ix_(i_tess * n_tsn_node + tsn_node_microgrid,
                               self.var_tess_arc_x[i_tess,
                               :])] = mapping_tsn_t2arc[
                                      tsn_node_microgrid,
                                      :] - mapping_tsn_f2arc[
                                           tsn_node_microgrid, :]
        # !!!! error has been corrected
        # aeq_transitflow[ix_(i_tess*n_tsn_node+j_tsn_node_range,
        # var_ev_arc_x[i_tess, :])] =

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_transitflow])
    model_x_rhs += beq_transitflow.tolist()
    model_x_senses += ['E'] * n_tess * n_tsn_node

    # --------------------------------------------------------------------------
    # charging/discharging with respect to position
    n_station = ssnet.n_station
    tsn_arc_holding = tsnnet.tsn_arc_holding

    aineq_pchposition = lil_matrix(
        (n_tess * n_interval * n_station, n_all_vars_x))
    bineq_pchposition = np.zeros(n_tess * n_interval * n_station)
    aineq_pdchposition = lil_matrix(
        (n_tess * n_interval * n_station, n_all_vars_x))
    bineq_pdchposition = np.zeros(n_tess * n_interval * n_station)

    for i_tess in range(n_tess):
        for j_interval in range(n_interval):
            # pch <= ***Pch,max
            aineq_pchposition[np.ix_(i_tess * n_interval * n_station
                + j_interval * n_station + np.arange(n_station),
                self.var_tess2st_pch_x[i_tess, :, j_interval])] = np.eye(
                                                                    n_station)
            # todo !!! need to correct
            # Check if there are holding arcs for the interval
            # If so, enter it, otherwise skip it
            if tsn_arc_holding[:, j_interval].any():
                aineq_pchposition[np.ix_(i_tess * n_interval * n_station
                    + j_interval * n_station + ssnet.idx_microgrid,
                    self.var_tess_arc_x[i_tess, tsn_arc_holding[:,
                        j_interval]])] = -tessnet.tess_pch_u[i_tess] * np.eye(
                                                            ssnet.n_microgrid)

            # Pdch <= ***Pdch,max
            aineq_pdchposition[np.ix_(i_tess * n_interval * n_station
                + j_interval * n_station + np.arange(n_station),
                self.var_tess2st_pdch_x[i_tess, :, j_interval])] = np.eye(
                                                                    n_station)
            if tsn_arc_holding[:, j_interval].any():
                aineq_pdchposition[np.ix_(i_tess * n_interval * n_station
                    + j_interval * n_station + ssnet.idx_microgrid,
                    self.var_tess_arc_x[i_tess, tsn_arc_holding[:,
                        j_interval]])] = -tessnet.tess_pdch_u[i_tess] * np.eye(
                                                            ssnet.n_microgrid)

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_pchposition,
                                    aineq_pdchposition])
    model_x_rhs += (bineq_pchposition.tolist() + bineq_pdchposition.tolist())
    model_x_senses += ['L'] * 2 * n_tess * n_interval * n_station

    # --------------------------------------------------------------------------
    # charging/discharging with respect to battery status
    aineq_pchstatus = lil_matrix((n_tess * n_interval, n_all_vars_x))
    bineq_pchstatus = np.zeros(n_tess * n_interval)

    aineq_pdchstatus = lil_matrix((n_tess * n_interval, n_all_vars_x))
    bineq_pdchstatus = np.zeros(n_tess * n_interval)

    for i_tess in range(n_tess):
        for j_interval in range(n_interval):
            aineq_pchstatus[i_tess * n_interval + j_interval,
                            self.var_tess2st_pch_x[i_tess, :, j_interval]] = 1
            aineq_pchstatus[i_tess * n_interval + j_interval,
                            self.var_sign_ch_x[i_tess, j_interval]] \
                = -tessnet.tess_pch_u[i_tess]

            aineq_pdchstatus[i_tess * n_interval + j_interval,
                             self.var_tess2st_pdch_x[i_tess, :, j_interval]] = 1
            aineq_pdchstatus[i_tess * n_interval + j_interval,
                             self.var_sign_dch_x[i_tess, j_interval]] \
                = -tessnet.tess_pdch_u[i_tess]

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_pchstatus,
                                    aineq_pdchstatus])
    model_x_rhs += (bineq_pchstatus.tolist() + bineq_pdchstatus.tolist())
    model_x_senses += ['L'] * 2 * n_tess * n_interval

    # --------------------------------------------------------------------------
    # charging/discharging status
    aineq_chdchstatus = lil_matrix((n_tess * n_interval, n_all_vars_x))
    bineq_chdchstatus = np.zeros(n_tess * n_interval)

    for i_tess in range(n_tess):
        for j_interval in range(n_interval):
            aineq_chdchstatus[i_tess * n_interval + j_interval,
                              self.var_sign_ch_x[i_tess, j_interval]] = 1
            aineq_chdchstatus[i_tess * n_interval + j_interval,
                              self.var_sign_dch_x[i_tess, j_interval]] = 1
            # check if there are holding arcs in the interval
            if tsn_arc_holding[:, j_interval].any():
                aineq_chdchstatus[i_tess * n_interval + j_interval,
                              self.var_tess_arc_x[i_tess, tsn_arc_holding[
                                  :, j_interval]]] = -1

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aineq_chdchstatus])
    model_x_rhs += bineq_chdchstatus.tolist()
    model_x_senses += ['L'] * n_tess * n_interval

    # --------------------------------------------------------------------------
    # constraint for sign_onroad
    aeq_signonroad = lil_matrix((n_tess * n_interval, n_all_vars_x))
    beq_signonroad = np.ones(n_tess * n_interval)

    for i_tess in range(n_tess):
        for j_interval in range(n_interval):
            aeq_signonroad[i_tess * n_interval + j_interval,
                           self.var_sign_onroad_x[i_tess, j_interval]] = 1
            # check if there are holding arcs in the interval
            if tsn_arc_holding[:, j_interval].any():
                aeq_signonroad[i_tess * n_interval + j_interval,
                           self.var_tess_arc_x[i_tess, tsn_arc_holding[
                               :, j_interval]]] = 1

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_signonroad])
    model_x_rhs += beq_signonroad.tolist()
    model_x_senses += ['E'] * n_tess * n_interval

    # --------------------------------------------------------------------------
    # constraints for energy of tess
    delta_t = dsnet.delta_t
    aeq_energy = lil_matrix((n_tess * n_interval, n_all_vars_x))
    beq_energy = np.zeros(n_tess * n_interval)

    for i_tess in range(n_tess):
        for j_interval in range(n_interval - 1):
            aeq_energy[i_tess * n_interval + j_interval,
                       self.var_tess_e_x[i_tess, j_interval + 1]] = 1
            aeq_energy[i_tess * n_interval + j_interval,
                       self.var_tess_e_x[i_tess, j_interval]] = -1

            aeq_energy[i_tess * n_interval + j_interval,
                       self.var_tess2st_pch_x[i_tess, :, j_interval + 1]] \
                = -delta_t * tessnet.tess_ch_efficiency[i_tess]
            aeq_energy[i_tess * n_interval + j_interval,
                       self.var_tess2st_pdch_x[i_tess, :, j_interval + 1]] \
                = delta_t / tessnet.tess_dch_efficiency[i_tess]

        # Considering t = 0 and intial status, it is stored at the end
        # !!!
        aeq_energy[i_tess * n_interval + n_interval - 1,
                   self.var_tess_e_x[i_tess, 0]] = 1
        aeq_energy[i_tess * n_interval + n_interval - 1,
                   self.var_tess2st_pch_x[i_tess, :, 0]] \
            = -delta_t * tessnet.tess_ch_efficiency[i_tess]
        aeq_energy[i_tess * n_interval + n_interval - 1,
                   self.var_tess2st_pdch_x[i_tess, :, 0]] \
            = delta_t / tessnet.tess_dch_efficiency[i_tess]
        beq_energy[i_tess * n_interval + n_interval - 1] \
            = tessnet.tess_e_init[i_tess]

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_energy])
    model_x_rhs += beq_energy.tolist()
    model_x_senses += ['E'] * n_tess * n_interval

    # --------------------------------------------------------------------------
    # constraints for generation of station
    aeq_generationp = lil_matrix((n_station * n_interval, n_all_vars_x))
    beq_generationp = np.zeros(n_station * n_interval)

    aeq_generationq = lil_matrix((n_station * n_interval, n_all_vars_x))
    beq_generationq = np.zeros(n_station * n_interval)

    for i_station in range(n_station):
        for j_interval in range(n_interval):
            # pch
            aeq_generationp[i_station * n_interval + j_interval,
                        self.var_tess2st_pch_x[:, i_station, j_interval]] = 1
            # pdch
            aeq_generationp[i_station * n_interval + j_interval,
                        self.var_tess2st_pdch_x[:, i_station, j_interval]] = -1
            # station_p
            aeq_generationp[i_station * n_interval + j_interval,
                            self.var_station_p_x[i_station, j_interval]] = -1
            # aggregated_pg
            aeq_generationp[i_station * n_interval + j_interval,
                            self.var_aggregate_pg_x[i_station, j_interval]] = 1

            # station_q
            aeq_generationq[i_station * n_interval + j_interval,
                            self.var_station_q_x[i_station, j_interval]] = -1
            # aggregated_q
            aeq_generationq[i_station * n_interval + j_interval,
                            self.var_aggregate_qg_x[i_station, j_interval]] = 1

    model_x_matrix_a = spar.vstack(
        [model_x_matrix_a, aeq_generationp, aeq_generationq])
    model_x_rhs += beq_generationp.tolist() + beq_generationq.tolist()
    model_x_senses += ['E'] * 2 * n_station * n_interval

    # --------------------------------------------------------------------------
    # The amount of energy for each station in the time point t (end point of
    # time interval t)
    aeq_mgenergy = lil_matrix((n_station * n_interval, n_all_vars_x))
    beq_mgenergy = np.zeros(n_station * n_interval)

    for i_station in range(n_station):
        for j_interval in range(n_interval - 1):
            # Estation_t+1
            aeq_mgenergy[i_station * n_interval + j_interval,
                         self.var_station_e_x[i_station, j_interval + 1]] = 1
            # Estation_t
            aeq_mgenergy[i_station * n_interval + j_interval,
                         self.var_station_e_x[i_station, j_interval]] = -1
            # station_p
            aeq_mgenergy[i_station * n_interval + j_interval,
                         self.var_station_p_x[
                             i_station, j_interval + 1]] = delta_t

        aeq_mgenergy[(i_station + 1) * n_interval - 1,
                     self.var_station_e_x[i_station, 0]] = 1
        aeq_mgenergy[(i_station + 1) * n_interval - 1,
                     self.var_station_p_x[i_station, 0]] = delta_t
        beq_mgenergy[(i_station + 1) * n_interval - 1] \
            = ssnet.station_e_u[i_station, 0]

    model_x_matrix_a = spar.vstack(
        [model_x_matrix_a, aeq_mgenergy])  # error because of indentation
    model_x_rhs += beq_mgenergy.tolist()
    model_x_senses += ['E'] * n_station * n_interval

    # --------------------------------------------------------------------------
    # topology constraint, |N|-|M|
    n_bus = dsnet.n_bus
    n_microgrid = ssnet.n_microgrid
    aeq_dstree = lil_matrix((1, n_all_vars_x))
    beq_dstree = np.zeros(1)

    aeq_dstree[:, self.var_alpha_branch_x] = 1
    beq_dstree[:] = n_bus - n_microgrid  # todo ?? if bus deleted?

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dstree])
    model_x_rhs += beq_dstree.tolist()
    model_x_senses += ['E']

    # --------------------------------------------------------------------------
    # topology constraint, bij + bji = alphaij
    n_line = dsnet.n_line
    aeq_dsbranchstatus = lil_matrix((n_line, n_all_vars_x))
    beq_dsbranchstatus = np.zeros(n_line)

    aeq_dsbranchstatus[:, self.var_alpha_branch_x] = -np.eye(n_line)
    aeq_dsbranchstatus[:, self.var_betaij_x] = np.eye(n_line)
    aeq_dsbranchstatus[:, self.var_betaji_x] = np.eye(n_line)

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsbranchstatus])
    model_x_rhs += beq_dsbranchstatus.tolist()
    model_x_senses += ['E'] * n_line

    # --------------------------------------------------------------------------
    # topology constraint, exact one parent for each bus other than mg bus
    aeq_dsoneparent = lil_matrix((n_bus - n_microgrid, n_all_vars_x))
    beq_dsoneparent = np.ones(n_bus - n_microgrid)

    aeq_dsoneparent[:, self.var_betaij_x] \
        = dsnet.incidence_ds_fbus2line[dsnet.idx_pq_bus, :]
    aeq_dsoneparent[:, self.var_betaji_x] \
        = dsnet.incidence_ds_tbus2line[dsnet.idx_pq_bus, :]

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsoneparent])
    model_x_rhs += beq_dsoneparent.tolist()
    model_x_senses += ['E'] * (n_bus - n_microgrid)

    # --------------------------------------------------------------------------
    # topology constraint, mg buses has no parent
    aeq_dsnoparent = lil_matrix((n_microgrid, n_all_vars_x))
    beq_dsnoparent = np.zeros(n_microgrid)

    aeq_dsnoparent[:, self.var_betaij_x] \
        = dsnet.incidence_ds_fbus2line[dsnet.idx_ref_bus, :]
    aeq_dsnoparent[:, self.var_betaji_x] \
        = dsnet.incidence_ds_tbus2line[dsnet.idx_ref_bus, :]

    # n_index_betaij = dsnet.idx_beta_ij.shape[0]
    # n_index_betaji = dsnet.idx_beta_ji.shape[0]
    # aeq_dsnoparent = lil_matrix(
    #     (n_index_betaij + n_index_betaji, n_all_vars_x))
    # beq_dsnoparent = np.zeros(n_index_betaij + n_index_betaji)
    #
    # # index_beta_ij is different of array and csr_matrix
    # aeq_dsnoparent[:n_index_betaij, self.var_betaij_x[dsnet.idx_beta_ij]
    # ] = np.eye(n_index_betaij)
    # aeq_dsnoparent[n_index_betaij:, self.var_betaji_x[dsnet.idx_beta_ji]
    # ] = np.eye(n_index_betaji)

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dsnoparent])
    model_x_rhs += beq_dsnoparent.tolist()
    model_x_senses += ['E'] * (n_microgrid)

    # -----------------------------------------------------------------------
    # power balance !!!
    aeq_dskclp = lil_matrix((n_interval * n_bus, n_all_vars_x))
    beq_dskclp = np.zeros(n_interval * n_bus)
    aeq_dskclq = lil_matrix((n_interval * n_bus, n_all_vars_x))
    beq_dskclq = np.zeros(n_interval * n_bus)

    for j_interval in range(n_interval):
        # pij
        aeq_dskclp[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_pij_x[:, j_interval])] = (
                dsnet.incidence_ds_tbus2line -
                dsnet.incidence_ds_fbus2line)
        # aggregate generation p
        aeq_dskclp[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_aggregate_pg_x[:, j_interval])] \
            = ssnet.mapping_station2dsbus.T
        # gama_load
        aeq_dskclp[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_gama_load_x[:, j_interval])] \
            = -dsnet.mapping_load2dsbus.T * np.diag(
            dsnet.pload[:, j_interval])
        # qij
        aeq_dskclq[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_qij_x[:, j_interval])] = (
                dsnet.incidence_ds_tbus2line -
                dsnet.incidence_ds_fbus2line)
        # aggregate generation q
        aeq_dskclq[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_aggregate_pg_x[:, j_interval])] \
            = ssnet.mapping_station2dsbus.T
        # gama_load
        aeq_dskclq[np.ix_(j_interval * n_bus + np.arange(n_bus),
                          self.var_gama_load_x[:, j_interval])] \
            = -dsnet.mapping_load2dsbus.T * np.diag(
            dsnet.qload[:, j_interval])

    model_x_matrix_a = spar.vstack([model_x_matrix_a, aeq_dskclp, aeq_dskclq])
    model_x_rhs += beq_dskclp.tolist() + beq_dskclq.tolist()
    model_x_senses += ['E'] * 2 * n_bus * n_interval
    '''
    # --------------------------------------------------------------------------
    # KVL with branch status
    large_m = 1e6
    n_line = dsnet.n_line
    v0 = dsnet.v0
    branch_r = dsnet.ppnet.line['branch_r_pu']
    branch_x = dsnet.ppnet.line['branch_x_pu']

    aineq_dskvl_u = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dskvl_u = large_m * np.ones(n_line * n_interval)
    aineq_dskvl_l = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dskvl_l = large_m * np.ones(n_line * n_interval)

    for j_interval in range(n_interval):
        # v_j^t - v_i^t <= M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_pij_x[:, j_interval])] = -np.diag(
            branch_r) / v0
        aineq_dskvl_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_qij_x[:, j_interval])] = -np.diag(
            branch_x) / v0
        aineq_dskvl_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_vm_x[:, j_interval])] = (
                dsnet.incidence_ds_tbus2line.T -
                dsnet.incidence_ds_fbus2line.T).toarray()
        aineq_dskvl_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = large_m * np.eye(
            n_line)

        # v_j^t - v_i^t >= -M(1-alphabranch) + (rij*pij + xij*qij) / v0
        aineq_dskvl_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_pij_x[:, j_interval])] = np.diag(
            branch_r) / v0
        aineq_dskvl_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_qij_x[:, j_interval])] = np.diag(
            branch_x) / v0
        aineq_dskvl_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_vm_x[:, j_interval])] = (
                dsnet.incidence_ds_fbus2line.T -
                dsnet.incidence_ds_tbus2line.T).toarray()
        aineq_dskvl_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = large_m * np.eye(
            n_line)

    model_x_matrix_a = spar.vstack(
        [model_x_matrix_a, aineq_dskvl_u, aineq_dskvl_l])
    model_x_rhs += bineq_dskvl_u.tolist() + bineq_dskvl_l.tolist()
    model_x_senses += ['L'] * 2 * n_line * n_interval

    # branch power limit pij and qij, respectively
    slmax = dsnet.slmax
    # alphabranch * -slmax <= pij <= alphabranch * slmax
    aineq_dspij_u = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspij_u = np.zeros(n_line * n_interval)
    aineq_dspij_l = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspij_l = np.zeros(n_line * n_interval)
    # alphabranch * -slmax <= qij <= alphabranch * slmax
    aineq_dsqij_u = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dsqij_u = np.zeros(n_line * n_interval)
    aineq_dsqij_l = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dsqij_l = np.zeros(n_line * n_interval)

    for j_interval in range(n_interval):
        # pij - alphabranch * slmax <= 0
        aineq_dspij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_pij_x[:, j_interval])] = np.eye(n_line)
        aineq_dspij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = -np.diag(slmax)
        # -pij - alphabranch * slmax <= 0
        aineq_dspij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_pij_x[:, j_interval])] = -np.eye(n_line)
        aineq_dspij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = -np.diag(slmax)
        # qij - alphabranch * slmax <= 0
        aineq_dsqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_qij_x[:, j_interval])] = np.eye(n_line)
        aineq_dsqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = -np.diag(slmax)
        # -qij - alphabranch * slmax <= 0
        aineq_dsqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_qij_x[:, j_interval])] = -np.eye(n_line)
        aineq_dsqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                             self.var_alpha_branch_x)] = -np.diag(slmax)

    model_x_matrix_a = spar.vstack(
        [model_x_matrix_a, aineq_dspij_u, aineq_dspij_l, aineq_dsqij_u,
         aineq_dsqij_l])
    model_x_rhs += bineq_dspij_u.tolist() + bineq_dspij_l.tolist() + bineq_dsqij_u.tolist() + bineq_dsqij_l.tolist()
    model_x_senses += ['L'] * 4 * n_line * n_interval

    # Branch power limit pij + qij and pij - qij
    # *** <= pij + qij <= ***
    aineq_dspijaddqij_u = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspijaddqij_u = np.zeros(n_line * n_interval)
    aineq_dspijaddqij_l = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspijaddqij_l = np.zeros(n_line * n_interval)
    # *** <= pij - qij <= ***
    aineq_dspijsubqij_u = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspijsubqij_u = np.zeros(n_line * n_interval)
    aineq_dspijsubqij_l = lil_matrix((n_line * n_interval, n_all_vars_x))
    bineq_dspijsubqij_l = np.zeros(n_line * n_interval)

    for j_interval in range(n_interval):
        # pij + qij <= ***
        aineq_dspijaddqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_pij_x[:, j_interval])] = np.eye(
            n_line)
        aineq_dspijaddqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_qij_x[:, j_interval])] = np.eye(
            n_line)
        aineq_dspijaddqij_u[np.ix_(j_interval * n_line +
                                   np.arange(n_line), self.var_alpha_branch_x)] \
            = -np.sqrt(2) * np.diag(slmax)
        # *** < pij + qij
        aineq_dspijaddqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_pij_x[:, j_interval])] = -np.eye(
            n_line)
        aineq_dspijaddqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_qij_x[:, j_interval])] = -np.eye(
            n_line)
        aineq_dspijaddqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_alpha_branch_x)] = -np.sqrt(
            2) * np.diag(slmax)
        # pij - qij <= ***
        aineq_dspijsubqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_pij_x[:, j_interval])] = np.eye(
            n_line)
        aineq_dspijsubqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_qij_x[:, j_interval])] = -np.eye(
            n_line)
        aineq_dspijsubqij_u[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_alpha_branch_x)] = -np.sqrt(
            2) * np.diag(slmax)
        # *** < pij - qij
        aineq_dspijsubqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_pij_x[:, j_interval])] = -np.eye(
            n_line)
        aineq_dspijsubqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_qij_x[:, j_interval])] = np.eye(
            n_line)
        aineq_dspijsubqij_l[np.ix_(j_interval * n_line + np.arange(n_line),
                                   self.var_alpha_branch_x)] = -np.sqrt(
            2) * np.diag(slmax)

    model_x_matrix_a = spar.vstack(
        [model_x_matrix_a, aineq_dspijaddqij_u, aineq_dspijaddqij_l,
         aineq_dspijsubqij_u, aineq_dspijsubqij_l])
    model_x_rhs += bineq_dspijaddqij_u.tolist() + \
                   bineq_dspijaddqij_l.tolist() + \
                   bineq_dspijsubqij_u.tolist() + \
                   bineq_dspijsubqij_l.tolist()
    model_x_senses += ['L'] * 4 * n_line * n_interval
    '''
    # ----------------------------------------------------------------------
    # Add constraints into Cplex model
    # model_x_matrix_a = csr_matrix(model_x_matrix_a)
    # a_rows = model_x_matrix_a.nonzero()[0].tolist()
    a_rows = model_x_matrix_a.row.tolist()
    # No computation, only query of attributes, faster than nonzero.
    # a_cols = model_x_matrix_a.nonzero()[1].tolist()
    a_cols = model_x_matrix_a.col.tolist()
    # tolist() is for 'non-integral value in input sequence'
    # a_data = model_x_matrix_a.data.tolist()
    #  model_x_matrix_a is csr_matrix matrix, a_data element needs to be
    # float
    # a_vals = model_x_matrix_a[model_x_matrix_a.nonzero()]

    # a_vals = model_x_matrix_a[model_x_matrix_a != 0]
    #  boolean mask index array, it's faster than nonzero,
    # but result is different?
    # Note that boolean mask returns 1D-array
    # a_vals = model_x_matrix_a[a_rows, a_cols].tolist()
    #  faster than boolean mask
    a_vals = model_x_matrix_a.data

    self.linear_constraints.add(rhs=model_x_rhs, senses=model_x_senses,
                                names=['constraint{0}'.format(i)
                                       for i in range(len(model_x_rhs))])

    self.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))

def sort_results_cpx(self):
    '''

    :param self:
    :return:
    '''

    self.re_status = self.solution.status[self.solution.get_status()]
    self.re_goal = self.solution.get_objective_value()

    # coloumn vector for solutions
    values = array(self.solution.get_values())
