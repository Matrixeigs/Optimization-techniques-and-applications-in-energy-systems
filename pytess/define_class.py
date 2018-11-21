import numpy as np
from numpy import array, zeros, arange, ones

from pandas import DataFrame
import scipy.sparse as spar
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import cplex as cpx

class CaseParam():
    def __init__(self, name):
        self.name = name


class DistributionSystem():
    '''

    '''

    def __init__(self, name, ppc):

        import pandapower.converter as pc

        self.name = name
        # create pandapower net from ppc
        self.ppnet = pc.from_ppc(ppc)
        # Convert MW to kW
        MW_KW = 1000

        # shortener, note it is view other than copy.
        bus = self.ppnet.bus
        line = self.ppnet.line
        ext_grid = self.ppnet.ext_grid
        load = self.ppnet.load
        gen = self.ppnet.gen

        # since line will be modified, store the original line, load
        self.origin_line = line

        # Get bus index lists of each type of bus
        self.idx_ref_bus = np.array(ext_grid['bus'], dtype=int)
        self.idx_pv_bus = np.array(gen['bus'], dtype=int)
        self.idx_pq_bus = np.setdiff1d(np.array(bus.index, dtype=int),
                               np.hstack((self.idx_ref_bus, self.idx_pv_bus)))

        # The no. of bus, ... in distribution system
        self.n_bus = bus.shape[0]
        self.n_line = line.shape[0]
        self.n_ext_grid = ext_grid.shape[0]
        self.n_load = load.shape[0]

        # Base value for S capacity
        self.sn_mva = self.ppnet.sn_kva / MW_KW
        self.vn_kv = bus.loc[0, 'vn_kv']
        # Base value for current
        self.in_ka = self.sn_mva / (np.sqrt(3) * self.vn_kv)

        # branch resistance and reactance
        # check if there are transformers?
        # if no trafo:
        line['branch_r_pu'] = (line['r_ohm_per_km'] * line['length_km']
                            / line['parallel'] /
                        ((self.vn_kv * 1e3) ** 2 / (self.sn_mva * 1e6)))
        line['branch_x_pu'] = (line['x_ohm_per_km'] * line['length_km']
                            / line['parallel'] /
                        ((self.vn_kv * 1e3) ** 2 / (self.sn_mva * 1e6)))

        # Find all the immediately neighboring nodes connecting to each MG bus
        # The index for branches starting from MG bus
        # self.idx_beta_ij = line['from_bus'].index[
        #     np.isin(line['from_bus'], self.idx_ref_bus)].values
        #
        # # self.idx_beta_ij = array()
        # # The index for branches ending at MG bus
        # self.idx_beta_ji = line['to_bus'].index[
        #     np.isin(line['to_bus'], self.idx_ref_bus)].values

    def init_load(self):
        '''

        :return:
        '''

        from pytess.load_info import init_load_type_cost, init_load_profile, \
            get_load_info

        # generate load type, load interruption cost and load profile
        self.ppnet = init_load_type_cost(self.ppnet)

        (self.pload, self.qload, self.n_interval) = \
            init_load_profile(load=self.ppnet.load, dsnet=self)

        # consolidate load information
        self.load_information = get_load_info(self.ppnet)

    def update_fault_mapping(self, idx_off_line):
        '''
        Update distribution system considering outage lines and buses
        :param idx_off_line:
        :return:
        '''

        import copy
        import pandapower as pp
        import pandapower.topology as ptop

        bus = self.ppnet.bus
        ext_grid = self.ppnet.ext_grid
        gen = self.ppnet.gen
        load = self.ppnet.load
        # the updated line should be a copy other than a view
        line = copy.deepcopy(self.origin_line)
        # view to self.line
        self.ppnet.line = line

        # Find faults and update self.ppnet
        line['in_service'] = True
        bus['in_service'] = True

        line.loc[idx_off_line, 'in_service'] = False
        # Identify areas and remove isolated ones -------------------------------------------------------------------------
        # Set all isolated buses and all elements connected to isolated buses
        # out of service.
        # Before this, it needs to add microgrid to ext_grid
        pp.set_isolated_areas_out_of_service(self.ppnet)  # isolated means that
        # a bus is disconnected from ext_grid
        idx_off_bus = bus[bus['in_service'] == False].index.values
        idx_off_line = line[line['in_service'] == False].index.values

        # Remove all the faulty lines and update no. of lines
        line.drop(labels=idx_off_line, inplace=True)
        # Update the no. of lines
        self.n_line = line.shape[0]
        # Reset line index starting at zero
        line.reset_index(drop=True, inplace=True)

        # Mask outage load in mapping of load to distribution bus
        idx_off_load = np.flatnonzero(np.isin(load['bus'], idx_off_bus))
        idx_on_load = np.setdiff1d(load.index, idx_off_load)
        n_on_load = idx_on_load.size  # no. of on load
        # the mask of load on/off status, (n_load, 1)
        self.on_off_load = np.isin(load.index, idx_on_load)
        # update mapping function of load to distribution bus
        self.mapping_load2dsbus = csr_matrix(
            (ones(n_on_load), (idx_on_load, load.loc[idx_on_load, 'bus'])),
            shape=(self.n_load, self.n_bus), dtype=int)

        # set distribution system graph
        self.dsgraph = ptop.create_nxgraph(self.ppnet)

        # # todo update index lists of each type of bus
        # self.idx_ref_bus = np.array(ext_grid['bus'], dtype=int)
        # self.idx_pv_bus = np.array(gen['bus'], dtype=int)
        # self.idx_pq_bus = np.setdiff1d(np.array(bus.index, dtype=int),
        #                        np.hstack((self.idx_ref_bus, self.idx_pv_bus)))

        # Find all the immediately neighboring nodes connecting to each MG bus
        # The index for branches starting from MG bus
        # self.idx_beta_ij = line['from_bus'].index[
        #     np.isin(line['from_bus'], self.idx_ref_bus)].values
        # # The index for branches ending at MG bus
        # self.idx_beta_ji = line['to_bus'].index[
        #     np.isin(line['to_bus'], self.idx_ref_bus)].values

        # set up new incidence matrix of distribution bus to lines
        # from_bus to line
        self.incidence_ds_fbus2line = csr_matrix(
            (ones(self.n_line), (range(self.n_line), line['from_bus'])),
            shape=(self.n_line, self.n_bus), dtype=int).T
        # to_bus to line
        self.incidence_ds_tbus2line = csr_matrix(
            (ones(self.n_line), (range(self.n_line), line['to_bus'])),
            shape=(self.n_line, self.n_bus), dtype=int).T

    def set_optimization_case(self):
        '''

        :return:
        '''
        line = self.ppnet.line
        bus = self.ppnet.bus
        ext_grid = self.ppnet.ext_grid
        load = self.ppnet.load

        # load interruption cost at each interval, (n_load, n_interval)
        self.load_interruption_cost = np.tile(
            A=load['load_cost'][:, np.newaxis], reps=(1, self.n_interval))

        # parameters for line capacity limit
        self.slmax = line['max_i_ka'] / self.in_ka

        # Upper bounds
        # line capacity for active power at each interval, (n_line, n_interval)
        self.pij_u = np.tile(A=self.slmax[:, np.newaxis],
                             reps=(1, self.n_interval))
        # line capacity for reactive power at each interval,
        # (n_line, n_interval)
        self.qij_u = np.tile(A=self.slmax[:, np.newaxis],
                             reps=(1, self.n_interval))
        # line capacity for apparent power at each interval,
        # (n_line, n_interval
        self.sij_u = np.tile(A=self.slmax[:, np.newaxis],
                             reps=(1, self.n_interval))
        # bus voltage
        # v0 is for bus voltage constant
        self.v0 = 1
        self.vm_u = np.tile(A=bus['max_vm_pu'][:, np.newaxis],
                            reps=(1, self.n_interval))
        self.vm_u[ext_grid['bus'].astype(int), :] = self.v0

        # Lower bounds
        # since the power flow on line is bidirectional
        self.pij_l = np.tile(A=-self.slmax[:, np.newaxis],
                             reps=(1, self.n_interval))
        self.qij_l = np.tile(A=-self.slmax[:, np.newaxis],
                             reps=(1, self.n_interval))
        # bus voltage
        self.vm_l = np.tile(A=bus['min_vm_pu'][:, np.newaxis],
                            reps=(1, self.n_interval))
        self.vm_l[ext_grid['bus'].astype(int), :] = self.v0


class TransportationSystem():
    '''

    '''

    def __init__(self, name, tsc):
        '''
        Initialization of transportation system instance
        :param name:
        :param tsc: transportation system case, in ndarray type.
        '''
        import networkx as nx

        # instance's name
        self.name = name

        # form DataFrame from ndarray
        self.node = DataFrame(tsc['node'])
        self.n_node = self.node.shape[0]

        self.edge = DataFrame(tsc['edge'])
        self.n_edge = self.node.shape[0]

        # graph for transportation network
        self.graph = nx.from_pandas_edgelist(df=self.edge, source='init_node',
                                             target='term_node',
                                             edge_attr='length')


class StationSystem():
    def __init__(self, name, ssc):
        '''
        Initialization of station system and identify station types.
        :param name:
        :param ssc:
        '''

        self.name = name
        # form DataFrame from ndarray
        self.station = DataFrame(ssc['station'])
        # sort the DataFrame self.station by bus no. and reset index
        self.station.sort_values(by=['bus_i', 'node_i'], inplace=True)
        self.station.reset_index(drop=True, inplace=True)

        # set depot's power parameters to zero
        self.station.loc[self.station['station_type'] == 'depot',
                                                'max_p_kw':'qload_kvar'] = 0

        # find the index of depot and microgrid, ndarray
        self.idx_station = self.station.index.values
        self.idx_depot = self.station[self.station['station_type']
                                      == 'depot'].index.values
        self.idx_microgrid = self.station[self.station['station_type']
                                          == 'microgrid'].index.values
        # no. of various types of stations
        self.n_station = self.idx_station.size
        self.n_depot = self.idx_depot.size
        self.n_microgrid = self.idx_microgrid.size

    def init_localload(self, dsnet):


        from load_info import init_load_profile

        self.p_localload, self.q_localload, no_use = init_load_profile(
            load=self.station, dsnet=dsnet,
            P_LOAD='pload_kw', Q_LOAD='qload_kvar')

    def map_station2dsts(self, dsnet, tsnet):
        '''
        construct mapping of station to distribution bus and transportation node
        :return:
        '''

        from scipy.sparse import csr_matrix

        # # list(chain(*a)) convert list of list to list
        # # only update microgrids' bus, not including depots' bus
        # # it updates the microgrids' bus no. in current distribution network
        # self.station.loc[self.idx_microgrid, 'bus_i'] = [
        #     dsnet.ppnet.bus.loc[
        #         dsnet.ppnet.bus['e_name']==e_station, 'name'].tolist()[0]
        #     for e_station in self.station['bus_i'] if e_station >= 0]
        #
        # # Update station's node no. in current transportation network
        # self.station['node_i'] = [tsnet.node.loc[
        #             tsnet.node['e_node_i']==e_station, 'node_i'].tolist()[0]
        #     for e_station in self.station['node_i']]

        # extract idx_depot and idx_microgrid
        idx_depot = self.idx_depot
        idx_microgrid = self.idx_microgrid
        n_depot = self.n_depot
        n_microgrid = self.n_microgrid
        n_station = self.n_station

        # no. of station, bus and node
        n_station = self.n_station
        n_bus = dsnet.n_bus
        n_node = tsnet.n_node

        # each microgrid corresponds to each distribution bus
        # (n_station, n_bus)
        self.mapping_station2dsbus = csr_matrix(
            (ones(n_microgrid), (idx_microgrid,
                                 self.station.loc[idx_microgrid, 'bus_i'])),
            shape=(n_station, n_bus), dtype=int)

        # each station corresponds to each transportation node
        # (n_station, n_node)
        self.mapping_station2tsnode = csr_matrix(
            (ones(n_station), (range(n_station), self.station['node_i'])),
            shape=(n_station, n_node), dtype=int)

    def find_travel_time(self, tsnet, tessnet):
        '''
        Find the shortest path among all the microgrid and depot stations.
        :param ssnet:
        :return:
        '''
        from itertools import combinations, permutations
        import networkx as nx

        # ndarray to represent shortest path length, (n_station, n_station),
        # square matrix
        shortest_path_length = zeros((self.n_station, self.n_station))
        # only compute the upper triangle by combinations
        for e_s_node, e_t_node in combinations(self.idx_station, 2):
            shortest_path_length[e_s_node, e_t_node] = nx.shortest_path_length(
                G=tsnet.graph, source=self.station.loc[e_s_node, 'node_i'],
                target=self.station.loc[e_t_node, 'node_i'], weight='length')
        # form the symmetric matrix from upper triangle
        shortest_path_length += shortest_path_length.T - np.diag(
            shortest_path_length.diagonal())

        # To set shortest path
        shortest_path = zeros((self.n_station, self.n_station),
                              dtype=object)
        # compute all the permutations
        for e_s_node, e_t_node in permutations(self.idx_station, 2):
            shortest_path[e_s_node, e_t_node] = nx.shortest_path(G=tsnet.graph,
            source=self.station.loc[e_s_node, 'node_i'],
            target=self.station.loc[e_t_node, 'node_i'], weight='length')

        # todo need to improve to multi-layer
        travel_time = np.ceil(shortest_path_length
                              / tessnet.tess['avg_v_km/h'].mean()).astype(int)

        # store in class
        self.shortest_path_length = shortest_path_length
        # store shortest_path
        self.shortest_path = shortest_path
        self.travel_time = travel_time
        # return path_length_table # it's view not copy

    def set_optimization_case(self, dsnet):

        n_interval = dsnet.n_interval
        # upper bounds of active/reactive power output of station
        # (including depot)
        self.station_p_u = np.tile(A=self.station['max_p_kw'][:, np.newaxis]
                            / dsnet.ppnet.sn_kva, reps=(1, n_interval))
        self.station_q_u = np.tile(A=self.station['max_q_kvar'][:, np.newaxis]
                            / dsnet.ppnet.sn_kva, reps=(1, n_interval))

        # lower bounds of active/reactive power output of station
        self.station_p_l = np.tile(A=self.station['min_p_kw'][:, np.newaxis]
                            / dsnet.ppnet.sn_kva, reps=(1, n_interval))
        self.station_q_l = np.tile(A=self.station['min_q_kvar'][:, np.newaxis]
                            / dsnet.ppnet.sn_kva, reps=(1, n_interval))

        # station energy capacity of station
        ratio_capacity = 0.6
        ratio_reserve = 0.1

        # bounds for energy capacity of station, (n_station, n_interval)
        self.station_e_u = self.station['max_p_kw'] \
                    * ratio_capacity * dsnet.n_interval / dsnet.ppnet.sn_kva
        self.station_e_u = np.tile(A=self.station_e_u[:,np.newaxis],
                                   reps=(1, dsnet.n_interval))

        self.station_e_l = self.station_e_u * ratio_reserve

        # station generation cost, (n_station, n_interval)
        self.station_gencost = np.tile(
            A=self.station['cost_$/kwh'][:, np.newaxis], reps=(1, n_interval))


class TransportableEnergyStorage():
    def __init__(self, name, tessc):
        # instance's name
        self.name = name
        # form DataFrame from ndarray
        self.tess = DataFrame(tessc['tess'])
        # The no. of tess
        self.n_tess = self.tess.shape[0]

    def set_optimization_case(self, dsnet, ssnet):

        # upper and lower bounds for tess's charging/discharging power
        self.tess_pch_u = self.tess['ch_p_kw'].values / dsnet.ppnet.sn_kva
        self.tess_pdch_u = self.tess['dch_p_kw'].values / dsnet.ppnet.sn_kva


        # tess' energy capacity
        self.tess_cap_e = self.tess['cap_e_kwh'] / dsnet.ppnet.sn_kva

        # tess's initial energy
        self.tess_e_init = self.tess_cap_e * self.tess['init_soc']
        # tess's energy upper/lower bounds
        self.tess_e_u = np.tile(A=self.tess_cap_e * self.tess['max_soc'],
                                reps=(1, dsnet.n_interval))
        self.tess_e_l = np.tile(A=self.tess_cap_e * self.tess['min_soc'],
                                reps=(1, dsnet.n_interval))

        # charging/discharging efficiency
        self.tess_ch_efficiency = self.tess['ch_efficiency'].values
        self.tess_dch_efficiency = self.tess['dch_efficiency'].values

        # The battery maintenance  and transportation cost (n_tess, n_interval)
        self.tess_cost_power = np.tile(
            A=self.tess['cost_power'][:, np.newaxis],
            reps=(1, dsnet.n_interval))

        self.tess_cost_transportation = np.tile(
            A=self.tess['cost_transportation'][:, np.newaxis],
            reps=(1, dsnet.n_interval))

        # Indicate tess's initial position in transportation system, (n_tess, )
        idx_tess_init_location = array([np.flatnonzero(
            ssnet.station['node_i'] == self.tess['init_location'][i_tess])
             for i_tess in range(self.n_tess)]).ravel()
        # mapping of each tess's initial location to transportation node
        self.mapping_tess_init2tsnode = csr_matrix(
            (ones(self.n_tess), (range(self.n_tess), idx_tess_init_location)),
            shape=(self.n_tess, ssnet.n_station))


class TimeSpaceNetwork():
    '''

    '''

    def __init__(self, name):
        self.name = name

    # def __repr__(self):  # pragma: no cover
    #     r = "This pandapower network includes the following parameter tables:"
    #     par = []
    #     res = []
    #     for tb in list(self.keys()):
    #         if isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
    #             if 'res_' in tb:
    #                 res.append(tb)
    #             else:
    #                 par.append(tb)
    #     for tb in par:
    #         length = len(self[tb])
    #         r += "\n   - %s (%s %s)" % (
    #         tb, length, "elements" if length > 1 else "element")
    #     if res:
    #         r += "\n and the following results tables:"
    #         for tb in res:
    #             length = len(self[tb])
    #             r += "\n   - %s (%s %s)" % (
    #             tb, length, "elements" if length > 1 else "element")
    #     return r

    def set_tsn_model(self, dsnet, tsnet, ssnet, tessnet):
        """Formation of time-space network, referring to model and figure in
        paper
        :param tsc:
        :param n_interval:
        :return:
        """
        # temporary parameters for tsc
        from itertools import permutations
        from scipy.sparse import csr_matrix, lil_matrix

        travel_time = ssnet.travel_time
        # extract time interval
        n_interval = dsnet.n_interval
        n_timepoint = n_interval + 1
        # no. of station on map
        n_station = ssnet.n_station
        # identify depot and microgrid
        idx_depot = ssnet.idx_depot
        idx_microgrid = ssnet.idx_microgrid
        n_depot = ssnet.n_depot
        n_microgrid = ssnet.n_microgrid

        # The number of tsn nodes in tsn graph
        n_tsn_node_source = n_depot  # For source node at the time point 0
        n_tsn_node_microgrid = n_microgrid * (n_timepoint-2)
        n_tsn_node_sink = n_depot  # For source node at the the last time point
        # total tsn_node
        n_tsn_node = n_tsn_node_source + n_tsn_node_microgrid + n_tsn_node_sink

        # indexing of tsn nodes, ndarray(n_station, n_timepoint)
        # in this array, (i, j) indicates the tsn node for i-th station at
        # time point j, if value >= 0, if value < 0, meaning there is no
        # associated tsn node, so the - 10 is for this reason.
        tsn_node = zeros((n_station, n_timepoint), dtype=int) - 10
        # at time point 0
        tsn_node[idx_depot, 0] = arange(n_tsn_node_source)
        # from time point 1 to second-to-last
        tsn_node[idx_microgrid, 1:(n_timepoint-1)] = arange(n_tsn_node_source,
                              n_tsn_node_source+n_tsn_node_microgrid).reshape(
            n_microgrid, n_timepoint-2, order='F')
        # at the last time point
        tsn_node[idx_depot, -1] = arange(n_tsn_node-n_tsn_node_sink,n_tsn_node)

        # Set up ndarray (??, 3) indicating tsn arcs
        #  [from_tsn_node to_tsn_node travel time]
        # column index
        F_TSN_NODE = 0
        T_TSN_NODE = 1
        TRAVEL_TIME = 2
        # for source arcs
        tsn_arc_source = np.array([
            [tsn_node[e_depot, 0],
             tsn_node[e_microgrid, 0+travel_time[e_depot, e_microgrid]],
             travel_time[e_depot, e_microgrid]]
            for e_depot in idx_depot
            for e_microgrid in idx_microgrid])

        # for sink arcs
        tsn_arc_sink = np.array([
            [tsn_node[e_microgrid, -1-travel_time[e_microgrid, e_depot]],
             tsn_node[e_depot, -1],
             travel_time[e_microgrid, e_depot]]
            for e_depot in idx_depot
            for e_microgrid in idx_microgrid])

        # fro normal arcs (transportation among microgrids)
        tsn_arc_normal = np.zeros((0, 3))

        for j_timepoint in range(1, n_timepoint-2):

            tsn_arc_moving_temp = np.array([
                [tsn_node[s_microgrid, j_timepoint],
                 tsn_node[t_microgrid,
                          j_timepoint+travel_time[s_microgrid, t_microgrid]],
                 travel_time[s_microgrid, t_microgrid]]
                for s_microgrid, t_microgrid in permutations(idx_microgrid, 2)
                if j_timepoint+travel_time[s_microgrid, t_microgrid]
                   <= n_timepoint-2])

            tsn_arc_holding_temp = np.array([
                [tsn_node[e_microgrid, j_timepoint],
                 tsn_node[e_microgrid, j_timepoint+1],
                 0]
                for e_microgrid in idx_microgrid])

            if tsn_arc_moving_temp.shape[0]:
                tsn_arc_normal = np.vstack([tsn_arc_normal,
                                    tsn_arc_moving_temp, tsn_arc_holding_temp])
            else:
                tsn_arc_normal = np.vstack([tsn_arc_normal,
                                            tsn_arc_holding_temp])
        # Consolidate all arcs
        tsn_arc = np.vstack([tsn_arc_source, tsn_arc_normal, tsn_arc_sink])

        n_tsn_arc = tsn_arc.shape[0]
        # index of the parking arc position in tsn_arc_table,
        # (n_station, n_interval)

        idx_tsn_arc_holding = zeros((n_station, n_interval), dtype=int)
        idx_tsn_arc_holding[idx_microgrid, 1:-1] = np.flatnonzero(
            tsn_arc[:, TRAVEL_TIME] == 0).reshape((n_microgrid, n_interval-2),
                                                  order='F')
        # convert to True/False matrix to indicate holding arcs at each interval
        tsn_arc_holding = zeros((n_tsn_arc, n_interval), dtype=bool)
        for j_interval in range(1, n_interval-1):
            tsn_arc_holding[idx_tsn_arc_holding[idx_microgrid, j_interval],
                            j_interval] = 1

        # ------------------------------------------------------------
        # tsn_cut_set matrix
        tsn_cut_set = zeros((n_tsn_arc, n_interval), dtype=bool)

        for j_interval in range(n_interval):
            # & bit-operators, parathesis is compulsory
            # a cut-set for each interval
            tsn_cut_set[:, j_interval] = (tsn_arc[:, F_TSN_NODE] <=
                max(tsn_node[:, j_interval])) & (
                    tsn_arc[:, T_TSN_NODE] > max(tsn_node[:, j_interval]))
        # tsn_cut_set = lil_matrix(tsn_cut_set)  #lil_matrix causes problem
        # with value assignment to ndarray

        # mapping_tsn_f,t matrix (n_tsn_node, n_tsn_arc), indicate each
        # arc's from bus and to bus
        mapping_tsn_f2arc = csr_matrix(
            (ones(n_tsn_arc), (tsn_arc[:, F_TSN_NODE], range(n_tsn_arc))),
            shape=(n_tsn_node, n_tsn_arc), dtype=int)

        mapping_tsn_t2arc = csr_matrix(
            (ones(n_tsn_arc), (tsn_arc[:, T_TSN_NODE], range(n_tsn_arc))),
            shape=(n_tsn_node, n_tsn_arc), dtype=int)

        # store in class
        self.n_tsn_node_source = n_tsn_node_source
        self.n_tsn_ndoe_microgrid = n_tsn_node_microgrid
        self.n_tsn_node_sink = n_tsn_node_sink
        self.n_tsn_node = n_tsn_node
        self.tsn_node = tsn_node
        self.tsn_arc = tsn_arc
        # self.idx_tsn_arc_holding = idx_tsn_arc_holding
        self.tsn_arc_holding = tsn_arc_holding
        self.n_tsn_arc = n_tsn_arc
        self.tsn_cut_set = tsn_cut_set
        self.mapping_tsn_f2arc = mapping_tsn_f2arc
        self.mapping_tsn_t2arc = mapping_tsn_t2arc


class OptimizationModel(cpx.Cplex):


    def add_variables(self, dsnet, ssnet, tessnet, tsnnet):

        from pytess.optimization_model_toolbox import add_variables_cpx

        add_variables_cpx(self=self, dsnet=dsnet, ssnet=ssnet,
                          tessnet=tessnet, tsnnet=tsnnet)

    def add_objectives(self, dsnet, ssnet, tessnet, tsnnet):

        # Add Objective function
        # In comparison with a = zip(var_sign_onroad_x.tolist(),
        # [cost_ev_transit] * var_sign_onroad_x.size)
        # transportation cost (n_tess, n_interval)
        # zip(a, b) a is index and must conduct .tolist() but
        # b doesn't need to do .tolist()
        tess_cost_transportation = tessnet.tess_cost_transportation
        self.objective.set_linear(zip(self.var_sign_onroad_x.ravel().tolist(),
                tess_cost_transportation.ravel()))

        # charging cost, (n_tess, n_interval)
        sn_kva = dsnet.ppnet.sn_kva
        # note the dimension mismatch between var_tess2st_pch_x and
        # tess_cost_power, so make corrections !!!
        tess_cost_power = np.tile(tessnet.tess_cost_power[:, np.newaxis, :],
                                  reps=(1, ssnet.n_station, 1))
        delta_t = dsnet.delta_t

        self.objective.set_linear(zip(self.var_tess2st_pch_x.ravel().tolist(),
                                (tess_cost_power * delta_t * sn_kva).ravel()))

        # discharging cost
        self.objective.set_linear(zip(self.var_tess2st_pdch_x.ravel().tolist(),
                                (tess_cost_power * delta_t * sn_kva).ravel()))

        # generation cost
        station_gencost = ssnet.station_gencost
        self.objective.set_linear(zip(self.var_station_p_x.ravel().tolist(),
                                (station_gencost * delta_t * sn_kva).ravel()))

        # load interruption cost
        load_interruption_cost = dsnet.load_interruption_cost
        self.objective.set_linear(
            zip(self.var_gama_load_x.ravel().tolist(),
                (-load_interruption_cost * dsnet.pload *
                                            delta_t * sn_kva).ravel()))

        # Add objective constant
        self.objective.set_offset(
            (load_interruption_cost * dsnet.pload *
                                delta_t * sn_kva).ravel().sum())

        # Set objective sense
        self.objective.set_sense(self.objective.sense.minimize)

    def add_constraints(self, dsnet, ssnet, tessnet, tsnnet):

        from pytess.optimization_model_toolbox import add_constraints_cpx

        add_constraints_cpx(self=self, dsnet=dsnet, ssnet=ssnet,
                            tessnet=tessnet, tsnnet=tsnnet)

    def sort_results(self):

        from pytess.optimization_model_toolbox import sort_results_cpx

        sort_results_cpx(self=self)

    def query_error(self, constraint_rows):
        '''
        Query the variable names involved in specific constraints
        :param self: cplex model
        :param constraint_rows: the indices of constraints
        :return:
        '''

        # get indices and values of variables involved in infeasible constraints
        ind_var, val_var = self.linear_constraints.get_rows(
            constraint_rows).unpack()
        # get variable names
        name_var = self.variables.get_names(ind_var)

        return name_var

if __name__ == '__main__':

    from pypower.loadcase import loadcase
    from pytess.test_case import siouxfalls, case33, sscase, tesscase

    # dsnet = pc.from_ppc(loadcase(case33()))
    dsnet = DistributionSystem(name='This is distribution system',
                               ppc=loadcase(case33()))
    # aa= TsnModel()

    # traffic_network_case = case3_tsn.transportation_network()
    # tsn_model = aa.set_tsn_model(traffic_network_case, 3)
    tsnet = TransportationSystem(name='this is for test', tsc=siouxfalls())

    ssnet = StationSystem(name='microgrid system', ssc=sscase())

    tessnet = TransportableEnergyStorage(name='mobile energy storage system',
                                         tessc=tesscase())

    tsnnet = TimeSpaceNetwork(name='time sapce network')

    ############################################################################
    dsnet.init_load()
    dsnet.update_fault_mapping([0])
    ssnet.map_station2dsts(dsnet=dsnet, tsnet=tsnet)

    ssnet.find_travel_time(tsnet=tsnet, tessnet=tessnet)

    tsnnet.set_tsn_model(dsnet=dsnet, tsnet=tsnet, ssnet=ssnet, tessnet=tessnet)

    dsnet.set_optimization_case()

    ###########################################################################
    model_x = OptimizationModel()

    pass







