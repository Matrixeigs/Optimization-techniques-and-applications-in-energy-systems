"""
-------------------------------------------------
   File Name：     pytess_main_4
   Description :
   Author :       yaoshuhan
   date：          12/11/18
-------------------------------------------------
   Change Activity:
                   12/11/18: Restruct the function
-------------------------------------------------
"""

from pytess.define_class import DistributionSystem, TransportationSystem, \
    StationSystem, TransportableEnergyStorage, TimeSpaceNetwork, \
    OptimizationModel

import cplex as cpx
import pandapower as pp

def pytess(solver='cplex'):
    '''

    :param solver:
    :return:
    '''

    from pypower.loadcase import loadcase
    from pytess.test_case import siouxfalls, case33, sscase, tesscase

    # Initialization of test systems
    # Initialization of distribution systems
    # ppnet_test = pp.from_excel()

    dsnet = DistributionSystem(name='Modified 33-bus test system',
                               ppc=loadcase(case33()))
    # Initialization of Sioux Falls transportation systems
    tsnet = TransportationSystem(name='Sioux Falls transportation network',
                                 tsc=siouxfalls())
    # Initialization of a station system including microgrids and depots
    ssnet = StationSystem(name=' system', ssc=sscase())
    # Initialization of mobile energy storage systems
    tessnet = TransportableEnergyStorage(name='mobile energy storage system',
                                         tessc=tesscase())
    # Initialization of time-space network
    tsnnet = TimeSpaceNetwork(name='Time-sapce network')

    # -------------------------------------------------------------------------

    MW_KW = 1000
    index_off_line = []

    # Time step
    dsnet.delta_t = 1
    # Time window
    n_timewindow_set = 2
    n_timewindow_effect = 1

    dsnet.init_load()
    n_interval = dsnet.n_interval

    # ssnet.add_microgrid2ext_grid(dsnet=dsnet)

    dsnet.update_fault_mapping(index_off_line)
    ssnet.map_station2dsts(dsnet=dsnet, tsnet=tsnet)
    ssnet.init_localload(dsnet=dsnet)
    ssnet.find_travel_time(tsnet=tsnet, tessnet=tessnet)

    tsnnet.set_tsn_model(dsnet=dsnet, tsnet=tsnet, ssnet=ssnet, tessnet=tessnet)

    dsnet.set_optimization_case()
    ssnet.set_optimization_case(dsnet=dsnet)
    tessnet.set_optimization_case(dsnet=dsnet, ssnet=ssnet)

    # ---------------------------------------------------------
    model_x = OptimizationModel()
    model_x.add_variables(dsnet=dsnet, ssnet=ssnet,
                          tessnet=tessnet, tsnnet=tsnnet)
    model_x.add_objectives(dsnet=dsnet, ssnet=ssnet,
                           tessnet=tessnet, tsnnet=tsnnet)
    model_x.add_constraints(dsnet=dsnet, ssnet=ssnet,
                            tessnet=tessnet, tsnnet=tsnnet)
    model_x.solve()
    # Export model to file *.mps
    # model_x.write('tess_model_1.mps')

    print(model_x.solution.get_objective_value())
    x = model_x.solution.get_values()# The solutions


    # All power values are given in the consumer system, therefore p_kw is
    # positive if the external grid is absorbing
    # power and negative if it is supplying power.
    # but we choose the same convention as pypower. ext_grid and gen represent
    # output parameters.
    # todo update data structure that can be self-adaptive to expanding network
    ext_grid['max_p_kw'] = array([1.6, 1.6, 1.8]) * MW_KW
    # ext_grid['max_p_kw'] = array([[1.6, 1.6, 1.8],
    #                               [1.6, 1.6, 1.8],
    #                               [1.6, 1.6, 1.8]
    #                              ]).ravel() * MW_KW  # Use .loc for indexing
    # if not the whole column
    ext_grid['min_p_kw'] = 0  # Need to correct min_p_kw in pandapower
    # For max_q_kw and min_q_kw
    ext_grid['max_q_kvar'] = ppnet.ext_grid['max_p_kw'] * 0.8
    ext_grid['min_q_kvar'] = -ppnet.ext_grid['max_p_kw'] * 0.8

    # Configure bus voltage upper and lower bound
    bus['max_vm_pu'] = 1.05
    bus['min_vm_pu'] = 0.95

    # --------------------------------------------------------------------------
    if solver=='cplex':
        pass
    elif solver=='gurobi':
        pass
    else:
        pass

    pass



if __name__ == '__main__':
    pytess()