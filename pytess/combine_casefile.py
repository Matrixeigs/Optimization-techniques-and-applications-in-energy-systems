from numpy import array, concatenate

from pypower.idx_bus import BASE_KV, BUS_I, PQ, PV, REF, BUS_TYPE
from pypower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
from pypower.idx_gen import GEN_BUS

from copy import deepcopy

def combine_ppc(*args):

    # Initialization of ppc_joint from the first argument
    # ppc_joint = {}
    #
    # # for e_key in args[0].keys():
    # #     ppc_joint[e_key] = args[0][e_key]

    ppc_joint = deepcopy(args[0])

    for e_ppc in args[1:]:  # omit the first argument
        e_ppc = deepcopy(e_ppc)  # *** important！！！

        bus_offset = ppc_joint['bus'][-1, BUS_I] + 1

        e_ppc['bus'][:, BUS_I] += bus_offset
        ppc_joint['bus'] = concatenate((ppc_joint['bus'], e_ppc['bus']), axis=0)

        e_ppc['gen'][:, GEN_BUS] += bus_offset
        ppc_joint['gen'] = concatenate((ppc_joint['gen'], e_ppc['gen']), axis=0)

        e_ppc['branch'][:, [F_BUS, T_BUS]] += bus_offset
        ppc_joint['branch'] = concatenate((ppc_joint['branch'], e_ppc['branch']), axis=0)

        ppc_joint['areas'] = concatenate((ppc_joint['areas'], e_ppc['areas']), axis=0)

        ppc_joint['gencost'] = concatenate((ppc_joint['gencost'], e_ppc['gencost']), axis=0)


    return ppc_joint