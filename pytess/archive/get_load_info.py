from numpy import unique, nonzero, array, ix_, in1d

def get_load_information_ppc(ppc, BUS_I, LOAD_COST, LOAD_TYPE):
    '''

    :param ppc:
    :param BUS_I:
    :param LOAD_COST:
    :param LOAD_TYPE:
    :return: load_information: list of arrays, each item refers to each type of load and reveal load bus number and type
    '''

    bus = ppc['bus']
    n_load_type = unique(bus[:, LOAD_TYPE]).shape[0]

    load_information = []

    for i_load_type in range(n_load_type):
        index_load = nonzero(bus[:, LOAD_TYPE] == i_load_type + 1)[0]
        load_information.append(bus[ix_(index_load, [BUS_I, LOAD_COST])])

    return load_information

def get_load_information_net(ppnet, BUS_I='name', LOAD_COST='load_cost', LOAD_TYPE='load_type'):
    """

    :param ppnet: pand
    :param BUS_I:
    :param LOAD_COST:
    :param LOAD_TYPE:
    :return: load_information: dictionary of arrays, each item refers to each type of load and reveal load bus number
    and load cost
    """

    load = ppnet.load
    load_information = {}

    # n_load_type = unique(bus[LOAD_TYPE]).shape[0]
    # The prefix e means element
    # for e_load_type in unique(load[LOAD_TYPE]):
    #     index_load = nonzero(bus[LOAD_TYPE] == e_load_type)[0]
    #     load_information.append(bus.loc[index_load, [BUS_I, LOAD_COST]])

    for e_load_type in unique(load[LOAD_TYPE]):
        load_information[e_load_type] = load.loc[load['load_type'] == e_load_type, [BUS_I, LOAD_COST]]

    return load_information