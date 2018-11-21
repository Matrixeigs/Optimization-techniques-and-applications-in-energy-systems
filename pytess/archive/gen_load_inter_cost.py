from numpy import array, arange, setdiff1d, hstack, zeros, in1d


def gen_load_inter_cost_ppc():
    """
    
    :return:
    """

    load_inter_cost = zeros(33)

    prioa = array([2, 4, 7, 8, 14, 24, 25, 29, 30, 31, 32, 15, 16, 20, 22]) - 1
    priob = array([])

    prioc = arange(33)
    prioc = setdiff1d(prioc, hstack((prioa, priob)))

    load_inter_cost[prioa] = 10
    # loadInterCost[priob] = 2
    load_inter_cost[prioc] = 2

    return load_inter_cost


def gen_load_inter_cost_net(ppnet):
    '''

    :param ppnet:
    :return:
    '''

    bus, load = ppnet.bus, ppnet.load

    prioa = array([2, 4, 7, 8, 14, 24, 25, 29, 30, 31, 32, 15, 16, 20, 22]) - 1
    priob = array([])

    prioc = arange(99)
    prioc = setdiff1d(prioc, hstack((prioa, priob)))

    # numpy.in1d(ar1, ar2, assume_unique=False, invert=False) -
    # Test whether each element of a 1-D array ar1 is also present in a second array ar2.
    load.loc[in1d(load['bus'], prioa), 'load_cost'] = 10
    load.loc[in1d(load['bus'], prioc), 'load_cost'] = 2


    return ppnet

# if __name__ == '__main__':
#     a = gen_load_inter_cost_ppc()
#     b = 1