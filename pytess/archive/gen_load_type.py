from numpy import array, arange, setdiff1d, hstack, zeros, in1d

def gen_load_type_ppc():
    """

    :return:
    """
    category_i = array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]) - 1  # 1-base to 0-base
    category_c = array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) - 1

    category_r = arange(33)
    category_r = setdiff1d( category_r, hstack((category_i, category_c)) )

    load_type = zeros(33, dtype=int)
    load_type[category_i] = 1  # industrial load
    load_type[category_c] = 2  # commercial load
    load_type[category_r] = 3  # residential load

    return load_type


def gen_load_type_net(ppnet):
    """

    :return:
    """
    load = ppnet.load
    n_load = load.shape[0]
    n_bus = ppnet.bus.shape[0]

    # each category of loads' bus no.
    category_i = array([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]) - 1  # bus number, convert to 1-base to 0-base
    category_c = array([9, 10, 11, 12, 13, 14, 15, 16, 17, 18]) - 1
    category_r = setdiff1d(arange(n_bus), hstack((category_i, category_c)))

    # numpy.in1d(ar1, ar2, assume_unique=False, invert=False) -
    # Test whether each element of a 1-D array ar1 is also present in a second array ar2.
    load.loc[in1d(load['bus'], category_i), 'load_type'] = 'industrial'
    load.loc[in1d(load['bus'], category_c), 'load_type'] = 'commercial'
    load.loc[in1d(load['bus'], category_r), 'load_type'] = 'residential'

    return ppnet