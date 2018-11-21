


def intersect1d_mtlb(a, b):
    """ This only works for 1-D input array
    a = np.array([7, 1, 7, 7, 4]);
    b = np.array([7, 0, 4, 4, 0]);
    c, ia, ib = intersect_mtlb(a, b)
    print(c, ia, ib)
    clipped from https://stackoverflow.com/questions/45637778/how-to-find-intersect-indexes-and-values-in-python?noredirect=1&lq=1
    :param a:
    :param b:
    :return:
    """
    import numpy as np
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]

def intersect_rows(array_a, array_b):
    """
    Intersect rows across two 2D numpy arrays by using sets
    clipped from https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    :param array_a:
    :param array_b:
    :return:
    """
    import numpy as np

    array_c = np.array([x for x in
                        set(tuple(x) for x in array_a) &
                        set(tuple(x) for x in array_b)
                    ])

    return array_c

def intersect_rows_2(array_a, array_b):
    """
    Intersect rows across two 2D numpy arrays by using np.intersect1d
    clipped from https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    :param array_a:
    :param array_b:
    :return: array_c: an array of common rows
    """
    import numpy as np

    nrows, ncols = array_a.shape
    # For structured arrays
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [array_a.dtype]}
    array_c = np.intersect1d(array_a.view(dtype), array_b.view(dtype))
    # This last bit is optional if you're okay with "C" being a structured array...
    array_c = array_c.view(array_a.dtype).reshape(-1, ncols)

    return array_c

# if __name__ == "__main__":